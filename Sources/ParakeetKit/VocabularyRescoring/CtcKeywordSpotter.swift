//
//  Adapted from FluidAudio (https://github.com/FluidInference/FluidAudio),
//  upstream commit 667181a, files
//  Sources/FluidAudio/ASR/Parakeet/SlidingWindow/CustomVocabulary/WordSpotting/CtcKeywordSpotter.swift
//  and .../WordSpotting/CtcKeywordSpotter+Inference.swift (merged).
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//  Changes from upstream: the two files are merged; per-element `MLMultiArray`
//  subscripting is replaced by typed buffer access (same values, see the note
//  on `readLogits`); and the caller may ask for log-probabilities alone,
//  skipping a detection pass whose results it would discard.
//

import CoreML
import Foundation

/// CTC keyword spotting for the `parakeet-ctc-110m` verifier.
///
/// Runs the MelSpectrogram + AudioEncoder Core ML bundles, turns the CTC head's
/// logits into per-frame log-probabilities, and scores each keyword
/// independently with the NeMo dynamic program (no beam search, no competition
/// between keywords).
public struct CtcKeywordSpotter: Sendable {

    let logger = RescoreLog("CtcKeywordSpotter")
    let models: CtcModels
    public let blankId: Int

    let sampleRate = ContextBiasingConstants.sampleRate
    let maxModelSamples = ContextBiasingConstants.maxModelSamples
    /// Overlap between consecutive CTC windows on long audio.
    let chunkOverlapSamples = ContextBiasingConstants.chunkOverlapSamples

    let temperature = ContextBiasingConstants.ctcTemperature
    let blankBias = ContextBiasingConstants.blankBias
    var debugMode: Bool { ContextBiasingConstants.debugEnabled }

    /// Computed rather than stored: `MLPredictionOptions` is not `Sendable`,
    /// and building one is an allocation and an empty dictionary.
    var predictionOptions: MLPredictionOptions {
        let options = MLPredictionOptions()
        options.outputBackings = [:]
        return options
    }

    public init(models: CtcModels, blankId: Int? = nil) {
        self.models = models
        self.blankId = blankId ?? models.blankId
    }

    // MARK: - Result types

    /// Whole-file CTC log-probabilities plus the frame grid they sit on.
    public struct CtcLogProbResult: Sendable {
        public let logProbs: [[Float]]
        public let frameDuration: Double
        public let totalFrames: Int
        public let audioSamplesUsed: Int
    }

    /// One keyword detection.
    public struct KeywordDetection: Sendable {
        public let term: CustomVocabularyTerm
        public let score: Float
        public let totalFrames: Int
        public let startFrame: Int
        public let endFrame: Int
        public let startTime: TimeInterval
        public let endTime: TimeInterval
    }

    /// Detections plus the log-probabilities they were found in.
    public struct SpotKeywordsResult: Sendable {
        public let detections: [KeywordDetection]
        public let logProbs: [[Float]]
        public let frameDuration: Double
        public let totalFrames: Int
    }

    // MARK: - Public API

    /// Spot keywords in pre-computed log-probabilities. No model inference.
    public func spotKeywordsFromLogProbs(
        logProbs: [[Float]],
        frameDuration: Double,
        customVocabulary: CustomVocabularyContext,
        minScore: Float? = nil
    ) -> SpotKeywordsResult {
        let totalFrames = logProbs.count
        guard totalFrames > 0 else {
            return SpotKeywordsResult(detections: [], logProbs: [], frameDuration: 0, totalFrames: 0)
        }

        var results: [KeywordDetection] = []

        for term in customVocabulary.terms {
            // Short terms are skipped outright, per the NeMo CTC-WS paper's
            // false-positive guidance.
            guard term.text.count >= customVocabulary.minTermLength else {
                logger.debug("  skipping '\(term.text)': shorter than \(customVocabulary.minTermLength) chars")
                continue
            }
            guard let ids = term.ctcTokenIds, !ids.isEmpty else { continue }

            // Longer phrases accumulate lower per-token scores, so relax the
            // threshold for each token past the baseline.
            let adjustedThreshold: Float =
                minScore.map { base in
                    let extraTokens = max(
                        0, ids.count - ContextBiasingConstants.baselineTokenCountForThreshold)
                    return base - Float(extraTokens) * ContextBiasingConstants.thresholdRelaxationPerToken
                } ?? ContextBiasingConstants.defaultMinSpotterScore

            let detections = CtcDPAlgorithm.ctcWordSpotMultiple(
                logProbs: logProbs,
                keywordTokens: ids,
                minScore: adjustedThreshold,
                mergeOverlap: true,
                blankId: blankId
            )

            for (score, start, end) in detections {
                results.append(
                    KeywordDetection(
                        term: term,
                        score: score,
                        totalFrames: totalFrames,
                        startFrame: start,
                        endFrame: end,
                        startTime: TimeInterval(start) * frameDuration,
                        endTime: TimeInterval(end) * frameDuration
                    ))
            }
        }

        return SpotKeywordsResult(
            detections: results, logProbs: logProbs,
            frameDuration: frameDuration, totalFrames: totalFrames)
    }

    /// Constrained spotting inside one frame window.
    func ctcWordSpotConstrained(
        logProbs: [[Float]],
        keywordTokens: [Int],
        searchStartFrame: Int,
        searchEndFrame: Int
    ) -> (score: Float, startFrame: Int, endFrame: Int) {
        CtcDPAlgorithm.ctcWordSpotConstrained(
            logProbs: logProbs,
            keywordTokens: keywordTokens,
            searchStartFrame: searchStartFrame,
            searchEndFrame: searchEndFrame,
            blankId: blankId
        )
    }

    // MARK: - Inference

    /// Whole-file CTC log-probabilities `[T, V]`.
    ///
    /// Audio longer than the model's fixed 15 s input is processed in
    /// overlapping windows and stitched back into one matrix, so the caller
    /// sees a single continuous frame timeline.
    public func computeLogProbs(for audioSamples: [Float]) async throws -> CtcLogProbResult {
        guard !audioSamples.isEmpty else {
            return CtcLogProbResult(logProbs: [], frameDuration: 0, totalFrames: 0, audioSamplesUsed: 0)
        }
        if audioSamples.count > maxModelSamples {
            return try await computeLogProbsChunked(audioSamples: audioSamples)
        }
        return try await computeWithStagedModels(audioSamples: audioSamples)
    }

    /// Window long audio, run each window, and concatenate the log-probs,
    /// averaging the overlap region in probability space.
    private func computeLogProbsChunked(audioSamples: [Float]) async throws -> CtcLogProbResult {
        let totalSamples = audioSamples.count
        let chunkSize = maxModelSamples
        let overlap = chunkOverlapSamples
        let stride = chunkSize - overlap

        var windows: [(start: Int, end: Int)] = []
        var start = 0
        while start < totalSamples {
            let end = min(start + chunkSize, totalSamples)
            windows.append((start: start, end: end))
            if end >= totalSamples { break }
            start += stride
        }

        logger.debug(
            "CTC windows: \(windows.count) over \(String(format: "%.1f", Double(totalSamples) / Double(sampleRate)))s "
                + "(size \(chunkSize), overlap \(overlap))")

        var windowResults: [CtcLogProbResult] = []
        windowResults.reserveCapacity(windows.count)
        for window in windows {
            let windowAudio = Array(audioSamples[window.start..<window.end])
            windowResults.append(try await computeWithStagedModels(audioSamples: windowAudio))
        }

        guard let first = windowResults.first, first.frameDuration > 0 else {
            return CtcLogProbResult(logProbs: [], frameDuration: 0, totalFrames: 0, audioSamplesUsed: 0)
        }
        let frameDuration = first.frameDuration
        let overlapFrames = Int(Double(overlap) / Double(sampleRate) / frameDuration)

        var concatenated: [[Float]] = []
        concatenated.reserveCapacity(windows.count * first.totalFrames)

        for (index, result) in windowResults.enumerated() {
            let logProbs = result.logProbs
            guard !logProbs.isEmpty else { continue }

            if index == 0 {
                concatenated.append(contentsOf: logProbs)
                continue
            }

            let overlapCount = min(overlapFrames, concatenated.count, logProbs.count)
            if overlapCount > 0 {
                let existingStart = concatenated.count - overlapCount
                for i in 0..<overlapCount {
                    concatenated[existingStart + i] = Self.mergeOverlapFrame(
                        existing: concatenated[existingStart + i], incoming: logProbs[i])
                }
            }
            if overlapCount < logProbs.count {
                concatenated.append(contentsOf: logProbs.suffix(from: overlapCount))
            }
        }

        logger.debug("CTC concatenated to \(concatenated.count) frames (\(overlapFrames) averaged per boundary)")

        return CtcLogProbResult(
            logProbs: concatenated,
            frameDuration: frameDuration,
            totalFrames: concatenated.count,
            audioSamplesUsed: totalSamples
        )
    }

    /// One window: mel bundle → encoder bundle → log-probabilities.
    private func computeWithStagedModels(audioSamples: [Float]) async throws -> CtcLogProbResult {
        let (audioInput, clampedCount) = try prepareAudioArray(audioSamples)
        let melInput = try makeAudioFeatureProvider(array: audioInput, length: clampedCount)

        let melOutput = try await models.melSpectrogram.prediction(
            from: melInput, options: predictionOptions)

        guard let melFeatures = melOutput.featureValue(for: "melspectrogram_features")?.multiArrayValue else {
            throw GraphError.message("ctc: mel bundle returned no melspectrogram_features")
        }

        // Prefer an explicit mel_length; otherwise read the frames axis. For the
        // rank-4 [1, 1, frames, bins] layout the frames axis is index 2, not the
        // last one.
        var melLengthValue =
            melOutput.featureValue(for: "mel_length")?.multiArrayValue?[0].intValue
            ?? melFeatures.shape.last?.intValue
        if melFeatures.shape.count == 4 {
            melLengthValue = melFeatures.shape[2].intValue
        }

        let encoderInput = try makeEncoderInput(melFeatures: melFeatures, melLength: melLengthValue)
        let encoderOutput = try await models.encoder.prediction(
            from: encoderInput, options: predictionOptions)

        // Use `ctc_head_raw_output` (raw logits). `ctc_head_output` is already
        // post-softmax, and pushing it through log-softmax a second time yields
        // nonsense scores.
        guard
            let ctcRaw =
                encoderOutput.featureValue(for: "ctc_head_raw_output")?.multiArrayValue
                ?? encoderOutput.featureValue(for: "ctc_head_output")?.multiArrayValue
        else {
            throw GraphError.message("ctc: encoder returned no ctc head output")
        }

        let allLogProbs = try makeLogProbs(from: ctcRaw, temperature: temperature, blankBias: blankBias)
        let trimmed = trimLogProbs(allLogProbs, audioSampleCount: clampedCount)
        let frameCount = trimmed.count

        let frameDuration =
            frameCount > 0 ? Double(clampedCount) / Double(frameCount) / Double(sampleRate) : 0

        return CtcLogProbResult(
            logProbs: trimmed,
            frameDuration: frameDuration,
            totalFrames: frameCount,
            audioSamplesUsed: clampedCount
        )
    }

    // MARK: - Audio preparation

    /// Build the fixed-length audio array the mel bundle expects.
    ///
    /// The tail past `clampedCount` stays at the zero `MLMultiArray` starts
    /// with. That is deliberate and load-bearing: the reference implementation
    /// zero-pads here, and swapping in any other padding (silence at a DC
    /// offset, edge repetition) moves the mel statistics and therefore every
    /// log-probability in the final window.
    private func prepareAudioArray(_ audioSamples: [Float]) throws -> (MLMultiArray, Int) {
        let clampedCount = min(audioSamples.count, maxModelSamples)

        let audioDesc = models.melSpectrogram.modelDescription.inputDescriptionsByName["audio"]
        let expectedRank = audioDesc?.multiArrayConstraint?.shape.count ?? 1
        let dataType: MLMultiArrayDataType =
            audioDesc?.multiArrayConstraint?.dataType == .float16 ? .float16 : .float32

        let shape: [NSNumber] =
            expectedRank == 2
            ? [1, NSNumber(value: maxModelSamples)]
            : [NSNumber(value: maxModelSamples)]
        let array = try MLMultiArray(shape: shape, dataType: dataType)

        // Bulk-write through the typed buffer. Upstream assigns element by
        // element through `NSNumber`; at 240 000 samples per window and one
        // window every 13 s of audio that boxing is most of the wall clock,
        // and the stored values are identical either way.
        switch dataType {
        case .float16:
            array.withUnsafeMutableBufferPointer(ofType: Float16.self) { buffer, _ in
                for i in 0..<clampedCount { buffer[i] = Float16(audioSamples[i]) }
            }
        default:
            array.withUnsafeMutableBufferPointer(ofType: Float.self) { buffer, _ in
                for i in 0..<clampedCount { buffer[i] = audioSamples[i] }
            }
        }

        return (array, clampedCount)
    }

    private func makeAudioFeatureProvider(array: MLMultiArray, length: Int) throws -> MLFeatureProvider {
        let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        lengthArray[0] = NSNumber(value: length)
        // `audio_length` is accepted by some exports and ignored by the rest;
        // Core ML drops features the model does not declare.
        return try MLDictionaryFeatureProvider(dictionary: [
            "audio": MLFeatureValue(multiArray: array),
            "audio_length": MLFeatureValue(multiArray: lengthArray),
        ])
    }

    private func makeEncoderInput(melFeatures: MLMultiArray, melLength: Int?) throws -> MLFeatureProvider {
        let lengthValue = melLength ?? melFeatures.shape.last?.intValue ?? 0
        guard lengthValue > 0 else {
            throw GraphError.message("ctc: invalid mel_length for the encoder input")
        }

        let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        lengthArray[0] = NSNumber(value: lengthValue)

        var dict: [String: MLFeatureValue] = [
            "melspectrogram_features": MLFeatureValue(multiArray: melFeatures),
            "mel_length": MLFeatureValue(multiArray: lengthArray),
        ]
        // A placeholder some staged exports require and others ignore.
        if let input1 = try? MLMultiArray(shape: [1, 1, 1, 1], dataType: .float16) {
            input1[0] = 1
            dict["input_1"] = MLFeatureValue(multiArray: input1)
        }

        return try MLDictionaryFeatureProvider(dictionary: dict)
    }

    // MARK: - Overlap merging

    /// Merge two overlapping frames by the mean **in probability space**, via a
    /// numerically stable log-mean-exp:
    /// `logmeanexp(a, b) = max(a, b) + log(exp(a - m) + exp(b - m)) - log 2`.
    ///
    /// Averaging the log-probabilities directly would be a geometric mean,
    /// which is not a valid distribution and systematically underweights the
    /// frames where the two windows disagree — exactly the frames at a seam.
    static func mergeOverlapFrame(existing: [Float], incoming: [Float]) -> [Float] {
        let v = min(existing.count, incoming.count)
        if v == 0 { return existing }

        let log2: Float = 0.693_147_18
        var averaged = [Float](repeating: 0, count: v)
        for j in 0..<v {
            let a = existing[j]
            let b = incoming[j]
            let m = max(a, b)
            averaged[j] = m == -Float.infinity ? -Float.infinity : m + logf(expf(a - m) + expf(b - m)) - log2
        }
        return averaged
    }

    // MARK: - Log-probability processing

    /// Logits → per-frame log-probabilities, with temperature and blank bias.
    private func makeLogProbs(
        from ctcOutput: MLMultiArray,
        temperature: Float,
        blankBias: Float
    ) throws -> [[Float]] {
        let rank = ctcOutput.shape.count
        guard rank == 3 || rank == 4 else {
            throw GraphError.message("ctc: unexpected head output rank \(ctcOutput.shape)")
        }

        let vocabSize: Int
        let timeSteps: Int
        let timeStride: Int
        let vocabStride: Int

        if rank == 3 {
            // [1, timeSteps, vocabSize]
            timeSteps = ctcOutput.shape[1].intValue
            vocabSize = ctcOutput.shape[2].intValue
            timeStride = ctcOutput.strides[1].intValue
            vocabStride = ctcOutput.strides[2].intValue
        } else {
            // [1, vocabSize, 1, timeSteps]
            vocabSize = ctcOutput.shape[1].intValue
            timeSteps = ctcOutput.shape[3].intValue
            vocabStride = ctcOutput.strides[1].intValue
            timeStride = ctcOutput.strides[3].intValue
        }

        if vocabSize <= 0 || timeSteps <= 0 { return [] }

        let logits = readLogits(
            from: ctcOutput, timeSteps: timeSteps, vocabSize: vocabSize,
            timeStride: timeStride, vocabStride: vocabStride)

        var logProbs: [[Float]] = []
        logProbs.reserveCapacity(timeSteps)
        for t in 0..<timeSteps {
            var row = logSoftmax(Array(logits[(t * vocabSize)..<((t + 1) * vocabSize)]), temperature: temperature)
            if blankBias != 0.0 && blankId < row.count {
                row[blankId] -= blankBias
            }
            logProbs.append(row)
        }
        return logProbs
    }

    /// Read the head output into a flat `[timeSteps * vocabSize]` Float array.
    ///
    /// Upstream reads each element through `ctcOutput[[0, v, 0, t]]`, which
    /// boxes two `NSNumber`s per element — ~193 000 elements per window. The
    /// typed buffer below produces the same Float values (a `Float16` widened
    /// to `Float` is exact) at a fraction of the cost; the `NSNumber` path
    /// remains as the fallback for any dtype the buffer accessor refuses.
    private func readLogits(
        from ctcOutput: MLMultiArray,
        timeSteps: Int,
        vocabSize: Int,
        timeStride: Int,
        vocabStride: Int
    ) -> [Float] {
        var out = [Float](repeating: 0, count: timeSteps * vocabSize)

        // `withUnsafeBufferPointer(ofType:)` traps on a type mismatch rather
        // than reporting one, so the dtype is switched on first and the boxed
        // fallback is reached only for a dtype with no float view at all.
        func fill<T: BinaryFloatingPoint & MLShapedArrayScalar>(_ type: T.Type) {
            ctcOutput.withUnsafeBufferPointer(ofType: type) { src in
                out.withUnsafeMutableBufferPointer { dst in
                    for t in 0..<timeSteps {
                        let base = t * timeStride
                        let rowBase = t * vocabSize
                        for v in 0..<vocabSize {
                            dst[rowBase + v] = Float(src[base + v * vocabStride])
                        }
                    }
                }
            }
        }

        switch ctcOutput.dataType {
        case .float16:
            fill(Float16.self)
        case .float32:
            fill(Float.self)
        case .double:
            fill(Double.self)
        default:
            let rank = ctcOutput.shape.count
            for t in 0..<timeSteps {
                for v in 0..<vocabSize {
                    let index: [NSNumber] =
                        rank == 3
                        ? [0, NSNumber(value: t), NSNumber(value: v)]
                        : [0, NSNumber(value: v), 0, NSNumber(value: t)]
                    out[t * vocabSize + v] = ctcOutput[index].floatValue
                }
            }
        }
        return out
    }

    private func logSoftmax(_ logits: [Float], temperature: Float) -> [Float] {
        guard !logits.isEmpty else { return [] }

        let scaled = temperature != 1.0 ? logits.map { $0 / temperature } : logits
        let maxLogit = scaled.max() ?? 0
        var sumExp: Float = 0
        for value in scaled { sumExp += expf(value - maxLogit) }
        let logSumExp = logf(sumExp)

        var result = [Float](repeating: 0, count: scaled.count)
        for i in 0..<scaled.count {
            result[i] = (scaled[i] - maxLogit) - logSumExp
        }
        return result
    }

    /// Drop the frames that only cover zero padding in a short final window.
    private func trimLogProbs(_ logProbs: [[Float]], audioSampleCount: Int) -> [[Float]] {
        guard !logProbs.isEmpty, audioSampleCount < maxModelSamples else { return logProbs }

        let totalFrames = logProbs.count
        let samplesPerFrame = Double(maxModelSamples) / Double(totalFrames)
        let validFrames = Int(ceil(Double(audioSampleCount) / samplesPerFrame))
        let clampedFrames = max(1, min(validFrames, totalFrames))
        return Array(logProbs.prefix(clampedFrames))
    }
}
