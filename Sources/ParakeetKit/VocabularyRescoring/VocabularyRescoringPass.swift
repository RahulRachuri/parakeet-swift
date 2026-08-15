import Foundation

/// The custom-vocabulary rescoring pass: everything the host needs to correct a
/// finished transcript against a list of terms, behind one type.
///
/// **Where this sits.** It is a *post-pass*. The TDT decode runs to completion,
/// untouched, and only then does a second, independent CTC model get asked
/// whether any of its words should have been a vocabulary term. Nothing here
/// can reach back into the decode loop, which is deliberate: the shipping
/// transcription path stays byte-identical when no vocabulary is supplied, and
/// the parity gate keeps testing the same arithmetic it always did.
///
/// **Why a second model at all.** Proper nouns are exactly what a general ASR
/// model has never seen: a name like a fantasy novel's invented terms comes out
/// as whatever real word it sounds closest to. Fixing that by string similarity
/// alone rewrites every near-miss, including the correct ones. So the term and
/// the word already in the transcript are both scored against the same CTC
/// frames, and the term wins only on acoustic evidence plus a fixed boost.
public struct VocabularyRescoringPass: Sendable {

    /// Everything the pass is configured with, parsed from CLI flags.
    public struct Options: Sendable {
        /// Vocabulary file: JSON config, or one term per line.
        public var vocabularyPath: String
        /// Directory holding `MelSpectrogram.mlmodelc`, `AudioEncoder.mlmodelc`,
        /// `vocab.json` and `tokenizer.json`.
        public var ctcModelDirectory: URL
        /// Overrides the size-aware similarity gate when set.
        public var minSimilarity: Float?
        /// Overrides the size-aware context-biasing weight when set.
        public var cbw: Float?
        /// Overrides the frame-alignment margin when set.
        public var marginSeconds: Double?
        /// Short-term cbw taper pivot; `nil` keeps the default (disabled).
        public var shortTermTaperPivot: Int?
        /// Similarity floor for single-word acoustic rescues.
        public var spotterMinSimilarity: Float?
        /// Similarity floor for multi-word acoustic rescues.
        public var spotterMinSimilarityMultiWord: Float?
        /// Turn the acoustic rescue pass off entirely.
        public var disableSpotterRescue: Bool

        public init(
            vocabularyPath: String,
            ctcModelDirectory: URL,
            minSimilarity: Float? = nil,
            cbw: Float? = nil,
            marginSeconds: Double? = nil,
            shortTermTaperPivot: Int? = nil,
            spotterMinSimilarity: Float? = nil,
            spotterMinSimilarityMultiWord: Float? = nil,
            disableSpotterRescue: Bool = false
        ) {
            self.vocabularyPath = vocabularyPath
            self.ctcModelDirectory = ctcModelDirectory
            self.minSimilarity = minSimilarity
            self.cbw = cbw
            self.marginSeconds = marginSeconds
            self.shortTermTaperPivot = shortTermTaperPivot
            self.spotterMinSimilarity = spotterMinSimilarity
            self.spotterMinSimilarityMultiWord = spotterMinSimilarityMultiWord
            self.disableSpotterRescue = disableSpotterRescue
        }
    }

    /// What the pass did, for the caller to report.
    public struct Outcome: Sendable {
        /// The rewritten transcript.
        public var text: String
        /// Every applied replacement, as `(original, replacement)`.
        public var replacements: [(original: String, replacement: String)]
        /// Vocabulary terms loaded.
        public var termCount: Int
        /// CTC frames the verifier produced for the whole file.
        public var frameCount: Int
        /// Wall time spent computing CTC log-probabilities.
        public var ctcSeconds: Double
        /// Wall time spent in the rescoring pass itself.
        public var rescoreSeconds: Double
    }

    let rescorer: VocabularyRescorer
    let spotter: CtcKeywordSpotter
    let vocabulary: CustomVocabularyContext
    let minSimilarity: Float
    let cbw: Float
    let marginSeconds: Double

    private static let logger = RescoreLog("VocabularyRescoringPass")

    /// Load the vocabulary and the CTC verifier, and build the rescorer.
    ///
    /// Threshold precedence, strongest first: the `--vocab-min-similarity`
    /// flag, then a `minSimilarity` written in the JSON config, then the
    /// size-aware default. (The reference CLI skips the middle rung, so a JSON
    /// file could name a threshold and be quietly ignored.) Per-term
    /// `minSimilarity` values are applied inside the rescorer and outrank all
    /// three for the term that carries one.
    public static func load(options: Options) async throws -> VocabularyRescoringPass {
        let models = try await CtcModels.load(from: options.ctcModelDirectory)
        let tokenizer = try CtcTokenizer.load(from: options.ctcModelDirectory)
        let vocabulary = try CustomVocabularyContext.loadWithCtcTokens(
            from: options.vocabularyPath, tokenizer: tokenizer)
        guard !vocabulary.terms.isEmpty else {
            throw GraphError.message("custom vocabulary: no usable terms in \(options.vocabularyPath)")
        }

        let spotter = CtcKeywordSpotter(models: models)
        let sizeConfig = ContextBiasingConstants.rescorerConfig(forVocabSize: vocabulary.terms.count)

        let config = VocabularyRescorer.Config(
            shortTermCbwTaperPivot: options.shortTermTaperPivot
                ?? ContextBiasingConstants.defaultShortTermCbwTaperPivot,
            spotterRescueMinSimilarity: options.spotterMinSimilarity
                ?? ContextBiasingConstants.defaultSpotterRescueMinSimilarity,
            spotterRescueMultiWordMinSimilarity: options.spotterMinSimilarityMultiWord
                ?? ContextBiasingConstants.defaultSpotterRescueMultiWordMinSimilarity,
            spotterRescueEnabled: options.disableSpotterRescue
                ? false : ContextBiasingConstants.defaultSpotterRescueEnabled
        )

        let rescorer = VocabularyRescorer.create(
            spotter: spotter, vocabulary: vocabulary, config: config, tokenizer: tokenizer)

        return VocabularyRescoringPass(
            rescorer: rescorer,
            spotter: spotter,
            vocabulary: vocabulary,
            minSimilarity: options.minSimilarity
                ?? vocabulary.explicitMinSimilarity
                ?? sizeConfig.minSimilarity,
            cbw: options.cbw ?? sizeConfig.cbw,
            marginSeconds: options.marginSeconds ?? ContextBiasingConstants.defaultMarginSeconds
        )
    }

    /// Rescore a finished transcript.
    ///
    /// - Parameters:
    ///   - words: every word of the transcript, in order, with the absolute
    ///     times the decode assigned it.
    ///   - sampleCount: total 16 kHz samples in the file.
    ///   - read: `[from, to)` samples on demand. Windowed rather than handed a
    ///     whole `[Float]` because a chapter's worth of audio is a few hundred
    ///     megabytes, and the CTC front end only ever looks at 15 s at a time.
    public func run(
        words: [WordTimestamp],
        sampleCount: Int,
        read: (Int, Int) -> [Float]
    ) async throws -> Outcome {
        let tCtc = Date()
        let ctc = try await spotter.computeLogProbs(sampleCount: sampleCount, read: read)
        let ctcSeconds = -tCtc.timeIntervalSinceNow

        guard !ctc.logProbs.isEmpty, ctc.frameDuration > 0 else {
            throw GraphError.message("custom vocabulary: CTC verifier produced no frames")
        }

        let timings = Self.wordTimings(from: words)
        let tRescore = Date()
        let output = rescorer.ctcTokenRescore(
            wordTimings: timings,
            logProbs: ctc.logProbs,
            frameDuration: ctc.frameDuration,
            cbw: cbw,
            marginSeconds: marginSeconds,
            minSimilarity: minSimilarity
        )
        let rescoreSeconds = -tRescore.timeIntervalSinceNow

        return Outcome(
            text: output.text,
            replacements: output.replacements
                .filter { $0.shouldReplace }
                .map { ($0.originalWord, $0.replacementWord ?? "") },
            termCount: vocabulary.terms.count,
            frameCount: ctc.totalFrames,
            ctcSeconds: ctcSeconds,
            rescoreSeconds: rescoreSeconds
        )
    }

    /// Convert this package's word timestamps into the rescorer's view of them.
    ///
    /// Two corrections, both of which exist to line the transcript up with
    /// where the CTC head sees each word:
    ///
    /// 1. **Emission delay.** A TDT decode emits a token about one encoder
    ///    frame *after* the acoustic event that caused it, so every time is
    ///    shifted back by one frame (80 ms). Without this the search window
    ///    trails the word it is supposed to cover, and with the margin down at
    ///    100 ms that is most of the slack spent on a fixed, known offset.
    ///
    /// 2. **End times.** This package ends a word at the next word's start, and
    ///    ends the *last* word of a chunk at the chunk boundary — which can be
    ///    seconds of trailing silence. Rebuilding every end from the following
    ///    word's start (and giving the final word a single frame) keeps the
    ///    windows tight and matches the reference implementation, which reads
    ///    its end times off the next token in one continuous stream.
    ///
    /// The one place this is approximate: a word starting at frame 0 of a chunk
    /// shifts 80 ms into the previous chunk rather than clamping at the chunk
    /// head. That is inside the alignment margin, and it only affects the first
    /// word of a chunk.
    static func wordTimings(from words: [WordTimestamp]) -> [VocabularyRescorer.WordTiming] {
        let frame = WordTimestamps.secondsPerFrame
        func shifted(_ t: Double) -> Double { max(0, t - frame) }

        return words.indices.map { i in
            let start = shifted(words[i].start)
            let end =
                i + 1 < words.count
                ? max(start + frame, shifted(words[i + 1].start))
                : start + frame
            return VocabularyRescorer.WordTiming(word: words[i].word, startTime: start, endTime: end)
        }
    }
}

extension CtcKeywordSpotter {
    /// Whole-file CTC log-probabilities, reading the audio in windows.
    ///
    /// Same windowing and same overlap merge as the in-memory path; the only
    /// difference is that each window's samples are fetched when that window is
    /// about to run instead of the file being materialized up front.
    public func computeLogProbs(
        sampleCount: Int,
        read: (Int, Int) -> [Float]
    ) async throws -> CtcLogProbResult {
        guard sampleCount > 0 else {
            return CtcLogProbResult(logProbs: [], frameDuration: 0, totalFrames: 0, audioSamplesUsed: 0)
        }
        if sampleCount <= maxModelSamples {
            return try await computeLogProbs(for: read(0, sampleCount))
        }

        let stride = maxModelSamples - chunkOverlapSamples
        var windows: [(start: Int, end: Int)] = []
        var start = 0
        while start < sampleCount {
            let end = min(start + maxModelSamples, sampleCount)
            windows.append((start: start, end: end))
            if end >= sampleCount { break }
            start += stride
        }

        var frameDuration = 0.0
        var overlapFrames = 0
        var concatenated: [[Float]] = []

        for (index, window) in windows.enumerated() {
            let result = try await computeLogProbs(for: read(window.start, window.end))
            let logProbs = result.logProbs
            guard !logProbs.isEmpty else { continue }

            if index == 0 {
                guard result.frameDuration > 0 else { break }
                frameDuration = result.frameDuration
                overlapFrames = Int(
                    Double(chunkOverlapSamples) / Double(sampleRate) / frameDuration)
                concatenated.reserveCapacity(windows.count * logProbs.count)
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

        guard frameDuration > 0 else {
            return CtcLogProbResult(logProbs: [], frameDuration: 0, totalFrames: 0, audioSamplesUsed: 0)
        }

        return CtcLogProbResult(
            logProbs: concatenated,
            frameDuration: frameDuration,
            totalFrames: concatenated.count,
            audioSamplesUsed: sampleCount
        )
    }
}
