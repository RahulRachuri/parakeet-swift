//
//  Adapted from FluidAudio (https://github.com/FluidInference/FluidAudio),
//  upstream commit 667181a, file
//  Sources/FluidAudio/ASR/Parakeet/SlidingWindow/CustomVocabulary/WordSpotting/CtcModels.swift
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
//  Changes from upstream: every download / model-hub path is removed. This host
//  loads an already-present directory and nothing else.
//

import CoreML
import Foundation

/// The two plain Core ML bundles of the CTC verifier, plus its vocabulary.
///
/// Note the asymmetry with the rest of this package: the TDT graphs are Core AI
/// bundles, this one is stock Core ML. The verifier is a fixed-shape, run-once-
/// per-window model with no host-side decode loop, so there is nothing for a
/// Core AI export to buy; loading the published `.mlmodelc` directly also keeps
/// the log-probabilities bit-comparable with the reference implementation this
/// was ported from.
public struct CtcModels: @unchecked Sendable {
    public let melSpectrogram: MLModel
    public let encoder: MLModel
    /// id → piece, straight out of `vocab.json`. Only its *count* is load
    /// bearing (it fixes the blank id), but keeping the table makes a
    /// mis-pointed directory fail loudly instead of silently mis-scoring.
    public let vocabulary: [Int: String]

    /// The CTC blank sits one past the last real token.
    public var blankId: Int { vocabulary.count }

    private static let logger = RescoreLog("CtcModels")

    /// Compute-unit choice for the verifier.
    ///
    /// `cpuAndNeuralEngine` matches the reference implementation. It matters:
    /// the head's log-probabilities are what every replacement decision is
    /// compared against, and a different unit is a different fp16 rounding
    /// story, so this is a reproducibility setting rather than a speed one.
    public static func defaultConfiguration() -> MLModelConfiguration {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        return config
    }

    /// Load `MelSpectrogram.mlmodelc`, `AudioEncoder.mlmodelc` and `vocab.json`
    /// from `directory`.
    public static func load(from directory: URL) async throws -> CtcModels {
        let config = defaultConfiguration()

        func loadModel(named name: String) async throws -> MLModel {
            let path = directory.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: path.path) else {
                throw GraphError.message("ctc model: \(name) not found under \(directory.lastPathComponent)")
            }
            do {
                return try MLModel(contentsOf: path, configuration: config)
            } catch {
                // An uncompiled `.mlpackage`-style directory can still be made
                // to work; compiling is the documented fallback.
                let compiled = try await MLModel.compileModel(at: path)
                return try MLModel(contentsOf: compiled, configuration: config)
            }
        }

        let mel = try await loadModel(named: "MelSpectrogram.mlmodelc")
        let encoder = try await loadModel(named: "AudioEncoder.mlmodelc")
        let vocabulary = try loadVocabulary(from: directory)
        logger.info("loaded CTC verifier (\(vocabulary.count) vocab tokens, blank id \(vocabulary.count))")
        return CtcModels(melSpectrogram: mel, encoder: encoder, vocabulary: vocabulary)
    }

    private static func loadVocabulary(from directory: URL) throws -> [Int: String] {
        let vocabPath = directory.appendingPathComponent("vocab.json")
        guard FileManager.default.fileExists(atPath: vocabPath.path) else {
            throw GraphError.message("ctc model: vocab.json not found under \(directory.lastPathComponent)")
        }
        let data = try Data(contentsOf: vocabPath)
        let jsonDict = try JSONSerialization.jsonObject(with: data) as? [String: String] ?? [:]
        var vocabulary: [Int: String] = [:]
        for (key, value) in jsonDict {
            if let tokenId = Int(key) { vocabulary[tokenId] = value }
        }
        guard !vocabulary.isEmpty else {
            throw GraphError.message("ctc model: vocab.json parsed to zero tokens")
        }
        return vocabulary
    }
}
