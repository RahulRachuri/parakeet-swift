//
//  Adapted from FluidAudio (https://github.com/FluidInference/FluidAudio),
//  upstream commit 667181a, files
//  Sources/FluidAudio/ASR/Parakeet/SlidingWindow/CustomVocabulary/WordSpotting/BpeTokenizer.swift
//  and .../WordSpotting/CtcTokenizer.swift (merged).
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
//  Changes from upstream: the two type declarations are merged into one file
//  and the default-cache-directory lookup is dropped (the directory is always
//  passed explicitly here).
//

import Foundation

/// BPE encoder for the CTC verifier's vocabulary.
///
/// Encode only: the rescorer turns text into CTC token IDs and never decodes.
/// Text is lowercased and NFKC-normalized before encoding, matching the NeMo
/// CTC tokenization pipeline.
public final class BpeTokenizer: Sendable {
    private let vocab: [String: Int]
    private let merges: [String: Int]
    private let addedTokens: [String: Int]

    public enum Error: Swift.Error, LocalizedError {
        case fileNotFound(String)
        case invalidJSON(String)
        case missingField(String)
        case unsupportedTokenizerType(String)

        public var errorDescription: String? {
            switch self {
            case .fileNotFound(let name):
                return "tokenizer.json not found (\(name))"
            case .invalidJSON(let message):
                return "invalid JSON: \(message)"
            case .missingField(let field):
                return "missing required field: \(field)"
            case .unsupportedTokenizerType(let type):
                return "unsupported tokenizer type: \(type); only 'BPE' is supported"
            }
        }
    }

    /// Load from a folder containing `tokenizer.json`.
    public static func load(from modelFolder: URL) throws -> BpeTokenizer {
        let tokenizerPath = modelFolder.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: tokenizerPath.path) else {
            throw Error.fileNotFound(tokenizerPath.lastPathComponent)
        }

        let data = try Data(contentsOf: tokenizerPath)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw Error.invalidJSON("root is not a dictionary")
        }
        guard let model = json["model"] as? [String: Any] else {
            throw Error.missingField("model")
        }
        guard let modelType = model["type"] as? String else {
            throw Error.missingField("model.type")
        }
        guard modelType == "BPE" else {
            throw Error.unsupportedTokenizerType(modelType)
        }
        guard let vocabDict = model["vocab"] as? [String: Int] else {
            throw Error.missingField("model.vocab")
        }
        guard let mergesArray = model["merges"] as? [String] else {
            throw Error.missingField("model.merges")
        }

        // Upstream stores merges as an array and finds a pair's rank with a
        // linear `firstIndex(where:)` inside the inner BPE loop. Same ranks,
        // same winner, but as a dictionary keyed by "a\u{0}b" the lookup is
        // O(1); on a 58-minute chapter the linear form dominated the profile.
        var mergeRanks: [String: Int] = [:]
        mergeRanks.reserveCapacity(mergesArray.count)
        for (rank, mergeStr) in mergesArray.enumerated() {
            let parts = mergeStr.split(separator: " ", maxSplits: 1)
            guard parts.count == 2 else { continue }
            let key = String(parts[0]) + "\u{0}" + String(parts[1])
            // First occurrence wins, matching `firstIndex(where:)`.
            if mergeRanks[key] == nil { mergeRanks[key] = rank }
        }

        var addedTokensDict: [String: Int] = [:]
        let addedTokensList = (json["added_tokens"] as? [[String: Any]]) ?? []
        for token in addedTokensList {
            guard let content = token["content"] as? String,
                let id = token["id"] as? Int
            else { continue }
            addedTokensDict[content] = id
        }

        return BpeTokenizer(vocab: vocabDict, merges: mergeRanks, addedTokens: addedTokensDict)
    }

    private init(vocab: [String: Int], merges: [String: Int], addedTokens: [String: Int]) {
        self.vocab = vocab
        self.merges = merges
        self.addedTokens = addedTokens
    }

    /// Encode text to token IDs with BPE.
    ///
    /// - Parameter prependWordBoundary: when `true` (the default) a leading
    ///   SentencePiece `▁` is prepended, matching standard NeMo CTC
    ///   tokenization. Set `false` for a mid-utterance tokenization, useful
    ///   when a keyword does not begin at a word boundary in the audio.
    public func encode(
        _ text: String,
        prependWordBoundary: Bool = true
    ) -> [Int] {
        let normalized = text.lowercased().precomposedStringWithCompatibilityMapping
        let boundary = ContextBiasingConstants.sentencePieceWordBoundary
        let leading = prependWordBoundary ? boundary : ""
        let preprocessed = leading + normalized.replacingOccurrences(of: " ", with: boundary)

        var word = preprocessed.map { String($0) }

        while word.count >= 2 {
            var bestRank: Int? = nil
            var bestPair: (String, String)? = nil

            for i in 0..<word.count - 1 {
                guard let rank = merges[word[i] + "\u{0}" + word[i + 1]] else { continue }
                if bestRank.map({ rank < $0 }) ?? true {
                    bestRank = rank
                    bestPair = (word[i], word[i + 1])
                }
            }

            guard let (first, second) = bestPair else { break }

            // Apply the winning merge to every occurrence (standard BPE).
            var newWord: [String] = []
            var i = 0
            while i < word.count {
                if i < word.count - 1 && word[i] == first && word[i + 1] == second {
                    newWord.append(first + second)
                    i += 2
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }
            word = newWord
        }

        return word.compactMap { token -> Int? in
            if let id = addedTokens[token] { return id }
            if let id = vocab[token] { return id }
            return addedTokens["<unk>"] ?? vocab["<unk>"] ?? 0
        }
    }
}

/// CTC tokenizer, driven off the verifier model's `tokenizer.json`.
public final class CtcTokenizer: Sendable {
    private let bpeTokenizer: BpeTokenizer

    /// Load from a directory containing `tokenizer.json`.
    public static func load(from modelDirectory: URL) throws -> CtcTokenizer {
        CtcTokenizer(bpeTokenizer: try BpeTokenizer.load(from: modelDirectory))
    }

    private init(bpeTokenizer: BpeTokenizer) {
        self.bpeTokenizer = bpeTokenizer
    }

    /// Tokenize text into CTC token IDs, with the leading word boundary.
    public func encode(_ text: String) -> [Int] {
        bpeTokenizer.encode(text)
    }

    /// Tokenize without the leading `▁`. A keyword that does not start at a
    /// word boundary in the audio has no acoustic counterpart for that token,
    /// and scoring it penalizes the alignment unfairly.
    public func encodeWithoutBoundary(_ text: String) -> [Int] {
        bpeTokenizer.encode(text, prependWordBoundary: false)
    }
}
