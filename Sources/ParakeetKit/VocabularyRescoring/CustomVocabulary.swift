//
//  Adapted from FluidAudio (https://github.com/FluidInference/FluidAudio),
//  upstream commit 667181a, file
//  Sources/FluidAudio/ASR/Parakeet/SlidingWindow/CustomVocabulary/CustomVocabularyContext.swift
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
//  Changes from upstream: the model-download convenience loader is replaced by
//  an explicit-directory one, since this host never downloads anything.
//

import Foundation

/// A single custom-vocabulary entry.
public struct CustomVocabularyTerm: Codable, Sendable {
    public let text: String
    public let weight: Float?
    public let aliases: [String]?
    /// Optional pre-tokenized CTC vocabulary IDs for this phrase.
    public let ctcTokenIds: [Int]?

    /// Per-term minimum string similarity, overriding the vocabulary-level
    /// value for this term only.
    ///
    /// This is the knob that splits a threshold two terms would otherwise have
    /// to share: a global gate is one number, so raising it to kill a false
    /// positive on one term also kills the true positive on another sitting at
    /// the same edit distance. Vocabulary-wide safety guards (short-word,
    /// stopword-span, length-ratio) still clamp upward on top of this, so a
    /// per-term override can only loosen matching down to those guards.
    public let minSimilarity: Float?

    /// Pre-computed lowercased text (not serialized).
    public let textLowercased: String

    private enum CodingKeys: String, CodingKey {
        case text, weight, aliases, ctcTokenIds, minSimilarity
    }

    public init(
        text: String,
        weight: Float? = nil,
        aliases: [String]? = nil,
        ctcTokenIds: [Int]? = nil,
        minSimilarity: Float? = nil
    ) {
        self.text = text
        self.weight = weight
        self.aliases = aliases
        self.ctcTokenIds = ctcTokenIds
        self.minSimilarity = minSimilarity.map { Self.clampSimilarity($0) }
        self.textLowercased = text.lowercased()
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        text = try container.decode(String.self, forKey: .text)
        weight = try container.decodeIfPresent(Float.self, forKey: .weight)
        aliases = try container.decodeIfPresent([String].self, forKey: .aliases)
        ctcTokenIds = try container.decodeIfPresent([Int].self, forKey: .ctcTokenIds)
        minSimilarity = try container.decodeIfPresent(Float.self, forKey: .minSimilarity)
            .map { Self.clampSimilarity($0) }
        textLowercased = text.lowercased()
    }

    /// Clamp into `[0, 1]` so a malformed config cannot disable or invert the gate.
    private static func clampSimilarity(_ value: Float) -> Float {
        min(1.0, max(0.0, value))
    }
}

/// On-disk JSON model for a structured vocabulary file.
struct CustomVocabularyConfig: Codable, Sendable {
    let alpha: Float?
    let terms: [CustomVocabularyTerm]
    let minCtcScore: Float?
    let minSimilarity: Float?
    let minCombinedConfidence: Float?
    /// Minimum character length for a term; shorter terms are skipped, per the
    /// NeMo CTC word-spotting paper's false-positive guidance.
    let minTermLength: Int?
}

/// Runtime vocabulary context handed to the rescorer.
public struct CustomVocabularyContext: Sendable {
    public let terms: [CustomVocabularyTerm]
    public let alpha: Float
    public let minCtcScore: Float
    public let minSimilarity: Float
    public let minCombinedConfidence: Float
    public let minTermLength: Int

    /// The vocabulary-level `minSimilarity` **as written in the file**, or nil
    /// when the file did not set one.
    ///
    /// `minSimilarity` above cannot answer that question: it has already
    /// collapsed "absent" into the default. The distinction matters because a
    /// threshold the author actually typed should outrank a size-derived guess,
    /// and the reference CLI drops the field on the floor — a JSON config could
    /// say 0.70 and silently run at 0.50.
    public let explicitMinSimilarity: Float?

    public init(
        terms: [CustomVocabularyTerm],
        alpha: Float = ContextBiasingConstants.defaultAlpha,
        minCtcScore: Float = ContextBiasingConstants.defaultMinVocabCtcScore,
        minSimilarity: Float = ContextBiasingConstants.defaultMinSimilarity,
        minCombinedConfidence: Float = ContextBiasingConstants.defaultMinCombinedConfidence,
        minTermLength: Int = 3,
        explicitMinSimilarity: Float? = nil
    ) {
        self.terms = terms
        self.alpha = alpha
        self.minCtcScore = minCtcScore
        self.minSimilarity = minSimilarity
        self.minCombinedConfidence = minCombinedConfidence
        self.minTermLength = minTermLength
        self.explicitMinSimilarity = explicitMinSimilarity
    }

    private static let logger = RescoreLog("CustomVocabulary")

    /// Load a structured JSON vocabulary config.
    public static func load(from url: URL) throws -> CustomVocabularyContext {
        let data = try Data(contentsOf: url)
        let config = try JSONDecoder().decode(CustomVocabularyConfig.self, from: data)

        let alpha = config.alpha ?? ContextBiasingConstants.defaultAlpha
        let minCtcScore = config.minCtcScore ?? ContextBiasingConstants.defaultMinVocabCtcScore
        let minSimilarity = config.minSimilarity ?? ContextBiasingConstants.defaultMinSimilarity
        let minCombinedConfidence =
            config.minCombinedConfidence ?? ContextBiasingConstants.defaultMinCombinedConfidence
        let minTermLength = config.minTermLength ?? 3

        var validatedTerms: [CustomVocabularyTerm] = []
        for term in config.terms {
            let (sanitized, warnings) = sanitizeVocabularyTerm(term.text)
            if !warnings.isEmpty {
                logger.warning("term '\(term.text)': \(warnings.joined(separator: ", "))")
            }
            guard !sanitized.isEmpty else {
                logger.warning("term '\(term.text)' is empty after sanitization, skipping")
                continue
            }
            let sanitizedAliases = term.aliases?.compactMap { alias -> String? in
                let (sanitizedAlias, _) = sanitizeVocabularyTerm(alias)
                return sanitizedAlias.isEmpty ? nil : sanitizedAlias
            }
            validatedTerms.append(
                CustomVocabularyTerm(
                    text: sanitized,
                    weight: term.weight,
                    aliases: sanitizedAliases?.isEmpty == true ? nil : sanitizedAliases,
                    ctcTokenIds: term.ctcTokenIds,
                    minSimilarity: term.minSimilarity
                ))
        }

        return CustomVocabularyContext(
            terms: validatedTerms,
            alpha: alpha,
            minCtcScore: minCtcScore,
            minSimilarity: minSimilarity,
            minCombinedConfidence: minCombinedConfidence,
            minTermLength: minTermLength,
            explicitMinSimilarity: config.minSimilarity.map { min(1.0, max(0.0, $0)) }
        )
    }

    /// Load the simple text format: one term per line, `#` comments allowed,
    /// optional `word: alias1, alias2` form.
    public static func loadFromSimpleFormat(from url: URL) throws -> CustomVocabularyContext {
        let contents = try String(contentsOf: url, encoding: .utf8)
        var terms: [CustomVocabularyTerm] = []

        for line in contents.split(whereSeparator: { $0.isNewline }) {
            let trimmed = String(line).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, !trimmed.hasPrefix("#") else { continue }

            if let colonIndex = trimmed.firstIndex(of: ":") {
                let word = String(trimmed[..<colonIndex]).trimmingCharacters(in: .whitespaces)
                let aliasesPart = String(trimmed[trimmed.index(after: colonIndex)...])
                let rawAliases = aliasesPart.split(separator: ",").map {
                    String($0).trimmingCharacters(in: .whitespaces)
                }.filter { !$0.isEmpty }

                let (sanitizedWord, _) = sanitizeVocabularyTerm(word)
                guard !sanitizedWord.isEmpty else { continue }

                let sanitizedAliases = rawAliases.compactMap { alias -> String? in
                    let (sanitized, _) = sanitizeVocabularyTerm(alias)
                    return sanitized.isEmpty ? nil : sanitized
                }

                terms.append(
                    CustomVocabularyTerm(
                        text: sanitizedWord,
                        weight: 10.0,
                        aliases: sanitizedAliases.isEmpty ? nil : sanitizedAliases
                    ))
            } else {
                let (sanitizedWord, _) = sanitizeVocabularyTerm(trimmed)
                guard !sanitizedWord.isEmpty else { continue }
                terms.append(CustomVocabularyTerm(text: sanitizedWord, weight: 10.0))
            }
        }

        return CustomVocabularyContext(terms: terms)
    }

    /// Sanitize a term, reporting anything a caller might want to know about.
    private static func sanitizeVocabularyTerm(_ text: String) -> (sanitized: String, warnings: [String]) {
        var warnings: [String] = []
        var result = text

        if result.rangeOfCharacter(from: .controlCharacters) != nil {
            warnings.append("contains control characters")
            result = result.filter { !$0.isNewline && !$0.isWhitespace || $0 == " " }
        }
        if result.folding(options: .diacriticInsensitive, locale: nil) != result {
            warnings.append("contains diacritics - consider adding an ASCII alias")
        }
        if result.rangeOfCharacter(from: .decimalDigits) != nil {
            warnings.append("contains numbers")
        }
        let allowedChars = CharacterSet.letters
            .union(.whitespaces)
            .union(CharacterSet(charactersIn: "-'"))
        if result.rangeOfCharacter(from: allowedChars.inverted) != nil {
            warnings.append("contains unusual characters")
        }

        return (result, warnings)
    }

    /// Load a vocabulary file and tokenize every term with the CTC tokenizer.
    ///
    /// The format is detected from the file contents: a first non-whitespace
    /// byte of `{` selects the structured JSON config (which carries per-term
    /// `minSimilarity` and the vocabulary-level thresholds); anything else is
    /// the simple one-term-per-line list.
    public static func loadWithCtcTokens(
        from path: String,
        tokenizer: CtcTokenizer
    ) throws -> CustomVocabularyContext {
        let loadedVocab = try loadVocabularyFile(at: URL(fileURLWithPath: path))

        let tokenizedTerms = loadedVocab.terms.compactMap { term -> CustomVocabularyTerm? in
            let tokenIds = tokenizer.encode(term.text)
            guard !tokenIds.isEmpty else { return nil }
            return CustomVocabularyTerm(
                text: term.text,
                weight: term.weight,
                aliases: term.aliases,
                ctcTokenIds: tokenIds,
                minSimilarity: term.minSimilarity
            )
        }

        return CustomVocabularyContext(
            terms: tokenizedTerms,
            alpha: loadedVocab.alpha,
            minCtcScore: loadedVocab.minCtcScore,
            minSimilarity: loadedVocab.minSimilarity,
            minCombinedConfidence: loadedVocab.minCombinedConfidence,
            minTermLength: loadedVocab.minTermLength,
            explicitMinSimilarity: loadedVocab.explicitMinSimilarity
        )
    }

    /// Auto-detect JSON config vs simple text list by the first meaningful byte.
    static func loadVocabularyFile(at url: URL) throws -> CustomVocabularyContext {
        let data = try Data(contentsOf: url)
        let whitespace: Set<UInt8> = [0x20, 0x09, 0x0a, 0x0d]
        let firstMeaningfulByte = data.first { !whitespace.contains($0) }
        if firstMeaningfulByte == UInt8(ascii: "{") {
            return try load(from: url)
        }
        return try loadFromSimpleFormat(from: url)
    }
}
