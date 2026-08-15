//
//  Adapted from FluidAudio (https://github.com/FluidInference/FluidAudio),
//  upstream commit 667181a, file
//  Sources/FluidAudio/ASR/Parakeet/SlidingWindow/CustomVocabulary/Rescorer/VocabularyRescorer+Utilities.swift
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
//  Changes from upstream: the UTF-8 span-alignment helpers, which exist only to
//  serve the diagnostic evidence API, are not ported.
//

import Foundation

extension VocabularyRescorer {

    // MARK: - String similarity

    /// Levenshtein similarity: `1 - editDistance / maxLength`, case-insensitive.
    static func stringSimilarity(_ a: String, _ b: String) -> Float {
        let aLower = a.lowercased()
        let bLower = b.lowercased()
        let distance = StringUtils.levenshteinDistance(aLower, bLower)
        let maxLen = max(aLower.count, bLower.count)
        guard maxLen > 0 else { return 1.0 }
        return 1.0 - Float(distance) / Float(maxLen)
    }

    /// Similarity with a length penalty, for compound matches.
    ///
    /// When several transcript words are concatenated to reach one vocabulary
    /// term, plain edit similarity is too generous about a big length mismatch;
    /// the square root softens the penalty without ignoring it.
    static func lengthPenalizedSimilarity(_ compound: String, _ vocabTerm: String) -> Float {
        let baseSimilarity = stringSimilarity(compound, vocabTerm)
        let compoundLen = Float(compound.count)
        let vocabLen = Float(vocabTerm.count)
        let lengthRatio = min(compoundLen, vocabLen) / max(compoundLen, vocabLen)
        return baseSimilarity * sqrt(lengthRatio)
    }

    // MARK: - Normalized forms

    /// A normalized spelling of a term: the canonical form, or one alias.
    struct NormalizedForm: Hashable {
        let normalized: String
        let wordCount: Int
        let matchedAlias: String?
    }

    /// Normalize canonical + aliases, deduplicating but remembering which alias
    /// each surviving form came from.
    static func normalizedForms(canonicalTerm: String, aliases: [String]) -> [NormalizedForm] {
        let rawForms: [(text: String, matchedAlias: String?)] =
            [(canonicalTerm, nil)] + aliases.map { ($0, $0) }
        var seen = Set<String>()
        var forms: [NormalizedForm] = []

        for rawForm in rawForms {
            let normalized = normalizeForSimilarity(rawForm.text)
            guard !normalized.isEmpty, seen.insert(normalized).inserted else { continue }
            forms.append(
                NormalizedForm(
                    normalized: normalized,
                    wordCount: normalized.split(separator: " ").count,
                    matchedAlias: rawForm.matchedAlias
                ))
        }
        return forms
    }

    /// All normalized forms for a term, pulling in aliases declared anywhere in
    /// the vocabulary for the same canonical text.
    func buildNormalizedForms(for term: CustomVocabularyTerm) -> [NormalizedForm] {
        var aliases: [String] = []
        let termLower = term.textLowercased

        for vocabTerm in vocabulary.terms where vocabTerm.textLowercased == termLower {
            if let vocabularyAliases = vocabTerm.aliases {
                aliases.append(contentsOf: vocabularyAliases)
            }
        }
        if let termAliases = term.aliases {
            aliases.append(contentsOf: termAliases)
        }

        return Self.normalizedForms(canonicalTerm: term.text, aliases: aliases)
    }

    // MARK: - Similarity thresholds

    /// Required similarity for a span of the given length.
    func requiredSimilarity(minSimilarity: Float, spanLength: Int) -> Float {
        // Multi-word spans replace more text on less evidence, so they carry a
        // floor of their own. Single words use the configured minimum as-is;
        // an earlier, stricter short-word rule here cost more recall than it
        // bought precision.
        if spanLength >= 2 {
            return max(minSimilarity, 0.55)
        }
        return minSimilarity
    }

    // MARK: - Text utilities

    /// Carry the original word's leading capital onto the replacement.
    func preserveCapitalization(original: String, replacement: String) -> String {
        guard let firstChar = original.first else { return replacement }
        if firstChar.isUppercase && replacement.first?.isLowercase == true {
            return replacement.prefix(1).uppercased() + replacement.dropFirst()
        }
        return replacement
    }

    /// Normalize for similarity: lowercase, collapse whitespace, drop
    /// punctuation but keep letters, digits, apostrophes and hyphens.
    static func normalizeForSimilarity(_ text: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "'-"))
        var result = ""
        var lastWasSpace = true

        for scalar in text.lowercased().unicodeScalars {
            if allowed.contains(scalar) {
                result.append(Character(scalar))
                lastWasSpace = false
            } else if scalar == " " || scalar == "\t" || scalar == "\n" {
                if !lastWasSpace && !result.isEmpty {
                    result.append(" ")
                    lastWasSpace = true
                }
            }
        }

        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Every normalized spelling the vocabulary already owns, canonical and
    /// alias alike. Used to leave a word alone when it is already some *other*
    /// vocabulary term spelled correctly.
    func buildVocabularyNormalizedSet() -> Set<String> {
        var normalizedSet = Set<String>()
        for term in vocabulary.terms {
            let normalized = Self.normalizeForSimilarity(term.text)
            if !normalized.isEmpty { normalizedSet.insert(normalized) }
            if let aliases = term.aliases {
                for alias in aliases {
                    let normalizedAlias = Self.normalizeForSimilarity(alias)
                    if !normalizedAlias.isEmpty { normalizedSet.insert(normalizedAlias) }
                }
            }
        }
        return normalizedSet
    }
}
