//
//  Adapted from FluidAudio (https://github.com/FluidInference/FluidAudio),
//  upstream commit 667181a, file
//  Sources/FluidAudio/ASR/Parakeet/SlidingWindow/CustomVocabulary/BKTree/VocabularyRescorer+CandidateMatching.swift
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

import Foundation

extension VocabularyRescorer {

    /// A vocabulary term proposed for a transcript word, with how many words it
    /// would consume.
    struct CandidateMatch {
        let term: CustomVocabularyTerm
        let similarity: Float
        /// Transcript words matched: 1 for a plain word, 2+ for a compound.
        let spanLength: Int
    }

    /// Candidate vocabulary terms for one transcript word: plain matches,
    /// two- and three-word compounds (for a term the decode split apart), and
    /// multi-word phrases.
    ///
    /// - Parameters:
    ///   - minSimilarity: vocabulary-level fallback threshold. Each candidate
    ///     is filtered against its own `term.minSimilarity` when it has one.
    ///   - searchFloor: the most permissive threshold anywhere in the
    ///     vocabulary. It only widens the BK-tree's distance bound, so that a
    ///     term with a low per-term override is not pruned away before its own
    ///     threshold gets a chance to apply.
    func findCandidateTermsForWord(
        normalizedWord: String,
        adjacentNormalized: [String],
        minSimilarity: Float,
        searchFloor: Float? = nil
    ) -> [CandidateMatch] {
        guard !normalizedWord.isEmpty else { return [] }

        let distanceFloor = min(searchFloor ?? minSimilarity, minSimilarity)

        func threshold(for term: CustomVocabularyTerm) -> Float {
            term.minSimilarity ?? minSimilarity
        }

        var candidates: [CandidateMatch] = []

        if useBKTree, let tree = bkTree {
            // 1. Single word.
            let maxLen1 = max(normalizedWord.count, 3)
            let maxDist1 = min(bkTreeMaxDistance, Int((1.0 - distanceFloor) * Float(maxLen1)))
            for result in tree.search(query: normalizedWord, maxDistance: maxDist1) {
                let similarity = Self.stringSimilarity(normalizedWord, result.normalizedText)
                if similarity >= threshold(for: result.term) {
                    candidates.append(CandidateMatch(term: result.term, similarity: similarity, spanLength: 1))
                }
            }

            // 2. Two-word compound.
            if let word2 = adjacentNormalized.first, !word2.isEmpty {
                let compound2 = normalizedWord + word2
                let maxLen2 = max(compound2.count, 3)
                let maxDist2 = min(bkTreeMaxDistance, Int((1.0 - distanceFloor) * Float(maxLen2)))
                for result in tree.search(query: compound2, maxDistance: maxDist2) {
                    let similarity = Self.lengthPenalizedSimilarity(compound2, result.normalizedText)
                    if similarity >= threshold(for: result.term) {
                        candidates.append(
                            CandidateMatch(term: result.term, similarity: similarity, spanLength: 2))
                    }
                }
            }

            // 3. Three-word compound, for longer terms only.
            if adjacentNormalized.count >= 2,
                let word2 = adjacentNormalized.first, !word2.isEmpty,
                let word3 = adjacentNormalized.dropFirst().first, !word3.isEmpty
            {
                let compound3 = normalizedWord + word2 + word3
                if compound3.count >= 6 {
                    let maxDist3 = min(bkTreeMaxDistance, Int((1.0 - distanceFloor) * Float(compound3.count)))
                    for result in tree.search(query: compound3, maxDistance: maxDist3) {
                        let similarity = Self.lengthPenalizedSimilarity(compound3, result.normalizedText)
                        if similarity >= threshold(for: result.term) {
                            candidates.append(
                                CandidateMatch(term: result.term, similarity: similarity, spanLength: 3))
                        }
                    }
                }
            }

            // 4. Multi-word phrases, for multi-word vocabulary terms.
            if !adjacentNormalized.isEmpty {
                for spanLen in 2...min(4, adjacentNormalized.count + 1) {
                    let phraseWords = [normalizedWord] + Array(adjacentNormalized.prefix(spanLen - 1))
                    let phrase = phraseWords.joined(separator: " ")
                    let maxLenPhrase = max(phrase.count, 3)
                    let maxDistPhrase = min(
                        bkTreeMaxDistance + 1, Int((1.0 - distanceFloor) * Float(maxLenPhrase)))
                    for result in tree.search(query: phrase, maxDistance: maxDistPhrase) {
                        let similarity = Self.stringSimilarity(phrase, result.normalizedText)
                        if similarity >= threshold(for: result.term) {
                            candidates.append(
                                CandidateMatch(term: result.term, similarity: similarity, spanLength: spanLen))
                        }
                    }
                }
            }

        } else {
            // Linear scan: O(V) per word, which wins outright on a vocabulary
            // of a handful of terms.
            for term in vocabulary.terms {
                let termNormalized = Self.normalizeForSimilarity(term.text)
                guard !termNormalized.isEmpty else { continue }

                let termWordCount = termNormalized.split(separator: " ").count
                let termThreshold = threshold(for: term)

                if termWordCount == 1 {
                    let similarity1 = Self.stringSimilarity(normalizedWord, termNormalized)
                    if similarity1 >= termThreshold {
                        candidates.append(CandidateMatch(term: term, similarity: similarity1, spanLength: 1))
                    }

                    if let word2 = adjacentNormalized.first, !word2.isEmpty {
                        let compound2 = normalizedWord + word2
                        let similarity2 = Self.lengthPenalizedSimilarity(compound2, termNormalized)
                        if similarity2 >= termThreshold {
                            candidates.append(CandidateMatch(term: term, similarity: similarity2, spanLength: 2))
                        }
                    }

                    if adjacentNormalized.count >= 2 {
                        let word2 = adjacentNormalized[0]
                        let word3 = adjacentNormalized[1]
                        if !word2.isEmpty && !word3.isEmpty {
                            let compound3 = normalizedWord + word2 + word3
                            if compound3.count >= 6 {
                                let similarity3 = Self.lengthPenalizedSimilarity(compound3, termNormalized)
                                if similarity3 >= termThreshold {
                                    candidates.append(
                                        CandidateMatch(term: term, similarity: similarity3, spanLength: 3))
                                }
                            }
                        }
                    }
                } else if !adjacentNormalized.isEmpty {
                    for spanLen in 2...min(4, adjacentNormalized.count + 1) {
                        let phraseWords = [normalizedWord] + Array(adjacentNormalized.prefix(spanLen - 1))
                        let phrase = phraseWords.joined(separator: " ")
                        let similarity = Self.stringSimilarity(phrase, termNormalized)
                        if similarity >= termThreshold {
                            candidates.append(
                                CandidateMatch(term: term, similarity: similarity, spanLength: spanLen))
                        }
                    }
                }
            }
        }

        return candidates.sorted {
            if $0.similarity != $1.similarity { return $0.similarity > $1.similarity }
            return $0.spanLength > $1.spanLength
        }
    }
}
