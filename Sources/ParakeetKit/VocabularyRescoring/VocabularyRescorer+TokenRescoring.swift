//
//  Adapted from FluidAudio (https://github.com/FluidInference/FluidAudio),
//  upstream commit 667181a, file
//  Sources/FluidAudio/ASR/Parakeet/SlidingWindow/CustomVocabulary/Rescorer/VocabularyRescorer+TokenRescoring.swift
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
//  Changes from upstream: candidate-evidence collection is not ported (it is
//  inert unless the diagnostic API asks for it), and the entry point takes
//  pre-built word timings instead of token timings.
//

import Foundation

extension VocabularyRescorer {

    @inline(__always)
    private func debugLog(_ message: @autoclosure () -> String) {
        guard debugMode else { return }
        logger.debug(message())
    }

    // MARK: - Stopwords

    /// Stopwords the **single-word** paths refuse to replace outright. Wider
    /// than the multi-word set, because substituting over a lone common word is
    /// the most error-prone thing this pass can do.
    static let stopwords: Set<String> = [
        // Articles and determiners
        "a", "an", "the", "some", "any", "no", "every", "each", "all",
        // Conjunctions
        "and", "or", "but", "so", "if", "then", "than", "as",
        // Prepositions
        "in", "on", "at", "to", "for", "of", "with", "by", "from", "up", "down",
        "out", "about", "into", "over", "after", "before", "between", "under",
        // Be verbs
        "is", "are", "was", "were", "be", "been", "being", "am",
        // Common verbs
        "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
        "go", "goes", "went", "come", "comes", "came", "get", "got", "take", "took",
        "make", "made", "say", "said", "see", "saw", "know", "knew", "think", "thought",
        // Pronouns
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "its", "our", "their", "this", "that", "these", "those",
        "who", "what", "which", "where", "when", "how", "why",
        // Common short words
        "just", "also", "only", "even", "still", "now", "here", "there", "very",
        "well", "back", "way", "own", "new", "old", "good", "great", "first", "last",
    ]

    /// The subset used to raise (not veto) the threshold on a **multi-word**
    /// span. Restricted to true function words, so a span of content words is
    /// not silently held to the stopword bar.
    static let multiWordStopwords: Set<String> = [
        // Articles and determiners
        "a", "an", "the", "some", "any", "no", "every", "each", "all",
        // Conjunctions
        "and", "or", "but", "so", "if", "then", "than", "as",
        // Prepositions
        "in", "on", "at", "to", "for", "of", "with", "by", "from", "up", "down",
        "out", "about", "into", "over", "after", "before", "between", "under",
        // Be verbs
        "is", "are", "was", "were", "be", "been", "being", "am",
        // Auxiliaries
        "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
        // Pronouns
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "its", "our", "their", "this", "that", "these", "those",
        "who", "what", "which", "where", "when", "how", "why",
    ]

    // MARK: - Match types

    /// One (transcript span → vocabulary term) pair awaiting the CTC decision.
    struct CTCMatchCandidate {
        let origin: CandidateOrigin
        let originalPhrase: String
        let vocabTerm: String
        let matchedAlias: String?
        let vocabTokens: [Int]
        let similarity: Float
        let spanLength: Int
        let spanIndices: [Int]
        let spanStartTime: Double
        let spanEndTime: Double
    }

    /// The verdict on one candidate.
    struct CTCMatchResult {
        let shouldReplace: Bool
        let comparisonWasPerformed: Bool
        let originalScore: Float
        let boostedVocabScore: Float
        let replacement: String
        let reason: String
    }

    /// A candidate that passed the CTC comparison and is queued for arbitration.
    struct PendingReplacement {
        let candidate: CTCMatchCandidate
        let result: CTCMatchResult
        let similarity: Float
    }

    // MARK: - Finalization

    /// Rank the accepted candidates, apply the non-overlapping ones, and
    /// rebuild the transcript.
    private func finalizeReplacements(
        pendingReplacements: [PendingReplacement],
        modifiedWords: inout [String],
        replacedIndices: inout Set<Int>,
        replacements: inout [RescoringResult]
    ) -> RescoreOutput {
        let applied = Self.arbitratePendingReplacements(
            pendingReplacements, occupiedIndices: replacedIndices)

        for pending in applied {
            applyReplacement(
                result: pending.result,
                candidate: pending.candidate,
                modifiedWords: &modifiedWords,
                replacedIndices: &replacedIndices,
                replacements: &replacements
            )
        }

        let modifiedText = modifiedWords.filter { !$0.isEmpty }.joined(separator: " ")
        let wasModified = !replacements.isEmpty
        debugLog("final: \(replacements.count) replacement(s)")

        return RescoreOutput(text: modifiedText, replacements: replacements, wasModified: wasModified)
    }

    /// Order the accepted candidates and take the ones that do not overlap.
    ///
    /// Similarity leads, span length only tiebreaks. When a longer multi-word
    /// span matches substantially better than a shorter overlapping one, the
    /// longer match should win — a "shortest span wins" rule picks the short
    /// distractor purely for being short, even when the long match is far
    /// closer.
    ///
    /// Similarity is quantized into 0.05 buckets first. Comparing
    /// `abs(difference) > 0.05` directly is *not* transitive — for 0.70 / 0.66 /
    /// 0.62 across span lengths 3 / 2 / 1, A-vs-B and B-vs-C dispatch to the
    /// span tiebreaker while A-vs-C dispatches to similarity, and the sort has
    /// a cycle. Bucketing gives a strict weak ordering by construction.
    static func arbitratePendingReplacements(
        _ pendingReplacements: [PendingReplacement],
        occupiedIndices: Set<Int> = []
    ) -> [PendingReplacement] {
        let quantized: (Float) -> Int = { Int(($0 / 0.05).rounded()) }
        let sortedReplacements = pendingReplacements.sorted { a, b in
            let aBucket = quantized(a.similarity)
            let bBucket = quantized(b.similarity)
            if aBucket != bBucket { return aBucket > bBucket }
            if a.candidate.spanLength != b.candidate.spanLength {
                return a.candidate.spanLength < b.candidate.spanLength
            }
            return a.similarity > b.similarity
        }

        var occupiedIndices = occupiedIndices
        var applied: [PendingReplacement] = []
        for pending in sortedReplacements {
            guard pending.candidate.spanIndices.allSatisfy({ !occupiedIndices.contains($0) }) else { continue }
            applied.append(pending)
            occupiedIndices.formUnion(pending.candidate.spanIndices)
        }
        return applied
    }

    // MARK: - Public API

    /// Rescore a transcript against the vocabulary using constrained CTC.
    ///
    /// - Parameters:
    ///   - wordTimings: the transcript, word by word, with the times the decode
    ///     assigned each word. These select the CTC frames each comparison is
    ///     allowed to look at, so their alignment to the log-probabilities is
    ///     what makes the whole pass meaningful.
    ///   - logProbs: CTC log-probabilities `[T, V]` over the same audio.
    ///   - frameDuration: seconds per CTC frame.
    ///   - cbw: context-biasing weight added to the vocabulary term's score.
    ///   - marginSeconds: slack around each word's window, for timing jitter.
    ///   - minSimilarity: vocabulary-level similarity gate; per-term overrides
    ///     take precedence over it.
    public func ctcTokenRescore(
        wordTimings: [WordTiming],
        logProbs: [[Float]],
        frameDuration: Double,
        cbw: Float = ContextBiasingConstants.defaultCbw,
        marginSeconds: Double = ContextBiasingConstants.defaultMarginSeconds,
        minSimilarity: Float = ContextBiasingConstants.minSimilarityFloor
    ) -> RescoreOutput {
        if useBKTree {
            return rescoreWordCentric(
                wordTimings: wordTimings, logProbs: logProbs, frameDuration: frameDuration,
                cbw: cbw, marginSeconds: marginSeconds, minSimilarity: minSimilarity)
        }
        return rescoreTermCentric(
            wordTimings: wordTimings, logProbs: logProbs, frameDuration: frameDuration,
            cbw: cbw, marginSeconds: marginSeconds, minSimilarity: minSimilarity)
    }

    // MARK: - Word-centric algorithm (experimental)

    /// For each transcript word, ask the BK-tree which vocabulary terms are
    /// close, then run the CTC comparison for each. O(W log V) with the tree.
    private func rescoreWordCentric(
        wordTimings: [WordTiming],
        logProbs: [[Float]],
        frameDuration: Double,
        cbw: Float,
        marginSeconds: Double,
        minSimilarity: Float
    ) -> RescoreOutput {
        let transcript = wordTimings.map(\.word).joined(separator: " ")
        guard !wordTimings.isEmpty, !logProbs.isEmpty else {
            return RescoreOutput(text: transcript, replacements: [], wasModified: false)
        }

        debugLog(
            "word-centric: \(wordTimings.count) words, \(logProbs.count) frames, "
                + "cbw=\(cbw), margin=\(marginSeconds)s, minSim=\(minSimilarity)")

        var replacements: [RescoringResult] = []
        var modifiedWords = wordTimings.map(\.word)
        var replacedIndices = Set<Int>()
        var pendingReplacements: [PendingReplacement] = []

        let vocabularyNormalizedSet = buildVocabularyNormalizedSet()

        // The BK-tree distance bound is derived from the most permissive
        // per-term threshold, so a term with a low override is not pruned
        // before its own threshold has a chance to apply.
        let searchFloor = vocabulary.terms.reduce(minSimilarity) { min($0, $1.minSimilarity ?? minSimilarity) }
        let normalizedWords = wordTimings.map { Self.normalizeForSimilarity($0.word) }

        for (wordIdx, timing) in wordTimings.enumerated() {
            guard !replacedIndices.contains(wordIdx) else { continue }

            let tdtWord = timing.word
            let normalizedWord = normalizedWords[wordIdx]
            guard !normalizedWord.isEmpty else { continue }

            var adjacentNormalized: [String] = []
            for offset in 1...3 {
                let idx = wordIdx + offset
                guard idx < wordTimings.count, !replacedIndices.contains(idx) else { break }
                let norm = normalizedWords[idx]
                guard !norm.isEmpty else { break }
                adjacentNormalized.append(norm)
            }

            let candidates = findCandidateTermsForWord(
                normalizedWord: normalizedWord,
                adjacentNormalized: adjacentNormalized,
                minSimilarity: minSimilarity,
                searchFloor: searchFloor
            )

            for candidate in candidates {
                let term = candidate.term
                let vocabTerm = term.text
                let similarity = candidate.similarity
                let spanLength = candidate.spanLength

                guard vocabTerm.count >= vocabulary.minTermLength else { continue }
                guard let vocabTokens = term.ctcTokenIds, !vocabTokens.isEmpty else { continue }
                guard wordIdx + spanLength <= wordTimings.count else { continue }

                let spanIndices = Array(wordIdx..<(wordIdx + spanLength))
                guard spanIndices.allSatisfy({ !replacedIndices.contains($0) }) else { continue }

                let originalPhrase =
                    spanLength == 1 ? tdtWord : spanIndices.map { wordTimings[$0].word }.joined(separator: " ")
                let normalizedPhrase =
                    spanLength == 1 ? normalizedWord : spanIndices.map { normalizedWords[$0] }.joined(separator: " ")

                // Already the canonical spelling: nothing to do.
                let normalizedCanonical = Self.normalizeForSimilarity(vocabTerm)
                if normalizedPhrase == normalizedCanonical { continue }

                // Already some *other* vocabulary term, spelled correctly.
                let normalizedCurrentSet = Set(buildNormalizedForms(for: term).map { $0.normalized })
                if vocabularyNormalizedSet.contains(normalizedPhrase)
                    && !normalizedCurrentSet.contains(normalizedPhrase)
                {
                    debugLog("  skipping '\(vocabTerm)': '\(originalPhrase)' is another vocab term")
                    continue
                }

                // Per-term override falls back to the vocabulary threshold; the
                // guards below can still clamp it upward.
                let termMinSimilarity = term.minSimilarity ?? minSimilarity
                var minSimilarityForSpan = requiredSimilarity(
                    minSimilarity: termMinSimilarity, spanLength: spanLength)

                if spanLength == 1 {
                    minSimilarityForSpan = checkLengthRatioRules(
                        normalizedWord: normalizedWord,
                        vocabTerm: vocabTerm,
                        currentSimilarity: similarity,
                        minSimilarity: minSimilarityForSpan
                    )
                }

                let spanWords = spanLength >= 2 ? spanIndices.map { normalizedWords[$0] } : []
                let (shouldSkipStopword, adjustedSimilarity) = checkStopwordRules(
                    normalizedWord: normalizedWord,
                    spanLength: spanLength,
                    spanWords: spanWords,
                    vocabTerm: vocabTerm,
                    currentSimilarity: minSimilarityForSpan
                )
                if shouldSkipStopword { continue }
                minSimilarityForSpan = adjustedSimilarity

                guard similarity >= minSimilarityForSpan else { continue }

                let matchCandidate = CTCMatchCandidate(
                    origin: .wordCentric,
                    originalPhrase: originalPhrase,
                    vocabTerm: vocabTerm,
                    matchedAlias: nil,
                    vocabTokens: vocabTokens,
                    similarity: similarity,
                    spanLength: spanLength,
                    spanIndices: spanIndices,
                    spanStartTime: wordTimings[wordIdx].startTime,
                    spanEndTime: wordTimings[wordIdx + spanLength - 1].endTime
                )

                let result = evaluateCTCMatch(
                    candidate: matchCandidate, logProbs: logProbs,
                    frameDuration: frameDuration, cbw: cbw, marginSeconds: marginSeconds)

                if result.shouldReplace {
                    pendingReplacements.append(
                        PendingReplacement(candidate: matchCandidate, result: result, similarity: similarity))
                }
            }
        }

        return finalizeReplacements(
            pendingReplacements: pendingReplacements,
            modifiedWords: &modifiedWords,
            replacedIndices: &replacedIndices,
            replacements: &replacements
        )
    }

    // MARK: - Term-centric algorithm (default)

    /// For each vocabulary term, sweep the transcript for words it might be,
    /// then run the CTC comparison for each hit. This is the benchmarked
    /// default: it processes the vocabulary in file order and scores better
    /// than the word-centric path.
    private func rescoreTermCentric(
        wordTimings: [WordTiming],
        logProbs: [[Float]],
        frameDuration: Double,
        cbw: Float,
        marginSeconds: Double,
        minSimilarity: Float
    ) -> RescoreOutput {
        let transcript = wordTimings.map(\.word).joined(separator: " ")
        guard !wordTimings.isEmpty, !logProbs.isEmpty else {
            return RescoreOutput(text: transcript, replacements: [], wasModified: false)
        }

        debugLog(
            "term-centric: \(wordTimings.count) words, \(logProbs.count) frames, "
                + "cbw=\(cbw), margin=\(marginSeconds)s, minSim=\(minSimilarity)")

        var replacements: [RescoringResult] = []
        var modifiedWords = wordTimings.map(\.word)
        var replacedIndices = Set<Int>()
        var pendingReplacements: [PendingReplacement] = []

        let vocabularyNormalizedSet = buildVocabularyNormalizedSet()
        // Upstream recomputes this inside the per-term loop; the normalization
        // does not depend on the term, so it is hoisted. Same strings.
        let normalizedWords = wordTimings.map { Self.normalizeForSimilarity($0.word) }

        for term in vocabulary.terms {
            let vocabTerm = term.text

            // The per-term similarity override. The safety guards
            // (requiredSimilarity / stopword / length-ratio) still clamp it
            // upward, so an override can only loosen matching down to them.
            let termMinSimilarity = term.minSimilarity ?? minSimilarity

            guard vocabTerm.count >= vocabulary.minTermLength else {
                debugLog("  skipping '\(vocabTerm)': shorter than \(vocabulary.minTermLength) chars")
                continue
            }
            guard let vocabTokens = term.ctcTokenIds, !vocabTokens.isEmpty else { continue }

            let normalizedForms = buildNormalizedForms(for: term)
            guard !normalizedForms.isEmpty else { continue }

            let normalizedCanonical = Self.normalizeForSimilarity(vocabTerm)
            let normalizedCurrentSet = Set(normalizedForms.map { $0.normalized })

            let multiWordForms = normalizedForms.filter { $0.wordCount > 1 }
            let singleWordForms = normalizedForms.filter { $0.wordCount == 1 }

            if !multiWordForms.isEmpty {
                // Multi-word phrase matching over consecutive transcript words.
                let maxWordCount = multiWordForms.map { $0.wordCount }.max() ?? 0
                let minWordCount = multiWordForms.map { $0.wordCount }.min() ?? 0
                let maxSpan = min(4, maxWordCount + 1)  // a little slack
                let minSpan = max(2, minWordCount)

                guard minSpan <= maxSpan else { continue }

                for spanLength in minSpan...maxSpan {
                    guard spanLength <= wordTimings.count else { break }
                    for startIdx in 0..<(wordTimings.count - spanLength + 1) {
                        let spanIndices = Array(startIdx..<(startIdx + spanLength))
                        guard spanIndices.allSatisfy({ !replacedIndices.contains($0) }) else { continue }

                        let tdtPhrase = spanIndices.map { wordTimings[$0].word }.joined(separator: " ")
                        // Normalize the joined phrase rather than joining the
                        // per-word normalizations: a word that normalizes away
                        // entirely must not leave a doubled space behind.
                        let normalizedPhrase = Self.normalizeForSimilarity(tdtPhrase)
                        guard !normalizedPhrase.isEmpty else { continue }

                        var bestSimilarity: Float = 0
                        var matchedAlias: String?
                        for form in multiWordForms {
                            let similarity = Self.stringSimilarity(normalizedPhrase, form.normalized)
                            if similarity > bestSimilarity {
                                bestSimilarity = similarity
                                matchedAlias = form.matchedAlias
                            }
                        }

                        if normalizedPhrase == normalizedCanonical { continue }
                        if vocabularyNormalizedSet.contains(normalizedPhrase)
                            && !normalizedCurrentSet.contains(normalizedPhrase)
                        {
                            debugLog("  [multi] skipping '\(vocabTerm)': '\(tdtPhrase)' is another vocab term")
                            continue
                        }

                        let minSimilarityForSpan = requiredSimilarity(
                            minSimilarity: termMinSimilarity, spanLength: spanLength)
                        if bestSimilarity < minSimilarityForSpan { continue }

                        let matchCandidate = CTCMatchCandidate(
                            origin: .termCentricMultiWord,
                            originalPhrase: tdtPhrase,
                            vocabTerm: vocabTerm,
                            matchedAlias: matchedAlias,
                            vocabTokens: vocabTokens,
                            similarity: bestSimilarity,
                            spanLength: spanLength,
                            spanIndices: spanIndices,
                            spanStartTime: wordTimings[spanIndices.first!].startTime,
                            spanEndTime: wordTimings[spanIndices.last!].endTime
                        )

                        let result = evaluateCTCMatch(
                            candidate: matchCandidate, logProbs: logProbs,
                            frameDuration: frameDuration, cbw: cbw, marginSeconds: marginSeconds)

                        if result.shouldReplace {
                            pendingReplacements.append(
                                PendingReplacement(
                                    candidate: matchCandidate, result: result, similarity: bestSimilarity))
                        }
                    }
                }
            }

            if !singleWordForms.isEmpty {
                for (wordIdx, timing) in wordTimings.enumerated() {
                    guard !replacedIndices.contains(wordIdx) else { continue }

                    let tdtWord = timing.word
                    let normalizedWord = normalizedWords[wordIdx]
                    guard !normalizedWord.isEmpty else { continue }

                    if normalizedWord == normalizedCanonical { continue }
                    if vocabularyNormalizedSet.contains(normalizedWord)
                        && !normalizedCurrentSet.contains(normalizedWord)
                    {
                        debugLog("  skipping '\(vocabTerm)': '\(tdtWord)' is another vocab term")
                        continue
                    }

                    var bestSimilarity: Float = 0
                    var matchedSpanLength = 1
                    var matchedAlias: String?
                    for form in singleWordForms {
                        let similarity = Self.stringSimilarity(normalizedWord, form.normalized)
                        if similarity > bestSimilarity {
                            bestSimilarity = similarity
                            matchedAlias = form.matchedAlias
                        }
                    }

                    // Compound matching: a single-word term the decode split
                    // across adjacent words. Length floors keep short terms out
                    // of it, where concatenation would find matches everywhere.
                    let minLengthFor2Word = 4
                    let minLengthFor3Word = 8

                    let normalized2: String? =
                        (wordIdx + 1 < wordTimings.count && !replacedIndices.contains(wordIdx + 1))
                        ? normalizedWords[wordIdx + 1] : nil
                    let normalized3: String? =
                        (wordIdx + 2 < wordTimings.count && !replacedIndices.contains(wordIdx + 2))
                        ? normalizedWords[wordIdx + 2] : nil

                    // Skip the compound when the second word already matches the
                    // term well on its own — it is the match, not half of one.
                    if let norm2 = normalized2, !norm2.isEmpty, vocabTerm.count >= minLengthFor2Word {
                        let norm2MatchesVocab = singleWordForms.contains {
                            Self.stringSimilarity(norm2, $0.normalized) >= 0.9
                        }
                        if !norm2MatchesVocab {
                            let concatenated = normalizedWord + norm2
                            for form in singleWordForms {
                                let concatSimilarity = Self.stringSimilarity(concatenated, form.normalized)
                                if concatSimilarity > bestSimilarity {
                                    bestSimilarity = concatSimilarity
                                    matchedSpanLength = 2
                                    matchedAlias = form.matchedAlias
                                }
                            }
                        }
                    }

                    if let norm2 = normalized2, let norm3 = normalized3,
                        !norm2.isEmpty, !norm3.isEmpty, vocabTerm.count >= minLengthFor3Word
                    {
                        let laterWordMatchesVocab = singleWordForms.contains {
                            Self.stringSimilarity(norm2, $0.normalized) >= 0.9
                                || Self.stringSimilarity(norm3, $0.normalized) >= 0.9
                        }
                        if !laterWordMatchesVocab {
                            let concatenated = normalizedWord + norm2 + norm3
                            for form in singleWordForms {
                                let concatSimilarity = Self.stringSimilarity(concatenated, form.normalized)
                                if concatSimilarity > bestSimilarity {
                                    bestSimilarity = concatSimilarity
                                    matchedSpanLength = 3
                                    matchedAlias = form.matchedAlias
                                }
                            }
                        }
                    }

                    var minSimilarityForSpan = requiredSimilarity(
                        minSimilarity: termMinSimilarity, spanLength: matchedSpanLength)

                    if matchedSpanLength == 1 {
                        minSimilarityForSpan = checkLengthRatioRules(
                            normalizedWord: normalizedWord,
                            vocabTerm: vocabTerm,
                            currentSimilarity: bestSimilarity,
                            minSimilarity: minSimilarityForSpan
                        )
                    }

                    let spanWords =
                        matchedSpanLength >= 2
                        ? (0..<matchedSpanLength).map { normalizedWords[wordIdx + $0] } : []
                    let (shouldSkipStopword, adjustedSimilarity) = checkStopwordRules(
                        normalizedWord: normalizedWord,
                        spanLength: matchedSpanLength,
                        spanWords: spanWords,
                        vocabTerm: vocabTerm,
                        currentSimilarity: minSimilarityForSpan
                    )
                    if shouldSkipStopword { continue }
                    minSimilarityForSpan = adjustedSimilarity

                    if bestSimilarity < minSimilarityForSpan { continue }

                    let spanIndices = Array(wordIdx..<(wordIdx + matchedSpanLength))
                    let originalPhrase =
                        matchedSpanLength == 1
                        ? tdtWord : spanIndices.map { wordTimings[$0].word }.joined(separator: " ")

                    let matchCandidate = CTCMatchCandidate(
                        origin: .termCentricSingleWord,
                        originalPhrase: originalPhrase,
                        vocabTerm: vocabTerm,
                        matchedAlias: matchedAlias,
                        vocabTokens: vocabTokens,
                        similarity: bestSimilarity,
                        spanLength: matchedSpanLength,
                        spanIndices: spanIndices,
                        spanStartTime: wordTimings[wordIdx].startTime,
                        spanEndTime: wordTimings[wordIdx + matchedSpanLength - 1].endTime
                    )

                    let result = evaluateCTCMatch(
                        candidate: matchCandidate, logProbs: logProbs,
                        frameDuration: frameDuration, cbw: cbw, marginSeconds: marginSeconds)

                    if result.shouldReplace {
                        pendingReplacements.append(
                            PendingReplacement(
                                candidate: matchCandidate, result: result, similarity: bestSimilarity))
                    }
                }
            }
        }

        // The spotter-anchored rescue.
        //
        // Everything above starts from string similarity, so it only fires when
        // the decode's guess is already near some vocabulary term. A name the
        // decode mangles past that gate is invisible to it — but the CTC
        // keyword spotter can still hear the name in the audio. This pass
        // collects those detections and proposes them.
        //
        // Size-gated: on a large vocabulary the spotter has too many
        // phonetically similar terms competing, and the rescue turns into a
        // false-positive generator. It is also opt-out, because on a *short*
        // vocabulary it over-fires more than it recovers.
        if config.spotterRescueEnabled,
            vocabulary.terms.count <= ContextBiasingConstants.largeVocabThreshold
        {
            collectSpotterAnchoredCandidates(
                wordTimings: wordTimings,
                normalizedWords: normalizedWords,
                logProbs: logProbs,
                frameDuration: frameDuration,
                cbw: cbw,
                marginSeconds: marginSeconds,
                vocabularyNormalizedSet: vocabularyNormalizedSet,
                pendingReplacements: &pendingReplacements
            )
        }

        return finalizeReplacements(
            pendingReplacements: pendingReplacements,
            modifiedWords: &modifiedWords,
            replacedIndices: &replacedIndices,
            replacements: &replacements
        )
    }

    // MARK: - Spotter-anchored rescue

    /// Minimum spotter score for a detection to be worth proposing. Looser than
    /// the production floor, because the CTC-vs-CTC comparison downstream is
    /// what actually decides.
    private static let spotterRescueMinScore: Float = -10.0

    /// Propose replacements grounded in acoustic detections rather than string
    /// similarity, mapping each detection's time window onto transcript words.
    private func collectSpotterAnchoredCandidates(
        wordTimings: [WordTiming],
        normalizedWords: [String],
        logProbs: [[Float]],
        frameDuration: Double,
        cbw: Float,
        marginSeconds: Double,
        vocabularyNormalizedSet: Set<String>,
        pendingReplacements: inout [PendingReplacement]
    ) {
        let result = spotter.spotKeywordsFromLogProbs(
            logProbs: logProbs,
            frameDuration: frameDuration,
            customVocabulary: vocabulary,
            minScore: Self.spotterRescueMinScore
        )
        guard !result.detections.isEmpty else { return }

        // Highest-confidence detection races for its span first.
        let detections = result.detections.sorted { $0.score > $1.score }
        var seenSpans = Set<String>()

        for detection in detections {
            let term = detection.term
            let vocabTerm = term.text
            guard vocabTerm.count >= vocabulary.minTermLength else { continue }

            let span = wordIndices(
                in: wordTimings, overlapping: detection.startTime, end: detection.endTime)
            guard !span.isEmpty else { continue }

            // Bound the width so a misaligned detection cannot swallow a
            // sentence. 4 mirrors the multi-word ceiling above.
            guard span.count <= 4 else { continue }

            let dedupeKey = "\(term.textLowercased):\(span.first!):\(span.count)"
            guard seenSpans.insert(dedupeKey).inserted else { continue }

            let originalPhrase = span.map { wordTimings[$0].word }.joined(separator: " ")
            let normalizedSpanWords = span.map { normalizedWords[$0] }
            let normalizedPhrase = Self.normalizeForSimilarity(originalPhrase)

            let normalizedForms = buildNormalizedForms(for: term)
            let normalizedCurrentSet = Set(normalizedForms.map { $0.normalized })
            if vocabularyNormalizedSet.contains(normalizedPhrase),
                !normalizedCurrentSet.contains(normalizedPhrase)
            {
                continue
            }

            let normalizedCanonical = Self.normalizeForSimilarity(vocabTerm)
            if normalizedPhrase == normalizedCanonical { continue }
            if normalizedCurrentSet.contains(normalizedPhrase) { continue }

            guard let vocabTokens = term.ctcTokenIds, !vocabTokens.isEmpty else { continue }

            // Reject a rescue whose whole span is common words. The spotter's
            // window is several frames wider than the lone word it lands on, so
            // its score can beat that word's inside the same window without the
            // word being wrong at all.
            if span.count == 1 && Self.stopwords.contains(normalizedSpanWords[0]) { continue }
            if span.count >= 2 && normalizedSpanWords.allSatisfy({ Self.stopwords.contains($0) }) { continue }

            var bestSimilarity: Float = 0
            var matchedAlias: String?
            for form in normalizedForms {
                let similarity = Self.stringSimilarity(normalizedPhrase, form.normalized)
                if similarity > bestSimilarity {
                    bestSimilarity = similarity
                    matchedAlias = form.matchedAlias
                }
            }

            // Similarity floor for the acoustic path. This pass otherwise
            // ignores similarity entirely, which is what lets it force-replace
            // a low-similarity span. The floor is the stricter of the term's
            // explicit override and the span-aware config floor; both default
            // to disabled, so by default similarity stays ranking-only.
            let configSpotterFloor =
                span.count >= 2
                ? config.spotterRescueMultiWordMinSimilarity : config.spotterRescueMinSimilarity
            let spotterSimFloor = max(term.minSimilarity ?? 0, configSpotterFloor)
            if bestSimilarity < spotterSimFloor {
                debugLog(
                    "  [rescue] skipping '\(vocabTerm)' over '\(originalPhrase)': "
                        + "sim \(String(format: "%.2f", bestSimilarity)) < floor "
                        + "\(String(format: "%.2f", spotterSimFloor)) (span=\(span.count))")
                continue
            }

            let candidate = CTCMatchCandidate(
                origin: .spotterRescue,
                originalPhrase: originalPhrase,
                vocabTerm: vocabTerm,
                matchedAlias: matchedAlias,
                vocabTokens: vocabTokens,
                similarity: bestSimilarity,
                spanLength: span.count,
                spanIndices: span,
                spanStartTime: wordTimings[span.first!].startTime,
                spanEndTime: wordTimings[span.last!].endTime
            )

            let evalResult = evaluateCTCMatch(
                candidate: candidate, logProbs: logProbs,
                frameDuration: frameDuration, cbw: cbw, marginSeconds: marginSeconds)
            guard evalResult.shouldReplace else { continue }

            debugLog(
                "  [rescue] '\(originalPhrase)' -> '\(vocabTerm)' "
                    + "spotter=\(String(format: "%.2f", detection.score)), "
                    + "sim=\(String(format: "%.2f", bestSimilarity))")

            pendingReplacements.append(
                PendingReplacement(candidate: candidate, result: evalResult, similarity: bestSimilarity))
        }
    }

    /// Transcript word indices whose windows overlap a detection, reduced to a
    /// contiguous run. With no overlap at all, the nearest word by midpoint.
    private func wordIndices(
        in wordTimings: [WordTiming],
        overlapping start: Double,
        end: Double
    ) -> [Int] {
        guard !wordTimings.isEmpty, start < end else { return [] }

        var overlapping: [Int] = []
        for (idx, w) in wordTimings.enumerated() {
            if w.endTime < start { continue }
            if w.startTime > end { break }
            overlapping.append(idx)
        }

        if overlapping.isEmpty {
            let center = (start + end) / 2.0
            var bestIdx = 0
            var bestDelta = Double.infinity
            for (idx, w) in wordTimings.enumerated() {
                let delta = abs((w.startTime + w.endTime) / 2.0 - center)
                if delta < bestDelta {
                    bestDelta = delta
                    bestIdx = idx
                }
            }
            return [bestIdx]
        }

        var contiguous: [Int] = [overlapping[0]]
        for i in 1..<overlapping.count {
            guard overlapping[i] == contiguous.last! + 1 else { break }
            contiguous.append(overlapping[i])
        }
        return contiguous
    }
}
