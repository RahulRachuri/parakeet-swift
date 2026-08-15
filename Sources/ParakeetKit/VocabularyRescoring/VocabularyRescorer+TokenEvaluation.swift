//
//  Adapted from FluidAudio (https://github.com/FluidInference/FluidAudio),
//  upstream commit 667181a, file
//  Sources/FluidAudio/ASR/Parakeet/SlidingWindow/CustomVocabulary/Rescorer/VocabularyRescorer+TokenEvaluation.swift
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
//  Changes from upstream: the evidence-row constructors are not ported (see
//  VocabularyRescorer.swift). The comparison itself is unmodified.
//

import Foundation

extension VocabularyRescorer {

    @inline(__always)
    private func debugLog(_ message: @autoclosure () -> String) {
        guard debugMode else { return }
        logger.debug(message())
    }

    // MARK: - CTC match evaluation

    /// The decision: score the vocabulary term and the transcript phrase
    /// against the same frames, and replace only if the boosted term wins.
    ///
    /// This is the whole point of the rescorer. Everything upstream of it —
    /// similarity gates, stopword rules, length ratios — only decides which
    /// pairs are *worth asking about*; this is what actually answers.
    func evaluateCTCMatch(
        candidate: CTCMatchCandidate,
        logProbs: [[Float]],
        frameDuration: Double,
        cbw: Float,
        marginSeconds: Double
    ) -> CTCMatchResult {
        let marginFrames = Int(marginSeconds / frameDuration)
        let spanStartFrame = Int(candidate.spanStartTime / frameDuration)
        let spanEndFrame = Int(candidate.spanEndTime / frameDuration)

        let searchStart = max(0, spanStartFrame - marginFrames)
        let searchEnd = min(logProbs.count, spanEndFrame + marginFrames)

        // Score the term two ways and keep the better: with the leading `▁`
        // word-boundary token, and without it. A compound match does not begin
        // at a word boundary in the audio, so that leading token has no
        // acoustic counterpart and drags the alignment down unfairly.
        var vocabTokensUsed = candidate.vocabTokens
        var vocabCtcScore: Float = -.infinity
        do {
            let (score, _, _) = spotter.ctcWordSpotConstrained(
                logProbs: logProbs,
                keywordTokens: candidate.vocabTokens,
                searchStartFrame: searchStart,
                searchEndFrame: searchEnd
            )
            vocabCtcScore = score
        }
        if let tokenizer = ctcTokenizer {
            let altTokens = tokenizer.encodeWithoutBoundary(candidate.vocabTerm)
            if !altTokens.isEmpty && altTokens != candidate.vocabTokens {
                let (altScore, _, _) = spotter.ctcWordSpotConstrained(
                    logProbs: logProbs,
                    keywordTokens: altTokens,
                    searchStartFrame: searchStart,
                    searchEndFrame: searchEnd
                )
                if altScore > vocabCtcScore {
                    vocabCtcScore = altScore
                    vocabTokensUsed = altTokens
                }
            }
        }

        guard let tokenizer = ctcTokenizer else {
            debugLog("  [warn] no tokenizer, skipping CTC comparison for '\(candidate.originalPhrase)'")
            return CTCMatchResult(
                shouldReplace: false,
                comparisonWasPerformed: false,
                originalScore: -Float.infinity,
                boostedVocabScore: vocabCtcScore,
                replacement: candidate.vocabTerm,
                reason: "no tokenizer available"
            )
        }

        let originalTokens = tokenizer.encode(candidate.originalPhrase)
        guard !originalTokens.isEmpty else {
            debugLog("  [warn] empty tokens for '\(candidate.originalPhrase)', skipping")
            return CTCMatchResult(
                shouldReplace: false,
                comparisonWasPerformed: false,
                originalScore: -Float.infinity,
                boostedVocabScore: vocabCtcScore,
                replacement: candidate.vocabTerm,
                reason: "empty tokens for the original phrase"
            )
        }

        let (originalCtcScore, _, _) = spotter.ctcWordSpotConstrained(
            logProbs: logProbs,
            keywordTokens: originalTokens,
            searchStartFrame: searchStart,
            searchEndFrame: searchEnd
        )

        // The boost is scaled to whichever tokenization actually scored best.
        let adaptiveCbwValue = config.adaptiveCbw(baseCbw: cbw, tokenCount: vocabTokensUsed.count)
        let boostedVocabScore = vocabCtcScore + adaptiveCbwValue
        let shouldReplace = boostedVocabScore > originalCtcScore

        if debugMode {
            let label = candidate.spanLength > 1 ? "[multi] " : ""
            debugLog(
                "  \(label)'\(candidate.originalPhrase)' vs '\(candidate.vocabTerm)' "
                    + "(sim=\(String(format: "%.2f", candidate.similarity)), span=\(candidate.spanLength), "
                    + "\(String(format: "%.2f", candidate.spanStartTime))-"
                    + "\(String(format: "%.2f", candidate.spanEndTime))s)")
            let variantLabel = vocabTokensUsed == candidate.vocabTokens ? "boundary" : "no-boundary"
            debugLog(
                "    original=\(String(format: "%.2f", originalCtcScore)) "
                    + "vocab[\(variantLabel)]=\(String(format: "%.2f", vocabCtcScore)) "
                    + "+ cbw \(String(format: "%.2f", adaptiveCbwValue)) "
                    + "= \(String(format: "%.2f", boostedVocabScore)) "
                    + "-> \(shouldReplace ? "REPLACE" : "KEEP")")
        }

        let firstOriginalWord =
            candidate.originalPhrase.split(separator: " ").first.map(String.init)
            ?? candidate.originalPhrase
        let replacement = preserveCapitalization(original: firstOriginalWord, replacement: candidate.vocabTerm)

        let reasonPrefix = candidate.spanLength > 1 ? "CTC-vs-CTC (multi-word)" : "CTC-vs-CTC"
        let reason = Self.ctcComparisonReason(
            prefix: reasonPrefix,
            vocabularyTerm: candidate.vocabTerm,
            boostedVocabularyScore: boostedVocabScore,
            originalPhrase: candidate.originalPhrase,
            originalScore: originalCtcScore,
            comparisonPassed: shouldReplace
        )

        return CTCMatchResult(
            shouldReplace: shouldReplace,
            comparisonWasPerformed: true,
            originalScore: originalCtcScore,
            boostedVocabScore: boostedVocabScore,
            replacement: replacement,
            reason: reason
        )
    }

    /// Human-readable account of one comparison.
    static func ctcComparisonReason(
        prefix: String,
        vocabularyTerm: String,
        boostedVocabularyScore: Float,
        originalPhrase: String,
        originalScore: Float,
        comparisonPassed: Bool
    ) -> String {
        let comparisonOperator = comparisonPassed ? ">" : "<="
        return
            "\(prefix): '\(vocabularyTerm)'=\(String(format: "%.2f", boostedVocabularyScore)) "
            + "\(comparisonOperator) '\(originalPhrase)'=\(String(format: "%.2f", originalScore))"
    }

    // MARK: - Replacement application

    /// Write one accepted replacement into the working word list.
    ///
    /// A multi-word span collapses to its first slot; the rest are blanked and
    /// filtered out when the transcript is rebuilt.
    func applyReplacement(
        result: CTCMatchResult,
        candidate: CTCMatchCandidate,
        modifiedWords: inout [String],
        replacedIndices: inout Set<Int>,
        replacements: inout [RescoringResult]
    ) {
        guard let firstIdx = candidate.spanIndices.first else { return }

        modifiedWords[firstIdx] = result.replacement
        for idx in candidate.spanIndices.dropFirst() {
            modifiedWords[idx] = ""
        }
        for idx in candidate.spanIndices {
            replacedIndices.insert(idx)
        }

        replacements.append(
            RescoringResult(
                originalWord: candidate.originalPhrase,
                originalScore: result.originalScore,
                replacementWord: result.replacement,
                replacementScore: result.boostedVocabScore,
                shouldReplace: true,
                reason: result.reason
            ))
    }

    // MARK: - Validation rules

    /// Stopword rules.
    ///
    /// A lone stopword is never replaced: substituting a proper noun over "the"
    /// or "and" is the single most damaging thing this pass can do, and no
    /// amount of acoustic evidence makes it worth the risk. A multi-word span
    /// containing a function word merely has its threshold raised.
    func checkStopwordRules(
        normalizedWord: String,
        spanLength: Int,
        spanWords: [String],
        vocabTerm: String,
        currentSimilarity: Float
    ) -> (shouldSkip: Bool, adjustedMinSimilarity: Float) {
        var minSimilarity = currentSimilarity

        if spanLength == 1 && Self.stopwords.contains(normalizedWord) {
            debugLog("    [stopword] '\(normalizedWord)' is a stopword, not replacing with '\(vocabTerm)'")
            return (shouldSkip: true, adjustedMinSimilarity: minSimilarity)
        }

        if spanLength >= 2 {
            let containsStopword = spanWords.contains { Self.multiWordStopwords.contains($0) }
            if containsStopword {
                minSimilarity = max(minSimilarity, ContextBiasingConstants.stopwordSpanSimilarity)
                debugLog(
                    "    [stopword] span '\(spanWords.joined(separator: " "))' contains a stopword, "
                        + "threshold raised to \(String(format: "%.2f", minSimilarity))")
            }
        }

        return (shouldSkip: false, adjustedMinSimilarity: minSimilarity)
    }

    /// Length-ratio rule for single words.
    ///
    /// A short word that is also much shorter than the term must clear a much
    /// higher similarity bar, because edit similarity is cheap to satisfy over
    /// three or four characters.
    func checkLengthRatioRules(
        normalizedWord: String,
        vocabTerm: String,
        currentSimilarity: Float,
        minSimilarity: Float
    ) -> Float {
        let lengthRatio = Float(normalizedWord.count) / Float(vocabTerm.count)
        if lengthRatio < ContextBiasingConstants.lengthRatioThreshold
            && normalizedWord.count <= ContextBiasingConstants.shortWordMaxLength
        {
            let adjusted = max(minSimilarity, ContextBiasingConstants.shortWordSimilarity)
            if currentSimilarity >= minSimilarity {
                debugLog(
                    "    [length] '\(normalizedWord)' too short (ratio=\(String(format: "%.2f", lengthRatio))), "
                        + "threshold raised to \(String(format: "%.2f", adjusted))")
            }
            return adjusted
        }
        return minSimilarity
    }
}
