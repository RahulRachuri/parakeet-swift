//
//  Adapted from FluidAudio (https://github.com/FluidInference/FluidAudio),
//  upstream commit 667181a, file
//  Sources/FluidAudio/ASR/Parakeet/SlidingWindow/CustomVocabulary/Rescorer/VocabularyRescorer.swift
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
//  Changes from upstream: the diagnostic candidate-evidence API
//  (`ctcTokenEvaluateCandidates` and everything it carries — evidence rows,
//  UTF-8 span alignment, legacy outcome reconciliation) is not ported. It is
//  opt-in upstream and inert when unused, so its removal cannot move a
//  replacement decision. `WordTiming` is built from this package's own decode
//  output rather than from upstream's `TokenTiming`.
//

import Foundation

/// CTC-based vocabulary rescoring.
///
/// The idea in one line: do not swap a word for a vocabulary term because the
/// two *look* alike — swap it only when a second, independent acoustic model
/// says the term is actually what was said there.
///
/// So this scores the vocabulary term AND the word already in the transcript
/// against the same CTC log-probabilities over the same frame window, and
/// replaces only when the term wins by more than a configured boost. That is
/// standard shallow fusion / CTC rescoring, and it is what stops a
/// string-similarity gate from rewriting every near-miss it finds.
public struct VocabularyRescorer: Sendable {

    let logger = RescoreLog("VocabularyRescorer")

    let spotter: CtcKeywordSpotter
    let vocabulary: CustomVocabularyContext
    let ctcTokenizer: CtcTokenizer?
    var debugMode: Bool { ContextBiasingConstants.debugEnabled }

    /// BK-tree candidate lookup, off by default (see `ContextBiasingConstants`).
    let useBKTree: Bool
    let bkTree: BKTree?
    let bkTreeMaxDistance: Int

    /// Rescoring behavior knobs.
    public struct Config: Sendable {
        /// Scale thresholds by a term's token count.
        public let useAdaptiveThresholds: Bool

        /// Token count treated as "reference length" for adaptive scaling.
        public let referenceTokenCount: Int

        /// Token-count pivot for the short-term cbw taper; `<= 1` disables it.
        /// Below the pivot a term's boost is scaled down so a short keyword
        /// cannot beat a correctly transcribed common word on the flat boost
        /// alone — it has to earn the margin acoustically.
        public let shortTermCbwTaperPivot: Int

        /// Exponent for that taper. Higher is more conservative.
        public let shortTermCbwTaperExponent: Float

        /// Similarity floor for a single-word spotter-anchored rescue.
        /// `0.0` disables the floor, leaving the rescue purely acoustic.
        public let spotterRescueMinSimilarity: Float

        /// Same floor for a multi-word rescue span.
        public let spotterRescueMultiWordMinSimilarity: Float

        /// Whether the spotter-anchored acoustic rescue runs at all. It
        /// recovers names the string-similarity gate cannot reach, but it is
        /// also the dominant source of short-keyword over-firing; on a short
        /// vocabulary turning it off is usually the right trade.
        public let spotterRescueEnabled: Bool

        public static let `default` = Config()

        public init(
            useAdaptiveThresholds: Bool = ContextBiasingConstants.defaultUseAdaptiveThresholds,
            referenceTokenCount: Int = ContextBiasingConstants.defaultReferenceTokenCount,
            shortTermCbwTaperPivot: Int = ContextBiasingConstants.defaultShortTermCbwTaperPivot,
            shortTermCbwTaperExponent: Float = ContextBiasingConstants.defaultShortTermCbwTaperExponent,
            spotterRescueMinSimilarity: Float = ContextBiasingConstants.defaultSpotterRescueMinSimilarity,
            spotterRescueMultiWordMinSimilarity: Float = ContextBiasingConstants
                .defaultSpotterRescueMultiWordMinSimilarity,
            spotterRescueEnabled: Bool = ContextBiasingConstants.defaultSpotterRescueEnabled
        ) {
            self.useAdaptiveThresholds = useAdaptiveThresholds
            self.referenceTokenCount = referenceTokenCount
            self.shortTermCbwTaperPivot = shortTermCbwTaperPivot
            self.shortTermCbwTaperExponent = shortTermCbwTaperExponent
            self.spotterRescueMinSimilarity = spotterRescueMinSimilarity
            self.spotterRescueMultiWordMinSimilarity = spotterRescueMultiWordMinSimilarity
            self.spotterRescueEnabled = spotterRescueEnabled
        }

        /// Context-biasing weight adjusted for a term's token count.
        ///
        /// Long terms get *more* boost, because a longer alignment accumulates
        /// more per-token scoring error. Short terms get *less*, because their
        /// per-token DP score is inflated by free-start alignment: a two-token
        /// keyword can pick its single best-matching pair of frames anywhere in
        /// the window and score near zero.
        public func adaptiveCbw(baseCbw: Float, tokenCount: Int) -> Float {
            guard useAdaptiveThresholds else { return baseCbw }

            let pivot = shortTermCbwTaperPivot
            if pivot > 1 && tokenCount < pivot {
                let ratio = Float(max(1, tokenCount)) / Float(pivot)
                return baseCbw * pow(ratio, shortTermCbwTaperExponent)
            }

            guard tokenCount > referenceTokenCount else { return baseCbw }
            let ratio = Float(tokenCount) / Float(referenceTokenCount)
            return baseCbw * (1.0 + log2(ratio) * 0.3)
        }
    }

    let config: Config

    /// Build a rescorer over a loaded spotter, vocabulary and CTC tokenizer.
    public static func create(
        spotter: CtcKeywordSpotter,
        vocabulary: CustomVocabularyContext,
        config: Config = .default,
        tokenizer: CtcTokenizer
    ) -> VocabularyRescorer {
        let useBKTree = ContextBiasingConstants.useBkTree
        return VocabularyRescorer(
            spotter: spotter,
            vocabulary: vocabulary,
            config: config,
            ctcTokenizer: tokenizer,
            useBKTree: useBKTree,
            bkTree: useBKTree ? BKTree(terms: vocabulary.terms) : nil,
            bkTreeMaxDistance: ContextBiasingConstants.bkTreeMaxDistance
        )
    }

    private init(
        spotter: CtcKeywordSpotter,
        vocabulary: CustomVocabularyContext,
        config: Config,
        ctcTokenizer: CtcTokenizer,
        useBKTree: Bool,
        bkTree: BKTree?,
        bkTreeMaxDistance: Int
    ) {
        self.spotter = spotter
        self.vocabulary = vocabulary
        self.config = config
        self.ctcTokenizer = ctcTokenizer
        self.useBKTree = useBKTree
        self.bkTree = bkTree
        self.bkTreeMaxDistance = bkTreeMaxDistance
    }

    // MARK: - Result types

    /// One word (or span) the rescorer decided about.
    public struct RescoringResult: Sendable {
        public let originalWord: String
        public let originalScore: Float
        public let replacementWord: String?
        public let replacementScore: Float?
        public let shouldReplace: Bool
        public let reason: String
    }

    /// The rewritten transcript and every replacement that produced it.
    public struct RescoreOutput: Sendable {
        public let text: String
        public let replacements: [RescoringResult]
        public let wasModified: Bool
    }

    /// Which discovery path proposed a candidate. Diagnostic only.
    public enum CandidateOrigin: String, Sendable, Equatable {
        /// Found by searching vocabulary terms around one transcript word.
        case wordCentric
        /// Found by a single-word term-centric match.
        case termCentricSingleWord
        /// Found by a multi-word term-centric match.
        case termCentricMultiWord
        /// Found by the CTC keyword-spotter rescue pass.
        case spotterRescue
    }

    /// One word of the transcript with the times the decode assigned it.
    public struct WordTiming: Sendable {
        public let word: String
        public let startTime: Double
        public let endTime: Double

        public init(word: String, startTime: Double, endTime: Double) {
            self.word = word
            self.startTime = startTime
            self.endTime = endTime
        }
    }
}
