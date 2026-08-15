//
//  Adapted from FluidAudio (https://github.com/FluidInference/FluidAudio),
//  upstream commit 667181a, file
//  Sources/FluidAudio/ASR/Parakeet/SlidingWindow/CustomVocabulary/ContextBiasingConstants.swift
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
//  Changes from upstream: TTS/diarization-adjacent constants dropped; the
//  environment-variable override names are kept verbatim so a tuning sweep
//  written against upstream reproduces here unchanged.
//

import Foundation

/// Centralized constants for the context-biasing (custom vocabulary) module.
///
/// ## Similarity threshold hierarchy
/// The similarity thresholds form a hierarchy from lenient to strict:
/// ```
/// 0.50 (floor) < 0.52 (default) < 0.55 (single-word) < 0.65 (alias) < 0.75 (length-ratio) < 0.80 (multi-word/short) < 0.85 (stopword)
/// ```
public enum ContextBiasingConstants {

    // MARK: - Token IDs

    /// Sentinel token ID meaning "matches any token at zero cost" in the CTC
    /// dynamic program (the `*` of NeMo's `ctc_word_spotter.py`).
    public static let wildcardTokenId: Int = -1

    /// Default CTC blank token ID. For `parakeet-ctc-110m` the vocabulary holds
    /// 1024 tokens (0…1023), so blank sits at 1024.
    public static let defaultBlankId: Int = 1024

    // MARK: - CTC score thresholds

    /// Default minimum CTC score for keyword-spotting detections. CTC scores are
    /// log-probabilities, so -15.0 is deliberately lenient: rescoring makes the
    /// final call.
    public static let defaultMinSpotterScore: Float = -15.0

    /// Default minimum CTC score for vocabulary context matching. Slightly
    /// stricter than the spotter floor, since this applies after detection.
    public static let defaultMinVocabCtcScore: Float = -12.0

    /// CTC softmax temperature. 1.0 is a standard softmax; lower sharpens the
    /// distribution, higher flattens it.
    public static let ctcTemperature: Float = 1.0

    /// Blank-bias correction subtracted from the blank token's log-probability.
    /// Positive values penalize blank; 0.0 leaves the distribution alone.
    public static let blankBias: Float = 0.0

    // MARK: - Similarity thresholds

    /// Absolute similarity floor for any vocabulary match, using Levenshtein
    /// similarity `1 - editDistance / maxLength`.
    public static let minSimilarityFloor: Float = 0.50

    /// Default minimum similarity for a term match, overridable per vocabulary
    /// and per term.
    public static let defaultMinSimilarity: Float = 0.52

    /// Default minimum combined (acoustic + string) confidence.
    public static let defaultMinCombinedConfidence: Float = 0.54

    /// Length ratio (`original.count / vocabTerm.count`) below which a stricter
    /// similarity is demanded, so that e.g. "and" cannot reach "Andre".
    public static let lengthRatioThreshold: Float = 0.75

    /// Similarity demanded of short words that also fail the length-ratio test.
    public static let shortWordSimilarity: Float = 0.80

    /// Similarity demanded of a multi-word span that contains a stopword.
    public static let stopwordSpanSimilarity: Float = 0.85

    // MARK: - Context-biasing weights

    /// Default context-biasing weight (a log-probability boost added to the
    /// vocabulary term's CTC score), per the NeMo context-biasing recipe.
    public static let defaultCbw: Float = 3.0

    /// Default alpha for weighted acoustic/LM score combination.
    public static let defaultAlpha: Float = 0.5

    /// Default CTC frame-alignment margin, in seconds, around a transcript word
    /// when searching for a vocabulary term (~1.25 CTC frames each side).
    public static let defaultMarginSeconds: Double = 0.10

    // MARK: - Vocabulary size

    /// Above this term count a vocabulary counts as "large" and tightens.
    public static let largeVocabThreshold: Int = 10

    /// Above this term count a vocabulary counts as "extra-large": the
    /// distractor density justifies a stricter similarity gate again.
    public static let extraLargeVocabThreshold: Int = 100

    /// Vocabulary-size-aware rescorer parameters.
    public struct VocabSizeConfig: Sendable {
        public let minSimilarity: Float
        public let cbw: Float
    }

    /// Rescorer configuration tuned for a given vocabulary size. All sizes
    /// converge on `cbw = 4.5`; only the similarity gate moves.
    public static func rescorerConfig(forVocabSize size: Int) -> VocabSizeConfig {
        let isExtraLarge = size > extraLargeVocabThreshold
        let isLarge = size > largeVocabThreshold
        let minSimilarity: Float
        if isExtraLarge {
            minSimilarity = 0.60
        } else if isLarge {
            minSimilarity = 0.55
        } else {
            minSimilarity = 0.50
        }
        return VocabSizeConfig(minSimilarity: minSimilarity, cbw: 4.5)
    }

    /// Token count past which a multi-token phrase's score threshold relaxes.
    public static let baselineTokenCountForThreshold: Int = 3

    /// How much the threshold relaxes per token beyond the baseline.
    public static let thresholdRelaxationPerToken: Float = 1.0

    /// Reference token count for adaptive threshold scaling.
    public static let defaultReferenceTokenCount: Int = 3

    // MARK: - Short-term over-fire controls (opt-in)
    //
    // The blank-aware DP score is a per-token average log-prob, so a short
    // keyword can free-start align to its single best-matching frame run and
    // beat a correctly transcribed common word. Gating that hard enough to
    // suppress short-vocab false positives also costs recall on distinctive
    // names, so these controls DEFAULT TO DISABLED and are opt-in.

    /// Token-count pivot for the short-term cbw taper; `<= 1` disables it.
    /// When enabled (e.g. 5), terms with fewer tokens than the pivot have their
    /// boost scaled by `(tokenCount / pivot) ** exponent`.
    public static var defaultShortTermCbwTaperPivot: Int {
        envInt("FLUID_CBW_TAPER_PIVOT") ?? 1
    }

    /// Exponent for the short-term cbw taper. Higher is more conservative.
    public static var defaultShortTermCbwTaperExponent: Float {
        envFloat("FLUID_CBW_TAPER_EXP") ?? 2.0
    }

    /// Minimum string similarity for a single-word spotter-anchored rescue.
    /// `0.0` disables the floor, leaving the rescue purely acoustic.
    public static var defaultSpotterRescueMinSimilarity: Float {
        envFloat("FLUID_SPOTTER_MIN_SIM") ?? 0.0
    }

    /// Same floor for a multi-word rescue span, which is the more error-prone
    /// case (several words collapsing into one term).
    public static var defaultSpotterRescueMultiWordMinSimilarity: Float {
        envFloat("FLUID_SPOTTER_MIN_SIM_MULTI") ?? 0.0
    }

    /// Whether the spotter-anchored acoustic rescue pass runs at all. `true`
    /// recovers names the string-similarity gate cannot reach, but it is also
    /// the dominant source of short-keyword over-firing; turning it off is the
    /// single biggest precision win on short vocabularies.
    public static var defaultSpotterRescueEnabled: Bool {
        envBool("FLUID_SPOTTER_RESCUE") ?? true
    }

    /// Whether the rescorer narrates its decisions on stderr.
    public static var debugEnabled: Bool {
        envBool("PARAKEET_VOCAB_DEBUG") ?? false
    }

    private static func envFloat(_ name: String) -> Float? {
        guard let raw = ProcessInfo.processInfo.environment[name], let value = Float(raw) else { return nil }
        return value
    }

    private static func envBool(_ name: String) -> Bool? {
        guard let raw = ProcessInfo.processInfo.environment[name]?.lowercased() else { return nil }
        switch raw {
        case "1", "true", "yes", "on": return true
        case "0", "false", "no", "off": return false
        default: return nil
        }
    }

    private static func envInt(_ name: String) -> Int? {
        guard let raw = ProcessInfo.processInfo.environment[name], let value = Int(raw) else { return nil }
        return value
    }

    /// Adaptive thresholds scale by token count, letting longer terms match a
    /// little more loosely per character.
    public static let defaultUseAdaptiveThresholds: Bool = true

    // MARK: - Word length thresholds

    /// At or below this character count a word counts as "short" and gets a
    /// stricter similarity requirement.
    public static let shortWordMaxLength: Int = 4

    // MARK: - BK-tree (experimental)

    /// Whether the word-centric path uses a BK-tree for candidate lookup.
    /// Disabled: the term-centric linear scan is the benchmarked default.
    public static let useBkTree: Bool = false

    /// Maximum edit distance for BK-tree fuzzy matching.
    public static let bkTreeMaxDistance: Int = 3

    // MARK: - CTC audio framing

    /// Sample rate every stage of this module assumes.
    public static let sampleRate: Int = 16_000

    /// Samples the CTC mel front end consumes in one pass (15 s at 16 kHz).
    public static let maxModelSamples: Int = 240_000

    /// Overlap between consecutive CTC windows on long audio (2 s at 16 kHz).
    public static let chunkOverlapSamples: Int = 32_000

    /// SentencePiece word-boundary marker.
    public static let sentencePieceWordBoundary: String = "\u{2581}"
}

/// Minimal stderr logger. The upstream module logs through the host app's
/// logging stack; here a category-tagged stderr line is the whole contract, and
/// it stays silent unless `PARAKEET_VOCAB_DEBUG` is set.
struct RescoreLog: Sendable {
    let category: String
    init(_ category: String) { self.category = category }

    func debug(_ message: @autoclosure () -> String) {
        guard ContextBiasingConstants.debugEnabled else { return }
        write("debug", message())
    }

    func info(_ message: @autoclosure () -> String) {
        guard ContextBiasingConstants.debugEnabled else { return }
        write("info", message())
    }

    func warning(_ message: @autoclosure () -> String) {
        write("warning", message())
    }

    private func write(_ level: String, _ message: String) {
        FileHandle.standardError.write(Data("[\(category)/\(level)] \(message)\n".utf8))
    }
}
