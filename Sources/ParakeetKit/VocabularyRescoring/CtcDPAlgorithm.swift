//
//  Adapted from FluidAudio (https://github.com/FluidInference/FluidAudio),
//  upstream commit 667181a, file
//  Sources/FluidAudio/ASR/Parakeet/SlidingWindow/CustomVocabulary/WordSpotting/CtcDPAlgorithm.swift
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
//  Ported verbatim apart from naming: this is the arithmetic every replacement
//  decision rests on, so it is deliberately the least-adapted file here.
//

import Foundation

/// The CTC word-spotting dynamic program from NeMo's `ctc_word_spotter.py`
/// (arXiv:2406.07096).
///
/// Unlike a naive token-only DP, this runs on the **blank-expanded** symbol
/// sequence `[B, t1, B, t2, …, tN, B]` and accumulates blank emission
/// log-probabilities along stay paths. That is what makes the score
/// probabilistically meaningful, and what correctly forces a blank between
/// repeated tokens.
enum CtcDPAlgorithm {

    /// Wildcard token ID: matches anything at zero cost.
    static let wildcardTokenId = ContextBiasingConstants.wildcardTokenId

    // MARK: - Expanded symbol helpers

    private enum ExpandedSymbol {
        case blank
        case token(Int)
        case wildcard
    }

    /// Build `[B, t1, B, t2, …, tN, B]`, of length `2N + 1`.
    private static func buildExpandedSequence(_ keywordTokens: [Int]) -> [ExpandedSymbol] {
        var s: [ExpandedSymbol] = []
        s.reserveCapacity(2 * keywordTokens.count + 1)
        for id in keywordTokens {
            s.append(.blank)
            s.append(id == wildcardTokenId ? .wildcard : .token(id))
        }
        s.append(.blank)
        return s
    }

    /// Emission log-probability for a symbol at a frame. Wildcards are free;
    /// an out-of-range blank id emits zero, an out-of-range token is impossible.
    @inline(__always)
    private static func emissionLogProb(
        symbol: ExpandedSymbol,
        frame: [Float],
        blankId: Int
    ) -> Float {
        switch symbol {
        case .blank:
            return blankId >= 0 && blankId < frame.count ? frame[blankId] : 0
        case .token(let id):
            return id >= 0 && id < frame.count ? frame[id] : -Float.greatestFiniteMagnitude
        case .wildcard:
            return 0
        }
    }

    /// Whether the DP may step from `s-2` straight to `s`, skipping the blank
    /// between them. Allowed only when `s` is a non-blank symbol distinct from
    /// `s-2`: repeated tokens MUST pass through the intervening blank.
    @inline(__always)
    private static func canSkipBlank(_ s: [ExpandedSymbol], at idx: Int) -> Bool {
        guard idx >= 2 else { return false }
        switch s[idx] {
        case .blank:
            return false
        case .token(let cur):
            if case .token(let prev) = s[idx - 2], prev == cur { return false }
            return true
        case .wildcard:
            if case .wildcard = s[idx - 2] { return false }
            return true
        }
    }

    // MARK: - Core DP

    /// Build the DP table over the blank-expanded sequence and project it back
    /// to the public "n tokens consumed" view.
    ///
    /// Three transitions are evaluated per state: **stay** at `s` (adds the
    /// symbol's emission cost), **advance** from `s-1`, and **skip blank** from
    /// `s-2` where the CTC rule permits it. The projection is
    /// `dp[t][n] = max(dpI[t][2n-1], dpI[t][2n])` — the better of "ended on
    /// token n" and "ended on the blank after token n". Free start is preserved
    /// with `dp[t][0] = 0` at every `t`.
    ///
    /// - Returns: `dp` (best raw log-prob for consuming `n` tokens by frame `t`,
    ///   blank emissions included), `backtrack` (inferred keyword start frame)
    ///   and `lastMatch` (frame of the most recent non-blank emission).
    static func fillDPTable(
        logProbs: [[Float]],
        keywordTokens: [Int],
        blankId: Int = ContextBiasingConstants.defaultBlankId
    ) -> (dp: [[Float]], backtrack: [[Int]], lastMatch: [[Int]]) {
        let T = logProbs.count
        let N = keywordTokens.count
        let neg = -Float.greatestFiniteMagnitude

        var dp = Array(repeating: Array(repeating: neg, count: N + 1), count: T + 1)
        var backtrack = Array(repeating: Array(repeating: 0, count: N + 1), count: T + 1)
        var lastMatch = Array(repeating: Array(repeating: 0, count: N + 1), count: T + 1)

        // Free start: matching zero tokens scores 0 at any frame.
        for t in 0...T { dp[t][0] = 0 }
        if N == 0 { return (dp, backtrack, lastMatch) }

        let s = buildExpandedSequence(keywordTokens)
        let sLen = s.count  // 2N + 1

        var dpI = Array(repeating: Array(repeating: neg, count: sLen), count: T + 1)
        var startI = Array(repeating: Array(repeating: 0, count: sLen), count: T + 1)
        var lastTokI = Array(repeating: Array(repeating: 0, count: sLen), count: T + 1)
        // s = 0 is the initial blank; free start means any frame can begin the
        // keyword, so the initial state scores 0 and records the current frame.
        for t in 0...T {
            dpI[t][0] = 0
            startI[t][0] = t
        }

        for t in 1...T {
            let frame = logProbs[t - 1]
            for sIdx in 1..<sLen {
                let sym = s[sIdx]
                let emitLogProb = emissionLogProb(symbol: sym, frame: frame, blankId: blankId)
                let isWildcard: Bool = { if case .wildcard = sym { return true } else { return false } }()
                let isToken: Bool = { if case .token = sym { return true } else { return false } }()
                let added: Float = isWildcard ? 0 : emitLogProb

                let stay = dpI[t - 1][sIdx]
                let advance = dpI[t - 1][sIdx - 1]
                let skipBlank = canSkipBlank(s, at: sIdx) ? dpI[t - 1][sIdx - 2] : neg

                var bestPred = stay
                var predKind = 0  // 0 = stay, 1 = advance, 2 = skip-blank
                if advance > bestPred {
                    bestPred = advance
                    predKind = 1
                }
                if skipBlank > bestPred {
                    bestPred = skipBlank
                    predKind = 2
                }

                if bestPred <= neg / 2 {
                    dpI[t][sIdx] = neg
                    continue
                }

                dpI[t][sIdx] = bestPred + added
                let isMatchFrame = isToken || isWildcard

                switch predKind {
                case 0:
                    startI[t][sIdx] = startI[t - 1][sIdx]
                    lastTokI[t][sIdx] = isMatchFrame ? t : lastTokI[t - 1][sIdx]
                case 1:
                    if sIdx == 1 {
                        // First non-blank symbol: the keyword starts here.
                        startI[t][sIdx] = t - 1
                    } else {
                        startI[t][sIdx] = startI[t - 1][sIdx - 1]
                    }
                    lastTokI[t][sIdx] = isMatchFrame ? t : lastTokI[t - 1][sIdx - 1]
                default:
                    startI[t][sIdx] = startI[t - 1][sIdx - 2]
                    lastTokI[t][sIdx] = isMatchFrame ? t : lastTokI[t - 1][sIdx - 2]
                }
            }
        }

        for t in 0...T {
            for n in 1...N {
                let sTok = 2 * n - 1
                let sBlank = 2 * n
                let scTok = sTok < sLen ? dpI[t][sTok] : neg
                let scBlank = sBlank < sLen ? dpI[t][sBlank] : neg
                if scTok >= scBlank {
                    dp[t][n] = scTok
                    backtrack[t][n] = startI[t][sTok]
                    lastMatch[t][n] = lastTokI[t][sTok]
                } else {
                    dp[t][n] = scBlank
                    backtrack[t][n] = startI[t][sBlank]
                    lastMatch[t][n] = lastTokI[t][sBlank]
                }
            }
        }

        return (dp, backtrack, lastMatch)
    }

    /// Non-wildcard token count, used to normalize scores.
    static func nonWildcardCount(_ keywordTokens: [Int]) -> Int {
        keywordTokens.filter { $0 != wildcardTokenId }.count
    }

    // MARK: - Word spotting

    /// Spot a keyword inside `[searchStartFrame, searchEndFrame)`.
    ///
    /// The returned score is normalized by the non-wildcard token count: the
    /// per-token average log-probability of the best alignment, blank emission
    /// costs included. Frames are returned in global coordinates.
    static func ctcWordSpotConstrained(
        logProbs: [[Float]],
        keywordTokens: [Int],
        searchStartFrame: Int,
        searchEndFrame: Int,
        blankId: Int = ContextBiasingConstants.defaultBlankId
    ) -> (score: Float, startFrame: Int, endFrame: Int) {
        let T = logProbs.count
        let N = keywordTokens.count

        let clampedStart = max(0, searchStartFrame)
        let clampedEnd = min(T, searchEndFrame)

        if N == 0 || clampedEnd <= clampedStart {
            return (-Float.infinity, clampedStart, clampedStart)
        }

        let windowLogProbs = Array(logProbs[clampedStart..<clampedEnd])
        let windowT = windowLogProbs.count

        if windowT < N {
            return (-Float.infinity, clampedStart, clampedStart)
        }

        let (dp, backtrack, lastMatch) = fillDPTable(
            logProbs: windowLogProbs,
            keywordTokens: keywordTokens,
            blankId: blankId
        )

        var bestEnd = 0
        var bestScore = -Float.greatestFiniteMagnitude
        for t in N...windowT where dp[t][N] > bestScore {
            bestScore = dp[t][N]
            bestEnd = t
        }

        let bestStart = backtrack[bestEnd][N]
        let actualEndFrame = lastMatch[bestEnd][N]

        let normFactor = nonWildcardCount(keywordTokens)
        let normalizedScore = normFactor > 0 ? bestScore / Float(normFactor) : bestScore

        return (normalizedScore, clampedStart + bestStart, clampedStart + actualEndFrame)
    }

    /// Find ALL occurrences of a keyword: every local maximum of the normalized
    /// score that clears `minScore`, optionally merging overlapping hits.
    static func ctcWordSpotMultiple(
        logProbs: [[Float]],
        keywordTokens: [Int],
        minScore: Float = ContextBiasingConstants.defaultMinSpotterScore,
        mergeOverlap: Bool = true,
        blankId: Int = ContextBiasingConstants.defaultBlankId
    ) -> [(score: Float, startFrame: Int, endFrame: Int)] {
        let T = logProbs.count
        let N = keywordTokens.count

        if N == 0 || T == 0 { return [] }

        let (dp, backtrack, lastMatch) = fillDPTable(
            logProbs: logProbs,
            keywordTokens: keywordTokens,
            blankId: blankId
        )

        let wildcardFreeCount = nonWildcardCount(keywordTokens)
        let normFactor = wildcardFreeCount > 0 ? Float(wildcardFreeCount) : 1.0

        var candidates: [(score: Float, startFrame: Int, endFrame: Int)] = []
        guard T >= N else { return [] }

        for t in N...T {
            let normalizedScore = dp[t][N] / normFactor
            let prevScore = t > N ? dp[t - 1][N] / normFactor : -Float.greatestFiniteMagnitude
            let nextScore = t < T ? dp[t + 1][N] / normFactor : -Float.greatestFiniteMagnitude

            let isLocalMax = normalizedScore >= prevScore && normalizedScore > nextScore
            let meetsThreshold = normalizedScore >= minScore

            if isLocalMax && meetsThreshold {
                candidates.append(
                    (score: normalizedScore, startFrame: backtrack[t][N], endFrame: lastMatch[t][N]))
            }
        }

        if candidates.isEmpty {
            var bestEnd = 0
            var bestScore = -Float.greatestFiniteMagnitude
            for t in N...T {
                let normalizedScore = dp[t][N] / normFactor
                if normalizedScore > bestScore {
                    bestScore = normalizedScore
                    bestEnd = t
                }
            }
            if bestScore >= minScore {
                candidates.append(
                    (score: bestScore, startFrame: backtrack[bestEnd][N], endFrame: lastMatch[bestEnd][N]))
            }
        }

        guard mergeOverlap else { return candidates }

        let sorted = candidates.sorted { $0.startFrame < $1.startFrame }
        var merged: [(score: Float, startFrame: Int, endFrame: Int)] = []
        for candidate in sorted {
            if let last = merged.last, candidate.startFrame <= last.endFrame {
                var best = candidate.score > last.score ? candidate : last
                best.endFrame = max(last.endFrame, candidate.endFrame)
                merged[merged.count - 1] = best
            } else {
                merged.append(candidate)
            }
        }
        return merged
    }
}
