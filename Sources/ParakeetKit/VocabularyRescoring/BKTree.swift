//
//  Adapted from FluidAudio (https://github.com/FluidInference/FluidAudio),
//  upstream commit 667181a, files
//  Sources/FluidAudio/ASR/Parakeet/SlidingWindow/CustomVocabulary/BKTree/BKTree.swift
//  and Sources/FluidAudio/Shared/StringUtils.swift (levenshteinDistance).
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

/// Levenshtein edit distance. The similarity gate the whole rescorer is built
/// on is `1 - distance / maxLength`, so this is the measure that decides which
/// vocabulary terms a transcript word is even allowed to become.
enum StringUtils {
    static func levenshteinDistance<T: Equatable>(_ a: [T], _ b: [T]) -> Int {
        let m = a.count
        let n = b.count
        guard m > 0 else { return n }
        guard n > 0 else { return m }

        // Two rolling rows rather than the full (m+1)×(n+1) table: the classic
        // DP only ever reads the previous row, and this runs tens of thousands
        // of times per chapter.
        var previous = Array(0...n)
        var current = [Int](repeating: 0, count: n + 1)

        for i in 1...m {
            current[0] = i
            for j in 1...n {
                let cost = a[i - 1] == b[j - 1] ? 0 : 1
                current[j] = min(
                    previous[j] + 1,  // deletion
                    current[j - 1] + 1,  // insertion
                    previous[j - 1] + cost  // substitution
                )
            }
            swap(&previous, &current)
        }
        return previous[n]
    }

    /// Character-level distance between two strings.
    static func levenshteinDistance(_ a: String, _ b: String) -> Int {
        levenshteinDistance(Array(a), Array(b))
    }
}

/// A BK-tree (Burkhard-Keller tree) for approximate string matching.
///
/// BK-trees organize strings by edit distance, so a fuzzy query examines
/// roughly O(log N) terms instead of all N. Nodes are immutable, which makes
/// the whole tree `Sendable` without qualification.
///
/// Only the experimental word-centric rescoring path uses this; the default
/// term-centric path scans the vocabulary linearly, which is the faster shape
/// when the vocabulary is small and the transcript is long.
public struct BKTree: Sendable {

    private struct Node: Sendable {
        let term: CustomVocabularyTerm
        let normalizedText: String
        let children: [Int: Node]  // edge distance → child
    }

    public struct SearchResult: Sendable {
        public let term: CustomVocabularyTerm
        public let normalizedText: String
        public let distance: Int
    }

    private let root: Node?
    private let termCount: Int

    public init(terms: [CustomVocabularyTerm]) {
        self.termCount = terms.count
        self.root = Self.buildTree(from: terms.map { ($0, $0.textLowercased) })
    }

    private static func buildTree(from terms: [(CustomVocabularyTerm, String)]) -> Node? {
        guard let first = terms.first else { return nil }

        var groups: [Int: [(CustomVocabularyTerm, String)]] = [:]
        for item in terms.dropFirst() {
            let dist = StringUtils.levenshteinDistance(item.1, first.1)
            groups[dist, default: []].append(item)
        }

        var children: [Int: Node] = [:]
        for (dist, group) in groups {
            if let child = buildTree(from: group) { children[dist] = child }
        }

        return Node(term: first.0, normalizedText: first.1, children: children)
    }

    /// All terms within `maxDistance` edits of `query`.
    public func search(query: String, maxDistance: Int) -> [SearchResult] {
        guard let root else { return [] }

        let normalizedQuery = query.lowercased()
        var results: [SearchResult] = []
        var stack: [Node] = [root]

        while let node = stack.popLast() {
            let distance = StringUtils.levenshteinDistance(normalizedQuery, node.normalizedText)
            if distance <= maxDistance {
                results.append(
                    SearchResult(term: node.term, normalizedText: node.normalizedText, distance: distance))
            }

            // The BK-tree property: only edges within `maxDistance` of the
            // measured distance can hold a match.
            let minEdge = max(0, distance - maxDistance)
            let maxEdge = distance + maxDistance
            for (edge, child) in node.children where edge >= minEdge && edge <= maxEdge {
                stack.append(child)
            }
        }

        return results
    }

    public var isEmpty: Bool { root == nil }
    public var count: Int { termCount }
}
