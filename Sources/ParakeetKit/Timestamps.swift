import Foundation

/// One word with absolute times (seconds into the input file).
public struct WordTimestamp: Sendable, Equatable {
    public var word: String
    public var start: Double
    public var end: Double
    public init(word: String, start: Double, end: Double) {
        self.word = word; self.start = start; self.end = end
    }
}

/// Token-emission frames → word-level times. Purely a *reading* of the decode's additive
/// metadata (`ChunkResult.frameIndices`); nothing here feeds back into the loop.
///
/// The rules match NeMo / parakeet-mlx:
///  * a token whose SentencePiece piece begins with "▁" starts a new word;
///  * a word's start is its first token's time: `chunkStart + frame × 0.08 s`
///    (encoder frames are 80 ms apart: 8× subsampling of the 10 ms mel hop);
///  * a word's end is the *next* word's start, and the chunk end for the last word.
///
/// One wrinkle the vocab audit surfaced: the vocabulary contains a bare "▁" piece. A run of
/// those would form a "word" whose text is empty after trimming; such groups are dropped, so
/// the surrounding words simply meet across the gap.
public enum WordTimestamps {
    /// Seconds of audio per encoder frame (8 × the 10 ms mel hop).
    public static let secondsPerFrame = 0.08

    public static func words(tokens: [Int], frameIndices: [Int],
                             tokenizer: ParakeetTokenizer,
                             chunkStart: Double, chunkEnd: Double) -> [WordTimestamp] {
        precondition(tokens.count == frameIndices.count,
                     "tokens and frameIndices must be parallel")
        guard !tokens.isEmpty else { return [] }

        // Split the token list into word groups: a boundary sits before every ▁-token.
        var groups: [(text: String, startFrame: Int)] = []
        var currentPieces = ""
        var currentStart = frameIndices[0]
        func close() {
            let text = currentPieces
                .replacingOccurrences(of: "\u{2581}", with: " ")
                .trimmingCharacters(in: .whitespaces)
            if !text.isEmpty { groups.append((text, currentStart)) }
        }
        for (i, id) in tokens.enumerated() {
            guard let piece = tokenizer.piece(id) else { continue }   // specials never time a word
            if tokenizer.startsWord(id) && !currentPieces.isEmpty {
                close()
                currentPieces = ""
                currentStart = frameIndices[i]
            }
            if currentPieces.isEmpty { currentStart = frameIndices[i] }
            currentPieces += piece
        }
        close()

        // Frames → absolute seconds; end = next word's start, chunk end for the last.
        func time(_ frame: Int) -> Double {
            min(max(chunkStart + Double(frame) * secondsPerFrame, chunkStart), chunkEnd)
        }
        return groups.indices.map { i in
            let start = time(groups[i].startFrame)
            let end = i + 1 < groups.count ? time(groups[i + 1].startFrame) : chunkEnd
            return WordTimestamp(word: groups[i].text, start: start, end: max(end, start))
        }
    }
}
