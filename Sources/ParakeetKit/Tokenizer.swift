import Foundation

/// The SentencePiece/BPE detokenizer for Parakeet-v2, driven straight off `tokenizer.json`.
///
/// We only ever *decode* (the TDT loop produces ids), so this deliberately implements just the
/// decode side, and it implements exactly what the validated bench harness used:
/// `tokenizers.Tokenizer.decode(ids)` with a `Metaspace` decoder
/// (`replacement = "▁"`, `prepend_scheme = "always"`).
///
/// The rule, from the Rust `Metaspace::decode_chain`:
///   * every "▁" becomes a space …
///   * … except in the *first* token of the sequence, where it is dropped outright;
///   * the pieces are then concatenated with no separator.
///
/// ⚠️ The trap this avoids: `transformers`' Parakeet `PreTrainedTokenizer.decode()` collapses
/// *consecutive duplicate ids* before detokenising, turning "coffee" into "coffe". Two
/// byte-identical token streams scored 0.33 % WER apart because of it. Nothing here collapses
/// anything; this file is the re-decode semantics the bench standardised on.
public struct ParakeetTokenizer: Sendable {
    /// id → piece, dense over the id space (gaps are empty strings).
    private let pieces: [String]
    private let specialIDs: Set<Int>
    private static let metaspace: Character = "\u{2581}"   // ▁

    public var vocabularySize: Int { pieces.count }

    public init(tokenizerJSON url: URL) throws {
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw GraphError.message("tokenizer.json: not a JSON object")
        }
        guard let model = root["model"] as? [String: Any],
              let vocab = model["vocab"] as? [String: Int] else {
            throw GraphError.message("tokenizer.json: missing model.vocab")
        }

        var maxID = vocab.values.max() ?? 0
        var special = Set<Int>()
        // Added tokens (<unk> = 0, <blank> = 1024) live outside model.vocab and are skipped
        // on decode, matching `skip_special_tokens=True`.
        let added = (root["added_tokens"] as? [[String: Any]]) ?? []
        for tok in added {
            guard let id = tok["id"] as? Int else { continue }
            maxID = max(maxID, id)
            if (tok["special"] as? Bool) ?? false { special.insert(id) }
        }

        var table = [String](repeating: "", count: maxID + 1)
        for (piece, id) in vocab { table[id] = piece }
        for tok in added {
            if let id = tok["id"] as? Int, let content = tok["content"] as? String, table[id].isEmpty {
                table[id] = content
            }
        }
        self.pieces = table
        self.specialIDs = special
    }

    /// The raw vocabulary piece for an id, or `nil` for out-of-range and special ids
    /// (`<unk>`, `<blank>`), which decode already skips.
    public func piece(_ id: Int) -> String? {
        guard id >= 0, id < pieces.count, !specialIDs.contains(id) else { return nil }
        return pieces[id]
    }

    /// Whether this token begins a new word: its piece starts with the SentencePiece
    /// word-boundary marker "▁". Special/out-of-range ids never start a word.
    public func startsWord(_ id: Int) -> Bool {
        piece(id)?.first == Self.metaspace
    }

    public func decode(_ ids: [Int]) -> String {
        var out = ""
        out.reserveCapacity(ids.count * 4)
        var isFirst = true
        for id in ids {
            guard id >= 0, id < pieces.count, !specialIDs.contains(id) else { continue }
            for ch in pieces[id] {
                if ch == Self.metaspace {
                    if isFirst { continue }      // leading ▁ of the sequence is dropped
                    out.append(" ")
                } else {
                    out.append(ch)
                }
            }
            isFirst = false
        }
        return out
    }
}
