import Foundation

/// A chunk manifest, as produced by the Python prep step `Chunker` reproduces.
public struct ChunkManifest: Sendable {
    public struct Chunk: Sendable {
        public let index: Int
        public let start: Int      // sample offsets into the wav
        public let end: Int
        public let seconds: Double
        /// Optional caller-supplied label for the chunk. The long-form chunker does not set it;
        /// the LibriSpeech manifest sets it to the utterance id (e.g. `1089-134686-0000`) so that
        /// `transcribe-list` can key hypotheses back to references.
        ///
        /// Several chunks MAY share one id: an utterance longer than the encoder bucket has to be
        /// cut into consecutive bucket-sized pieces, and `transcribe-list` rejoins their text in
        /// chunk order. `nil` here is not an error: it just means "unlabelled".
        public let id: String?

        /// Public memberwise init, so a caller that cut its own chunks (e.g. the CLI's
        /// `transcribe`, which runs `LongFormChunker` in-process) can drive the same
        /// manifest-shaped pipeline the stored manifests do.
        public init(index: Int, start: Int, end: Int, seconds: Double, id: String? = nil) {
            self.index = index; self.start = start; self.end = end
            self.seconds = seconds; self.id = id
        }
    }

    public let sampleRate: Int
    public let wavPath: String
    public let chunks: [Chunk]

    public init(url: URL) throws {
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let sr = root["sr"] as? Int,
              let path = root["path"] as? String,
              let raw = root["chunks"] as? [[String: Any]] else {
            throw GraphError.message("chunks.json: unexpected shape")
        }
        self.sampleRate = sr
        // Some manifests store the wav path relative to the harness root; resolve those
        // against the manifest's own directory rather than the process cwd.
        self.wavPath = path.hasPrefix("/")
            ? path
            : url.deletingLastPathComponent().appendingPathComponent(
                (path as NSString).lastPathComponent).path
        self.chunks = raw.compactMap {
            guard let i = $0["i"] as? Int, let s = $0["start"] as? Int, let e = $0["end"] as? Int
            else { return nil }
            return Chunk(index: i, start: s, end: e,
                         seconds: ($0["seconds"] as? Double) ?? Double(e - s) / Double(sr),
                         id: $0["id"] as? String)
        }
    }

    public var totalSeconds: Double { chunks.reduce(0) { $0 + $1.seconds } }
}

/// Minimal 16-bit-PCM mono WAV reader.
///
/// Deliberately not AVFoundation: the bench wavs are plain PCM_16 mono, and mapping the file
/// and converting on demand keeps memory flat over a 66-minute chapter (the file is 127 MB).
public final class PCM16WavFile {
    private let data: Data
    private let dataOffset: Int
    public let sampleRate: Int
    public let channels: Int
    public let frameCount: Int

    public init(url: URL) throws {
        // Swift requires every stored property to be initialised before `self` can be used, so
        // the header walk works on a local `bytes` and the properties are assigned at the end.
        let bytes = try Data(contentsOf: url, options: .alwaysMapped)
        guard bytes.count > 44,
              bytes[0..<4].elementsEqual("RIFF".utf8),
              bytes[8..<12].elementsEqual("WAVE".utf8) else {
            throw GraphError.message("\(url.lastPathComponent): not a RIFF/WAVE file")
        }
        func u16(_ i: Int) -> Int {
            Int(bytes.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: i, as: UInt16.self) })
        }
        func u32(_ i: Int) -> Int {
            Int(bytes.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: i, as: UInt32.self) })
        }
        // Walk the chunk list rather than assuming the canonical 44-byte header.
        var offset = 12
        var sr = 0, ch = 0, bits = 0
        var dataStart = -1, dataBytes = 0
        while offset + 8 <= bytes.count {
            let id = String(decoding: bytes[offset..<offset + 4], as: UTF8.self)
            let size = u32(offset + 4)
            let body = offset + 8
            if id == "fmt " {
                ch = u16(body + 2)
                sr = u32(body + 4)
                bits = u16(body + 14)
            } else if id == "data" {
                dataStart = body
                dataBytes = size
                break
            }
            offset = body + size + (size % 2)
        }
        guard dataStart >= 0, bits == 16, ch == 1 else {
            throw GraphError.message("expected mono 16-bit PCM, got \(ch)ch/\(bits)bit")
        }
        self.data = bytes
        self.dataOffset = dataStart
        self.sampleRate = sr
        self.channels = ch
        self.frameCount = min(dataBytes, bytes.count - dataStart) / 2
    }

    /// Samples `[start, end)` as normalised Float in [-1, 1) (the same scaling soundfile uses).
    public func samples(from start: Int, to end: Int) -> [Float] {
        let lo = max(0, start), hi = min(frameCount, end)
        guard hi > lo else { return [] }
        let count = hi - lo
        return data.withUnsafeBytes { raw in
            Array(unsafeUninitializedCapacity: count) { buf, initialized in
                for i in 0..<count {
                    let s = raw.loadUnaligned(fromByteOffset: dataOffset + (lo + i) * 2, as: Int16.self)
                    buf[i] = Float(s) / 32768.0
                }
                initialized = count
            }
        }
    }
}
