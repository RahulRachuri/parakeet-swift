import Foundation

/// Flat little-endian float32 files: the lingua franca between the Python reference dumps
/// and this Swift host. `Data.withUnsafeBytes` + `loadUnaligned` is the idiomatic Swift way to
/// reinterpret bytes; there is no `reinterpret_cast`, and `Array(unsafeUninitializedCapacity:)`
/// lets us fill the buffer without zeroing it first.
public enum BinaryIO {
    public static func readFloat32(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        let count = data.count / MemoryLayout<Float>.size
        return data.withUnsafeBytes { raw in
            Array(unsafeUninitializedCapacity: count) { buf, initialized in
                for i in 0..<count {
                    buf[i] = raw.loadUnaligned(fromByteOffset: i * 4, as: Float.self)
                }
                initialized = count
            }
        }
    }

    public static func writeFloat32(_ values: [Float], to url: URL) throws {
        try values.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: url)
    }
}

// MARK: - Similarity metrics (the gate vocabulary)

public enum Metrics {
    /// Cosine similarity of two flat vectors.
    public static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        precondition(a.count == b.count, "cosine: length mismatch \(a.count) vs \(b.count)")
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0..<a.count {
            let x = Double(a[i]), y = Double(b[i])
            dot += x * y; na += x * x; nb += y * y
        }
        return dot / (na.squareRoot() * nb.squareRoot())
    }

    /// Per-row cosine similarity for a [rows, cols] row-major pair (the zoo's gate metric).
    public static func perRowCosine(_ a: [Float], _ b: [Float], cols: Int) -> (mean: Double, min: Double) {
        let rows = a.count / cols
        var total = 0.0
        var worst = Double.infinity
        for r in 0..<rows {
            let lo = r * cols, hi = lo + cols
            let c = cosine(Array(a[lo..<hi]), Array(b[lo..<hi]))
            total += c
            worst = Swift.min(worst, c)
        }
        return (total / Double(rows), worst)
    }

    public static func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count)
        var m: Float = 0
        for i in 0..<a.count { m = Swift.max(m, abs(a[i] - b[i])) }
        return m
    }
}
