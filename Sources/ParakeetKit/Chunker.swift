import Foundation

/// The long-form cut policy, ported line-for-line from the Python reference prep step
/// (`rms_curve` + `cut_points`) so a wav chunked here gets **the same cuts** that step
/// would produce, which is what keeps a fresh `transcribe` run comparable with every
/// stored manifest and token stream.
///
/// Policy: from the current chunk start, look at the window [24 s, 28.8 s] ahead and cut at
/// the minimum of a 200 ms short-time RMS on a 10 ms grid: the quietest instant in the legal
/// band, which on narrated prose is a pause between sentences or clauses. 28.8 s keeps a
/// margin under the encoder's 28.85 s bucket, so every chunk fits after silence-padding.
///
/// Numeric faithfulness notes (each of these was a deliberate match, not an accident):
///  * the energy accumulates in `Double`, sequentially, exactly like `np.cumsum` on float64;
///  * the frame grid is `stride(from: 0, to: n - win, by: hop)`: the *exclusive* upper bound
///    matches `np.arange(0, len(wav) - RMS_WIN, hop)`;
///  * band edges truncate via `Int(seconds * rate)` (the same double product Python's
///    `int()` truncates);
///  * argmin takes the **first** minimum, like `np.argmin`.
public enum LongFormChunker {
    public static let minSeconds = 24.0
    public static let maxSeconds = 28.8
    public static let rmsWindowSeconds = 0.20

    /// `[start, end)` sample ranges covering all of `samples`, each ≤ 28.8 s.
    public static func cutPoints(samples: [Float], sampleRate: Int) -> [(start: Int, end: Int)] {
        cutPoints(sampleCount: samples.count, sampleRate: sampleRate) { lo, hi in
            Array(samples[lo..<hi])
        }
    }

    /// Streaming variant: same cuts, flat memory. `read` returns samples `[lo, hi)`: the wav
    /// reader's mmap-backed `samples(from:to:)` slots straight in. The whole-array version
    /// above materialised a `Double` prefix-sum over every sample (507 MB for a 66-minute
    /// chapter, and the freed pages stayed dirty in the malloc zone for the process lifetime);
    /// this one keeps a single running sum and stores it only at grid points (~3 MB).
    ///
    /// Numeric faithfulness is preserved exactly, not approximately: the running sum performs
    /// the SAME sequential `+ s*s` additions in the SAME order as the old `cumulative` array,
    /// and because `win` is an exact multiple of `hop` (3200 = 20 × 160 @ 16 kHz), every
    /// `cumulative[i + win] - cumulative[i]` the policy ever reads has BOTH ends on the 10 ms
    /// grid, so sampling the sum at grid points loses nothing. Same doubles, same argmin,
    /// same cuts.
    public static func cutPoints(sampleCount n: Int, sampleRate: Int,
                                 read: (Int, Int) -> [Float]) -> [(start: Int, end: Int)] {
        guard n > 0 else { return [] }
        let hop = sampleRate / 100                       // the 10 ms grid (160 @ 16 kHz)
        let win = Int(rmsWindowSeconds * Double(sampleRate))
        precondition(win % hop == 0, "RMS window must sit on the hop grid for streaming cumsum")
        let maxSamples = Int(maxSeconds * Double(sampleRate))
        let minSamples = Int(minSeconds * Double(sampleRate))

        // Short-time RMS over a boxcar of `win`, one value per `hop`.
        var energy: [Double] = []
        if n - win > 0 {
            // cumulative[g * hop] for every grid point g: built with one sequential running
            // sum over block reads, byte-identical to the full prefix-sum array's values.
            var gridSum: [Double] = []
            gridSum.reserveCapacity(n / hop + 1)
            var running = 0.0
            let block = 1 << 20
            var i = 0
            while i < n {
                let chunk = read(i, min(i + block, n))
                for s in chunk {
                    if i % hop == 0 { gridSum.append(running) }
                    running += Double(s) * Double(s)
                    i += 1
                }
            }
            let winHops = win / hop
            energy.reserveCapacity((n - win) / hop + 1)
            for i in stride(from: 0, to: n - win, by: hop) {
                let f = i / hop
                energy.append(((gridSum[f + winHops] - gridSum[f]) / Double(win)).squareRoot())
            }
        }

        var chunks: [(start: Int, end: Int)] = []
        var pos = 0
        while pos < n {
            if n - pos <= maxSamples {
                chunks.append((pos, n))
                break
            }
            let lo = pos + minSamples, hi = pos + maxSamples
            let fLo = lo / hop, fHi = min(hi / hop, energy.count - 1)
            let cut: Int
            if fHi <= fLo {
                cut = hi
            } else {
                var best = fLo
                for f in (fLo + 1)..<fHi where energy[f] < energy[best] { best = f }
                cut = best * hop
            }
            chunks.append((pos, cut))
            pos = cut
        }
        return chunks
    }
}
