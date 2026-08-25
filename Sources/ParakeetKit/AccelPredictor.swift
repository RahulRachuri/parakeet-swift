import Accelerate
import CParakeetBLAS
import Foundation

/// The predictor (embedding → 2× LSTM cell → projector) as plain Accelerate arithmetic:
/// the second half of the dispatch-elimination work `AccelJoint` started.
///
/// Semantics mirror `export_decoder.py`'s `Predict` exactly: PyTorch gate order `i,f,g,o`
/// in 640-row blocks, `c' = σ(f)·c + σ(i)·tanh(g)`, `h' = σ(o)·tanh(c')`, projector on the
/// second layer's `h`. The two per-gate biases are summed once at load (they are only ever
/// added together). Sigmoid is computed as `1/(1+exp(−x))` via vForce, matching PyTorch's
/// definition rather than the tanh identity.
///
/// Immutable weights + caller-owned state and scratch ⇒ one shared instance across all
/// decode workers, same as `AccelJoint`.
public final class AccelPredictor: Sendable {
    public let hidden: Int   // 640
    public let layers: Int   // 2
    let embed: [Float]       // [vocab, hidden]
    let wIH: [[Float]]       // per layer [4H, H]
    let wHH: [[Float]]       // per layer [4H, H]
    let bSum: [[Float]]      // per layer [4H] = b_ih + b_hh
    let projW: [Float]       // [H, H]
    let projB: [Float]       // [H]

    public init(artifactsDirectory dir: URL, config: ParakeetConfig) throws {
        hidden = config.hidden
        layers = config.layers
        func blob(_ name: String, count: Int) throws -> [Float] {
            let d = try Data(contentsOf: dir.appendingPathComponent(name))
            guard d.count == count * 4 else {
                throw NSError(domain: "AccelPredictor", code: 1, userInfo: [NSLocalizedDescriptionKey:
                    "\(name): \(d.count) B, want \(count * 4); re-run export_predictor_weights.py"])
            }
            return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        let H = hidden
        embed = try blob("pred_embed_f32.bin", count: config.vocabSize * H)
        var ih: [[Float]] = [], hh: [[Float]] = [], bs: [[Float]] = []
        for l in 0..<layers {
            ih.append(try blob("pred_w_ih\(l)_f32.bin", count: 4 * H * H))
            hh.append(try blob("pred_w_hh\(l)_f32.bin", count: 4 * H * H))
            let bi = try blob("pred_b_ih\(l)_f32.bin", count: 4 * H)
            let bh = try blob("pred_b_hh\(l)_f32.bin", count: 4 * H)
            bs.append(vDSP.add(bi, bh))
        }
        wIH = ih; wHH = hh; bSum = bs
        projW = try blob("pred_proj_w_f32.bin", count: H * H)
        projB = try blob("pred_proj_b_f32.bin", count: H)
    }

    /// Scratch a worker owns for the duration of a chunk: LSTM state plus gate buffers.
    /// `h`/`c` are `[layers · hidden]`, zeroed at chunk start (chunks are state-independent).
    public struct Scratch {
        public var h: [Float]
        public var c: [Float]
        var gates: [Float]     // [4H]
        var tanhC: [Float]     // [H]
        public var decOut: [Float]  // [H] (feeds AccelJoint directly)
        public init(hidden H: Int, layers L: Int) {
            h = [Float](repeating: 0, count: L * H)
            c = [Float](repeating: 0, count: L * H)
            gates = [Float](repeating: 0, count: 4 * H)
            tanhC = [Float](repeating: 0, count: H)
            decOut = [Float](repeating: 0, count: H)
        }
        public mutating func reset() {
            vDSP.fill(&h, with: 0)
            vDSP.fill(&c, with: 0)
        }
    }

    /// One predictor step: `token` in, `s.decOut` (and the advanced `s.h`/`s.c`) out.
    public func forward(token: Int, _ s: inout Scratch) {
        let H = hidden
        embed.withUnsafeBufferPointer { eb in
            var x = eb.baseAddress! + token * H          // layer 0 input: the embedding row
            s.gates.withUnsafeMutableBufferPointer { g in
                s.h.withUnsafeMutableBufferPointer { hb in
                    s.c.withUnsafeMutableBufferPointer { cb in
                        s.tanhC.withUnsafeMutableBufferPointer { tc in
                            for l in 0..<layers {
                                let hl = hb.baseAddress! + l * H
                                let cl = cb.baseAddress! + l * H
                                // gates = bSum + W_ih·x + W_hh·h  (two GEMVs accumulating)
                                bSum[l].withUnsafeBufferPointer {
                                    g.baseAddress!.update(from: $0.baseAddress!, count: 4 * H)
                                }
                                // g already holds b_ih + b_hh, and the wrapper accumulates.
                                wIH[l].withUnsafeBufferPointer {
                                    pk_sgemv_row_major_accumulate(4 * H, H, $0.baseAddress!, H,
                                                                  x, g.baseAddress!)
                                }
                                wHH[l].withUnsafeBufferPointer {
                                    pk_sgemv_row_major_accumulate(4 * H, H, $0.baseAddress!, H,
                                                                  hl, g.baseAddress!)
                                }
                                let i = g.baseAddress!, f = i + H, gg = f + H, o = gg + H
                                sigmoidInPlace(i, H); sigmoidInPlace(f, H); sigmoidInPlace(o, H)
                                var n = Int32(H)
                                vvtanhf(gg, gg, &n)                     // tanh(g)
                                vDSP_vmul(f, 1, cl, 1, cl, 1, vDSP_Length(H))       // c = σ(f)·c
                                vDSP_vma(i, 1, gg, 1, cl, 1, cl, 1, vDSP_Length(H)) // c += σ(i)·tanh(g)
                                vvtanhf(tc.baseAddress!, cl, &n)
                                vDSP_vmul(o, 1, tc.baseAddress!, 1, hl, 1, vDSP_Length(H)) // h = σ(o)·tanh(c)
                                x = UnsafePointer(hl)                   // next layer's input
                            }
                            // dec_out = projB + projW · h[last]
                            s.decOut.withUnsafeMutableBufferPointer { d in
                                projB.withUnsafeBufferPointer {
                                    d.baseAddress!.update(from: $0.baseAddress!, count: H)
                                }
                                projW.withUnsafeBufferPointer {
                                    pk_sgemv_row_major_accumulate(H, H, $0.baseAddress!, H,
                                                                  x, d.baseAddress!)
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// `x ← 1/(1+exp(−x))`, vectorized in place.
    private func sigmoidInPlace(_ p: UnsafeMutablePointer<Float>, _ count: Int) {
        var n = Int32(count)
        vDSP_vneg(p, 1, p, 1, vDSP_Length(count))
        vvexpf(p, p, &n)
        var one: Float = 1
        vDSP_vsadd(p, 1, &one, p, 1, vDSP_Length(count))
        vvrecf(p, p, &n)
    }
}
