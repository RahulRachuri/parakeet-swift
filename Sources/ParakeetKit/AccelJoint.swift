import Accelerate
import CParakeetBLAS
import Foundation

/// The joint head as plain Accelerate arithmetic: the dispatch-elimination spike.
///
/// The compiled joint graph costs ~123 µs/call on the CPU unit, of which the arithmetic
/// (`head(relu(enc + dec))`, one 1030×640 GEMV) is tens of microseconds at most: the loop
/// is dispatch-bound, not compute-bound: profiling put ~65 % of loop time in dispatch
/// rather than arithmetic. This class runs the identical math as a direct function call.
///
/// Weights come from `joint_head_{w,b}_f32.bin` next to the bundles, exported from the
/// same `model.safetensors` the graph was authored from (`export_joint_weights.py`;
/// shapes and hashes in `joint_head_manifest.json`).
///
/// Unlike the graph `InferenceFunction`s (one replica per worker: concurrent `run()` on
/// a shared function interleaves outputs silently), this is immutable weights + reentrant
/// BLAS: ONE instance is safely shared by every decode worker. Scratch buffers are the
/// caller's, so the shared object holds no mutable state at all.
public final class AccelJoint: Sendable {
    public let rows: Int      // vocab + durations = 1030
    public let hidden: Int    // 640
    let w: [Float]            // [rows, hidden] row-major
    let b: [Float]            // [rows]

    public init(artifactsDirectory dir: URL, config: ParakeetConfig) throws {
        hidden = config.hidden
        rows = config.vocabSize + config.durations.count
        let wData = try Data(contentsOf: dir.appendingPathComponent("joint_head_w_f32.bin"))
        let bData = try Data(contentsOf: dir.appendingPathComponent("joint_head_b_f32.bin"))
        guard wData.count == rows * hidden * 4, bData.count == rows * 4 else {
            throw NSError(domain: "AccelJoint", code: 1, userInfo: [NSLocalizedDescriptionKey:
                "joint blob size mismatch: w \(wData.count) B (want \(rows * hidden * 4)), "
                + "b \(bData.count) B (want \(rows * 4)); re-run export_joint_weights.py"])
        }
        w = wData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        b = bData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    /// `logits = W @ relu(enc + dec) + b`. `tmp` is [hidden], `logits` is [rows]; both are
    /// per-worker scratch so concurrent callers never share mutable memory.
    public func forward(dec: UnsafePointer<Float>, enc: UnsafePointer<Float>,
                        tmp: inout [Float], logits: inout [Float]) {
        let h = hidden, r = rows
        tmp.withUnsafeMutableBufferPointer { t in
            vDSP_vadd(dec, 1, enc, 1, t.baseAddress!, 1, vDSP_Length(h))
            var lo: Float = 0, hi = Float.greatestFiniteMagnitude
            vDSP_vclip(t.baseAddress!, 1, &lo, &hi, t.baseAddress!, 1, vDSP_Length(h))  // ReLU
            logits.withUnsafeMutableBufferPointer { out in
                // Preload bias, then beta=1 accumulates the GEMV on top.
                b.withUnsafeBufferPointer { bp in
                    out.baseAddress!.update(from: bp.baseAddress!, count: r)
                }
                w.withUnsafeBufferPointer { wp in
                    // out already holds b, and the wrapper accumulates: out += W·t.
                    pk_sgemv_row_major_accumulate(r, h, wp.baseAddress!, h,
                                                  t.baseAddress!, out.baseAddress!)
                }
            }
        }
    }
}

/// vDSP argmax over a buffer slice: the accel path's replacement for `argmaxNDArray`.
public func argmaxFloats(_ p: UnsafePointer<Float>, count: Int) -> Int {
    var maxV: Float = 0
    var maxI: vDSP_Length = 0
    vDSP_maxvi(p, 1, &maxV, &maxI, vDSP_Length(count))
    return Int(maxI)
}
