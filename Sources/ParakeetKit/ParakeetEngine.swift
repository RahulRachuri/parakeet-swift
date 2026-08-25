import CoreAI
import Foundation

/// Model constants for Parakeet-TDT-0.6B-**v2**. (v3 differs only in `blank`/`vocabSize`.)
public struct ParakeetConfig: Sendable {
    public var blank = 1024
    public var durations = [0, 1, 2, 3, 4]
    public var hidden = 640          // predictor/joint width
    public var layers = 2            // predictor LSTM layers
    public var vocabSize = 1025
    public init() {}
}

/// Where the three bundles + the tokenizer/filterbank assets live.
public struct ArtifactPaths: Sendable {
    public var encoder: URL
    public var predictor: URL
    public var joint: URL
    public var tokenizer: URL
    public var filterbank: URL
    /// Mel frames of the encoder bundle named above: it is part of the filename, so the two
    /// can never drift apart. Threaded on to `MelFrontend` so the front end pads to the same
    /// length the graph expects.
    public var bucketFrames: Int

    /// Standard layout: the zoo's `conversion/parakeet/artifacts_v2/`.
    ///
    /// `graphExtension` selects which *build* of the three graphs to load. The default
    /// `"aimodel"` is the portable source bundle, which the runtime specializes at load time
    /// (JIT). Passing an AOT extension such as `"h16s.aimodelc"` loads the ahead-of-time
    /// compiled slice produced by `coreai-build compile --platform macOS --architecture h16s`
    /// instead: same graphs, same weights, no load-time specialization.
    ///
    /// `assetsDirectory` stays pointed at the source tree, because the tokenizer and the mel
    /// filterbank are plain data files that `coreai-build` neither reads nor emits.
    public init(artifactsDirectory dir: URL,
                graphExtension: String = "aimodel",
                assetsDirectory: URL? = nil,
                bucketFrames: Int = MelFrontend.defaultBucketFrames) {
        self.bucketFrames = bucketFrames
        encoder = dir.appendingPathComponent(
            "parakeet_encoder_float16_L\(bucketFrames).\(graphExtension)")
        predictor = dir.appendingPathComponent("parakeet_predict_float32.\(graphExtension)")
        joint = dir.appendingPathComponent("parakeet_joint_float32.\(graphExtension)")
        let assets = (assetsDirectory ?? dir).appendingPathComponent("bundle_assets")
        tokenizer = assets.appendingPathComponent("tokenizer.json")
        filterbank = assets.appendingPathComponent("mel_filters_128x257_f32.bin")
    }
}

/// Which compute unit each graph is specialized for. The encoder is always GPU (fixed by the
/// port); the predictor and joint are the interesting knobs, because they are tiny graphs
/// called hundreds of times per chunk, where dispatch overhead can dominate arithmetic.
public struct ComputePlan: Sendable {
    public var encoder: ComputeUnit = .gpu
    public var predictor: ComputeUnit = .gpu
    public var joint: ComputeUnit = .gpu
    public init(encoder: ComputeUnit = .gpu, predictor: ComputeUnit = .gpu, joint: ComputeUnit = .gpu) {
        self.encoder = encoder; self.predictor = predictor; self.joint = joint
    }
    public var label: String { "enc:\(encoder.rawValue)/pred:\(predictor.rawValue)/joint:\(joint.rawValue)" }
}

/// One chunk's worth of results plus the timing split.
public struct ChunkResult: Sendable {
    public var tokens: [Int] = []
    /// The encoder-frame index each token in `tokens` was emitted at (parallel array).
    /// Additive metadata only: recording it never influences the decode. One encoder frame
    /// is 80 ms of audio (8× subsampling of the 10 ms mel hop), so a token's time within
    /// its chunk is `frameIndices[i] * 0.08` seconds.
    public var frameIndices: [Int] = []
    public var melSeconds = 0.0
    public var encoderSeconds = 0.0
    public var loopSeconds = 0.0
    public var jointCalls = 0
    public var predictorCalls = 0
    public var frames = 0
    /// Extra joint calls issued by the speculative path (0 on the plain loop).
    public var speculativeJointCalls = 0
    /// How many prefetched frames were actually consumed without a predictor step.
    public var usefulSpeculations = 0
    public init() {}
    public var engineCalls: Int { jointCalls + predictorCalls + 1 }
}

/// One **replica** of the host-side TDT greedy decode, owning its own predictor + joint
/// `InferenceFunction`s.
///
/// Why a whole object per decode worker rather than sharing the engine's graphs: concurrent
/// `run()` calls on a *single* `InferenceFunction` silently interleave their outputs: no error,
/// no crash, just a garbage token stream (measured in Round 4: 20× the tokens, 57/152 parity).
/// Giving each worker its own pair removes the question entirely instead of relying on a
/// submission discipline we would have to prove. The graphs are tiny: predictor and joint each
/// load in ~0.00 s and specialize from the same JIT cache, so N replicas are nearly free.
///
/// The loop body below is the *only* copy in the package: `ParakeetEngine.decode` delegates here,
/// so the serial path and the parallel path run byte-identical code and the parity gate is
/// testing the same arithmetic in both.
public final class TDTDecoder: @unchecked Sendable {
    public let config: ParakeetConfig
    let predictor: Graph
    let joint: Graph
    /// Non-nil ⇒ the joint runs as plain Accelerate arithmetic instead of the compiled
    /// graph (`--decode-impl accel-joint`). One shared instance across workers is safe;
    /// see `AccelJoint`.
    public var accelJoint: AccelJoint?
    /// Non-nil (with `accelJoint` also set) ⇒ the whole per-token loop is plain CPU
    /// arithmetic (`--decode-impl accel`): no graph calls, no awaits, see `decodeAccel`.
    public var accelPredictor: AccelPredictor?

    init(predictor: Graph, joint: Graph, config: ParakeetConfig) {
        self.predictor = predictor
        self.joint = joint
        self.config = config
    }

    /// The TDT greedy loop: the whole reason the host language matters.
    ///
    /// Semantics are exactly `conversion/parakeet/gate_e2e.py`:
    /// start the predictor from the blank token and zero LSTM state; at each encoder frame ask
    /// the joint for a token and a *duration*; a blank with duration 0 is forced to 1 so the
    /// loop always advances; a non-blank token is emitted and fed back through the predictor.
    ///
    /// Every chunk starts from zero state, so chunks are independent, which is precisely what
    /// makes cross-chunk parallelism legal rather than an approximation.
    ///
    /// **Zero-allocation marshalling** (backported from the iPhone fork's Round 10, where it
    /// measured +9.1%). The pre-Round-10 body rebuilt the world on every graph call: five
    /// `NDArray(scalars:shape:)` constructions per emitted token (3 predictor in, 2 joint in),
    /// plus five runtime-allocated output buffers each flattened into a fresh `[Float]`,
    /// including a 1025-wide logit vector that existed only to be argmaxed. None of that is
    /// arithmetic. This body allocates every buffer once per *chunk* and mutates in place,
    /// handing the runtime `outputViews:` so results land in buffers the host already owns.
    /// Graphs, call order and the greedy rule are untouched: a marshalling change,
    /// token-identical by construction, re-proved by the parity gate (same 151/152 fingerprint
    /// before/after the backport).
    ///
    /// Why everything is a local rather than a stored property: `NDArray.MutableView` and
    /// `InferenceFunction.MutableViews` are `~Escapable`: a view derived from a class property
    /// dies at the property access and cannot cross an `await`. Derived from a local `var`, the
    /// lifetime encloses the loop, and 9 allocations per chunk (instead of ~1 800) is close
    /// enough to zero.
    ///
    /// One deliberate divergence from the pre-Round-10 body: the predictor step is hoisted into
    /// the top of the loop (`needPredict`), which skips the trailing predictor call whose
    /// `dec_out` was never consumed: issued whenever the last frame emitted a token. That is
    /// ≤1 call per chunk of pure dead work; it cannot move a token, only `predictorCalls`.
    public func decode(enc: [Float], frames T: Int) async throws -> ChunkResult {
        if let ap = accelPredictor, let aj = accelJoint {
            return decodeAccel(enc: enc, frames: T, predictor: ap, joint: aj)
        }
        var result = ChunkResult()
        result.frames = T
        let L = config.layers, H = config.hidden
        let V = config.vocabSize, D = config.durations.count
        let blank = config.blank

        // ---- the buffers, allocated once for this chunk -------------------------------
        var tokenIn = NDArray(shape: [1, 1], scalarType: .int32)
        var hIn = NDArray(shape: [L, 1, H], scalarType: .float32)
        var cIn = NDArray(shape: [L, 1, H], scalarType: .float32)
        // h/c outputs cannot alias their own inputs, so they land here and are rolled back.
        var hOut = NDArray(shape: [L, 1, H], scalarType: .float32)
        var cOut = NDArray(shape: [L, 1, H], scalarType: .float32)
        // `dec_out` is a predictor output AND the joint's first input: one buffer, written by
        // one dispatch and read by the next, never passing through the host at all.
        var decOut = NDArray(shape: [1, H], scalarType: .float32)
        var encIn = NDArray(shape: [1, H], scalarType: .float32)
        var tokLogits = NDArray(shape: [1, V], scalarType: .float32)
        var durLogits = NDArray(shape: [1, D], scalarType: .float32)
        zeroFloats(&hIn, count: L * H)
        zeroFloats(&cIn, count: L * H)

        var frame = 0
        var emitted: [Int] = []
        emitted.reserveCapacity(T)
        let emissionCap = 12 * T
        var pendingToken = Int32(blank)
        var needPredict = true

        // Scratch for the accel joint path: per-chunk like every other buffer here, so
        // concurrent workers sharing one `AccelJoint` never share mutable memory.
        let aj = accelJoint
        var ajTmp = [Float](repeating: 0, count: aj != nil ? H : 0)
        var ajLogits = [Float](repeating: 0, count: aj?.rows ?? 0)

        let t0 = Date()
        while frame < T && emitted.count < emissionCap {
            if needPredict {
                var tokenView = tokenIn.mutableView(as: Int32.self)
                tokenView.withUnsafeMutablePointer { p, _, _ in p[0] = pendingToken }
                var views = InferenceFunction.MutableViews()
                views.insert(&decOut, for: "dec_out")
                views.insert(&hOut, for: "h_out")
                views.insert(&cOut, for: "c_out")
                _ = try await predictor.function.run(
                    inputs: ["token": tokenIn, "h": hIn, "c": cIn], outputViews: views)
                // Roll the recurrent state forward: 20 KB of memcpy per emitted token, and it
                // buys a fixed input set for every call.
                copyFloats(from: &hOut, to: &hIn, count: L * H)
                copyFloats(from: &cOut, to: &cIn, count: L * H)
                result.predictorCalls += 1
                needPredict = false
            }

            let token: Int
            let durIndex: Int
            if let aj {
                // Accel path: dec_out straight out of the predictor's output buffer, the
                // enc frame straight out of the Swift array, no NDArray marshalling, no
                // graph dispatch, same arithmetic.
                var decView = decOut.mutableView(as: Float.self)
                (token, durIndex) = decView.withUnsafeMutablePointer { dec, _, _ in
                    enc.withUnsafeBufferPointer { src in
                        aj.forward(dec: dec, enc: src.baseAddress! + frame * H,
                                   tmp: &ajTmp, logits: &ajLogits)
                        return ajLogits.withUnsafeBufferPointer { lp in
                            (argmaxFloats(lp.baseAddress!, count: V),
                             argmaxFloats(lp.baseAddress! + V, count: D))
                        }
                    }
                }
                result.jointCalls += 1
            } else {
                // enc_proj[frame:frame+1] straight into the runtime's buffer (no Swift array hop).
                enc.withUnsafeBufferPointer { src in
                    var encView = encIn.mutableView(as: Float.self)
                    encView.withUnsafeMutablePointer { dst, _, _ in
                        dst.update(from: src.baseAddress! + frame * H, count: H)
                    }
                }
                var views = InferenceFunction.MutableViews()
                views.insert(&tokLogits, for: "token_logits")
                views.insert(&durLogits, for: "dur_logits")
                _ = try await joint.function.run(
                    inputs: ["dec_out": decOut, "enc_frame": encIn], outputViews: views)
                result.jointCalls += 1

                token = argmaxNDArray(&tokLogits, count: V)
                durIndex = argmaxNDArray(&durLogits, count: D)
            }
            var duration = config.durations[durIndex]
            if token == blank && duration == 0 { duration = 1 }
            // The frame the joint was just evaluated at, recorded (not `frame` after the
            // advance) so a token's timestamp is the frame that produced it.
            let emissionFrame = frame
            frame += duration
            if token != blank {
                emitted.append(token)
                result.frameIndices.append(emissionFrame)
                pendingToken = Int32(token)
                needPredict = true
            }
        }
        result.loopSeconds = -t0.timeIntervalSinceNow
        result.tokens = emitted
        return result
    }

    /// The same TDT greedy loop with BOTH graphs replaced by Accelerate arithmetic:
    /// no dispatch, no NDArrays, no awaits. Semantics are line-for-line the loop above
    /// (emission cap, predictor hoisting, the blank/duration-0 rule, timestamp frames);
    /// the parity gate runs both paths against the same gold streams to keep them honest.
    func decodeAccel(enc: [Float], frames T: Int,
                     predictor ap: AccelPredictor, joint aj: AccelJoint) -> ChunkResult {
        var result = ChunkResult()
        result.frames = T
        let H = config.hidden, V = config.vocabSize, D = config.durations.count
        let blank = config.blank

        var s = AccelPredictor.Scratch(hidden: H, layers: config.layers)
        var jTmp = [Float](repeating: 0, count: H)
        var jLogits = [Float](repeating: 0, count: aj.rows)

        var frame = 0
        var emitted: [Int] = []
        emitted.reserveCapacity(T)
        let emissionCap = 12 * T
        var pendingToken = blank
        var needPredict = true

        let t0 = Date()
        while frame < T && emitted.count < emissionCap {
            if needPredict {
                ap.forward(token: pendingToken, &s)
                result.predictorCalls += 1
                needPredict = false
            }
            let (token, durIndex) = s.decOut.withUnsafeBufferPointer { dec in
                enc.withUnsafeBufferPointer { src -> (Int, Int) in
                    aj.forward(dec: dec.baseAddress!, enc: src.baseAddress! + frame * H,
                               tmp: &jTmp, logits: &jLogits)
                    return jLogits.withUnsafeBufferPointer { lp in
                        (argmaxFloats(lp.baseAddress!, count: V),
                         argmaxFloats(lp.baseAddress! + V, count: D))
                    }
                }
            }
            result.jointCalls += 1
            var duration = config.durations[durIndex]
            if token == blank && duration == 0 { duration = 1 }
            let emissionFrame = frame
            frame += duration
            if token != blank {
                emitted.append(token)
                result.frameIndices.append(emissionFrame)
                pendingToken = token
                needPredict = true
            }
        }
        result.loopSeconds = -t0.timeIntervalSinceNow
        result.tokens = emitted
        return result
    }
}

/// The full mel → encoder → TDT-greedy pipeline, in Swift.
public final class ParakeetEngine: @unchecked Sendable {
    public let config: ParakeetConfig
    public let plan: ComputePlan
    /// Kept so `makeDecoders` can load extra predictor/joint replicas from the same bundles.
    public let paths: ArtifactPaths
    public let tokenizer: ParakeetTokenizer
    public let frontend: MelFrontend
    public let loadSeconds: Double
    /// Per-graph load cost. Split out because AOT bundles change *only* this number:
    /// the encoder's load is where JIT specialization is paid.
    public let encoderLoadSeconds: Double
    public let predictorLoadSeconds: Double
    public let jointLoadSeconds: Double

    let encoder: Graph
    let predictor: Graph
    let joint: Graph
    /// Optional re-exported joint taking `enc_frames [K,640]`: K frames in ONE dispatch.
    /// See `decodeBatchedJoint`; nil unless the caller passed a batched bundle.
    let batchedJoint: Graph?
    public let batchK: Int
    /// The decode replica that the serial path uses. It wraps the engine's own predictor/joint,
    /// so `decode` below is exactly the pre-Round-6 code path, just relocated.
    public private(set) var primaryDecoder: TDTDecoder!

    public init(paths: ArtifactPaths,
                plan: ComputePlan = ComputePlan(),
                config: ParakeetConfig = ParakeetConfig(),
                batchedJointURL: URL? = nil,
                batchK: Int = 0) async throws {
        self.config = config
        self.batchK = batchK
        self.plan = plan
        self.paths = paths
        self.frontend = try MelFrontend(filterbankURL: paths.filterbank,
                                        bucketFrames: paths.bucketFrames)
        self.tokenizer = try ParakeetTokenizer(tokenizerJSON: paths.tokenizer)
        let t0 = Date()
        self.encoder = try await Graph(url: paths.encoder, unit: plan.encoder)
        let t1 = Date(); self.encoderLoadSeconds = t1.timeIntervalSince(t0)
        self.predictor = try await Graph(url: paths.predictor, unit: plan.predictor)
        let t2 = Date(); self.predictorLoadSeconds = t2.timeIntervalSince(t1)
        self.joint = try await Graph(url: paths.joint, unit: plan.joint)
        self.jointLoadSeconds = -t2.timeIntervalSinceNow
        // The batched joint is loaded on the same unit as the single-frame one, so a sweep
        // isolates the dispatch shape rather than confounding it with a compute-unit change.
        self.batchedJoint = batchedJointURL == nil
            ? nil
            : try await Graph(url: batchedJointURL!, unit: plan.joint)
        self.loadSeconds = -t0.timeIntervalSinceNow
        self.primaryDecoder = TDTDecoder(predictor: self.predictor, joint: self.joint,
                                         config: config)
    }

    /// Load `count` **additional** decode replicas, each with its own predictor + joint
    /// `InferenceFunction`. Replica 0 of the returned array is the primary, so a caller asking
    /// for N workers gets N decoders while only paying for N−1 extra loads.
    ///
    /// These specialize from the same JIT cache the primary did, so the cost is a few
    /// milliseconds each, not the ~35 s an encoder configuration would pay.
    public func makeDecoders(count: Int) async throws -> (decoders: [TDTDecoder], loadSeconds: Double) {
        guard count > 1 else { return ([primaryDecoder], 0) }
        let t0 = Date()
        var out: [TDTDecoder] = [primaryDecoder]
        for _ in 1..<count {
            let p = try await Graph(url: paths.predictor, unit: plan.predictor)
            let j = try await Graph(url: paths.joint, unit: plan.joint)
            out.append(TDTDecoder(predictor: p, joint: j, config: config))
        }
        return (out, -t0.timeIntervalSinceNow)
    }

    /// mel `[128×bucketFrames]` → encoder projection `[T×640]` (T = 361 for the 2885 bucket).
    public func encode(_ mel: [Float]) async throws -> (enc: [Float], frames: Int) {
        let input = ndarrayFloat16(mel, shape: [1, MelFrontend.melBins, frontend.bucketFrames])
        let out = try await encoder.run(["mel": input], wanting: ["enc_proj"])[0]
        return (out, out.count / config.hidden)
    }

    /// The TDT greedy loop. The body now lives on `TDTDecoder` so that the serial path and the
    /// chunk-parallel workers share one implementation; this is the same code as before Round 6,
    /// running on the engine's own predictor/joint pair.
    public func decode(enc: [Float], frames T: Int) async throws -> ChunkResult {
        try await primaryDecoder.decode(enc: enc, frames: T)
    }

    /// **Speculative joint batching.** Same greedy semantics, different dispatch shape.
    ///
    /// The idea: the joint is a pure function of (dec_out, enc_frame), and `dec_out` only
    /// changes when a token is *emitted*. So while the loop is taking blanks, the joint results
    /// for the next few frames could all have been computed against the same predictor state.
    /// We therefore prefetch `lookahead` frames concurrently, consume from the window, and
    /// throw the window away the moment a token is emitted (the state just changed).
    ///
    /// Because the durations are [0…4], a window of 5 covers every frame the loop could jump
    /// to without emitting, so the greedy result is unchanged **by construction**: this is a
    /// scheduling change, not an approximation. The `parity` command verifies that anyway.
    ///
    /// Whether it *pays* is an empirical question about how often the loop actually takes a
    /// blank; see the report in the package README.
    public func decodeSpeculative(enc: [Float], frames T: Int, lookahead K: Int) async throws -> ChunkResult {
        var result = ChunkResult()
        result.frames = T
        let H = config.hidden
        let stateCount = config.layers * H

        var h = [Float](repeating: 0, count: stateCount)
        var c = [Float](repeating: 0, count: stateCount)
        var dec: [Float]

        let t0 = Date()
        (dec, h, c) = try await predictStep(token: Int32(config.blank), h: h, c: c)
        result.predictorCalls += 1

        var frame = 0
        var emitted: [Int] = []
        let emissionCap = 12 * T
        // window[i] holds the joint outputs for frame windowBase + i.
        var windowBase = -1
        var window: [(token: Int, duration: Int)] = []

        while frame < T && emitted.count < emissionCap {
            if frame < windowBase || frame >= windowBase + window.count {
                let lo = frame
                let hi = min(T, frame + K)
                window = try await jointBatch(dec: dec, enc: enc, from: lo, to: hi)
                windowBase = lo
                result.jointCalls += hi - lo
                result.speculativeJointCalls += (hi - lo) - 1
            }
            let step = window[frame - windowBase]
            var duration = step.duration
            if step.token == config.blank && duration == 0 { duration = 1 }
            let emissionFrame = frame
            frame += duration
            if step.token != config.blank {
                emitted.append(step.token)
                result.frameIndices.append(emissionFrame)
                (dec, h, c) = try await predictStep(token: Int32(step.token), h: h, c: c)
                result.predictorCalls += 1
                window = []                  // predictor state changed: window is stale
                windowBase = -1
            } else {
                result.usefulSpeculations += 1
            }
        }
        result.loopSeconds = -t0.timeIntervalSinceNow
        result.tokens = emitted
        return result
    }

    /// **Batched-joint decode.** Same greedy semantics again; the difference from
    /// `decodeSpeculative` is that the K-frame window costs **one dispatch**, not K.
    ///
    /// This is the variant the README argued was the only one that could pay. Loop-level
    /// speculation loses because ~90 % of joint calls emit a token, so a K-wide window is
    /// discarded almost immediately, paying K× the calls to skip 10 % of the round trips.
    /// Here the window is one call regardless of K, so the wasted arithmetic is nearly free
    /// and what is saved is dispatch. Durations are [0…4], so K ≥ 5 covers every frame the
    /// loop can reach without emitting; the greedy result is unchanged by construction.
    public func decodeBatchedJoint(enc: [Float], frames T: Int) async throws -> ChunkResult {
        guard let batched = batchedJoint, batchK > 0 else {
            throw GraphError.message("no batched joint loaded")
        }
        var result = ChunkResult()
        result.frames = T
        let H = config.hidden, V = config.vocabSize, D = config.durations.count
        let K = batchK
        let stateCount = config.layers * H

        var h = [Float](repeating: 0, count: stateCount)
        var c = [Float](repeating: 0, count: stateCount)
        var dec: [Float]

        let t0 = Date()
        (dec, h, c) = try await predictStep(token: Int32(config.blank), h: h, c: c)
        result.predictorCalls += 1

        var frame = 0
        var emitted: [Int] = []
        let emissionCap = 12 * T
        var windowBase = -1
        var window: [(token: Int, duration: Int)] = []
        // Reused staging buffer: rows past the end of the encoder output stay zero and are
        // never read back, so the padding costs arithmetic but never correctness.
        var block = [Float](repeating: 0, count: K * H)

        while frame < T && emitted.count < emissionCap {
            if frame < windowBase || frame >= windowBase + window.count {
                let lo = frame
                let hi = min(T, lo + K)
                for i in 0..<(K * H) { block[i] = 0 }
                enc.withUnsafeBufferPointer { src in
                    block.withUnsafeMutableBufferPointer { dst in
                        dst.baseAddress!.update(from: src.baseAddress! + lo * H, count: (hi - lo) * H)
                    }
                }
                let outs = try await batched.run(
                    ["dec_out": ndarray(dec, shape: [1, H]),
                     "enc_frames": ndarray(block, shape: [K, H])],
                    wanting: ["token_logits", "dur_logits"])
                result.jointCalls += 1                       // ONE dispatch for the window
                result.speculativeJointCalls += (hi - lo) - 1
                window = (0..<(hi - lo)).map { j in
                    (token: argmax(outs[0], offset: j * V, count: V),
                     duration: config.durations[argmax(outs[1], offset: j * D, count: D)])
                }
                windowBase = lo
            }
            let step = window[frame - windowBase]
            var duration = step.duration
            if step.token == config.blank && duration == 0 { duration = 1 }
            let emissionFrame = frame
            frame += duration
            if step.token != config.blank {
                emitted.append(step.token)
                result.frameIndices.append(emissionFrame)
                (dec, h, c) = try await predictStep(token: Int32(step.token), h: h, c: c)
                result.predictorCalls += 1
                window = []                  // predictor state changed: window is stale
                windowBase = -1
            } else {
                result.usefulSpeculations += 1
            }
        }
        result.loopSeconds = -t0.timeIntervalSinceNow
        result.tokens = emitted
        return result
    }

    /// Run the joint for encoder frames `[lo, hi)` against one `dec_out`, concurrently.
    /// The graph is stateless, so the calls are independent and the runtime can overlap them.
    private func jointBatch(dec: [Float], enc: [Float], from lo: Int, to hi: Int)
        async throws -> [(token: Int, duration: Int)] {
        let H = config.hidden
        let durations = config.durations
        let joint = self.joint
        return try await withThrowingTaskGroup(of: (Int, Int, Int).self) { group in
            for f in lo..<hi {
                let encFrame = Array(enc[(f * H)..<((f + 1) * H)])
                group.addTask {
                    let outs = try await joint.run(
                        ["dec_out": ndarray(dec, shape: [1, H]),
                         "enc_frame": ndarray(encFrame, shape: [1, H])],
                        wanting: ["token_logits", "dur_logits"])
                    return (f, argmax(outs[0]), durations[argmax(outs[1])])
                }
            }
            var byFrame = [Int: (Int, Int)]()
            for try await (f, token, duration) in group { byFrame[f] = (token, duration) }
            return (lo..<hi).map { (token: byFrame[$0]!.0, duration: byFrame[$0]!.1) }
        }
    }

    /// One predictor step: (token, h, c) → (dec_out, h', c').
    func predictStep(token: Int32, h: [Float], c: [Float]) async throws -> ([Float], [Float], [Float]) {
        let L = config.layers, H = config.hidden
        let outs = try await predictor.run(
            ["token": ndarray([token], shape: [1, 1]),
             "h": ndarray(h, shape: [L, 1, H]),
             "c": ndarray(c, shape: [L, 1, H])],
            wanting: ["dec_out", "h_out", "c_out"])
        return (outs[0], outs[1], outs[2])
    }

    /// The GPU half of one chunk: mel (CPU/BLAS) + the encoder pass (GPU).
    ///
    /// Split out from `transcribe` so the bench/parity drivers can run it for chunk *n+1*
    /// concurrently with the host-side TDT loop of chunk *n*. The two halves use disjoint
    /// hardware: the encoder is a single big GPU dispatch, the loop is ~284 tiny CPU-pinned
    /// calls, so overlapping them should hide whichever is shorter.
    public struct FrontHalf: Sendable {
        public var enc: [Float]
        public var frames: Int
        public var melSeconds: Double
        public var encoderSeconds: Double
    }

    /// A front half whose encoder pass has been **encoded onto the GPU stream but not awaited**.
    /// The mel is already done (it is CPU/BLAS work); only the encoder result is outstanding.
    public struct PendingFront: @unchecked Sendable {
        var pending: Graph.Pending
        var melSeconds: Double
        var encodedAt: Date
        /// Wall time the synchronous `encode(to:)` call itself cost: this is the CPU-side
        /// command-buffer building, and keeping it OFF the critical path is the whole game.
        public var enqueueSeconds: Double
    }

    /// Queue one chunk's encoder pass without waiting for it. Cheap and synchronous apart from
    /// the mel; call it for chunk N+1 *before* awaiting chunk N to keep the GPU queue non-empty.
    public func enqueueFront(samples: [Float]) throws -> PendingFront {
        let tMel = Date()
        let mel = frontend.melBucket(samples)
        let melSeconds = -tMel.timeIntervalSinceNow
        let input = ndarrayFloat16(mel, shape: [1, MelFrontend.melBins, frontend.bucketFrames])
        let tEnq = Date()
        let pending = try encoder.enqueue(["mel": input])
        let enqueueSeconds = -tEnq.timeIntervalSinceNow
        return PendingFront(pending: pending, melSeconds: melSeconds, encodedAt: Date(),
                            enqueueSeconds: enqueueSeconds)
    }

    /// Block on a queued encoder pass and turn it into an ordinary `FrontHalf`.
    public func awaitFront(_ p: PendingFront) async throws -> FrontHalf {
        let enc = try await p.pending.take(["enc_proj"])[0]
        return FrontHalf(enc: enc, frames: enc.count / config.hidden,
                         melSeconds: p.melSeconds,
                         encoderSeconds: -p.encodedAt.timeIntervalSinceNow)
    }

    public func frontHalf(samples: [Float]) async throws -> FrontHalf {
        let tMel = Date()
        let mel = frontend.melBucket(samples)
        let melSeconds = -tMel.timeIntervalSinceNow
        let tEnc = Date()
        let (enc, frames) = try await encode(mel)
        return FrontHalf(enc: enc, frames: frames, melSeconds: melSeconds,
                         encoderSeconds: -tEnc.timeIntervalSinceNow)
    }

    /// The host half, run on a **specific** decode replica. This is what a chunk-parallel worker
    /// calls: the plain greedy loop only, since the speculative and batched-joint variants were
    /// both refuted in earlier rounds and carrying them into the parallel driver would only widen
    /// the surface the parity gate has to cover.
    public func backHalf(_ front: FrontHalf, decoder: TDTDecoder) async throws -> ChunkResult {
        var result = try await decoder.decode(enc: front.enc, frames: front.frames)
        result.melSeconds = front.melSeconds
        result.encoderSeconds = front.encoderSeconds
        return result
    }

    /// The host half: whichever decode variant this engine is configured for.
    public func backHalf(_ front: FrontHalf, lookahead: Int = 1) async throws -> ChunkResult {
        var result: ChunkResult
        if batchedJoint != nil && batchK > 0 {
            result = try await decodeBatchedJoint(enc: front.enc, frames: front.frames)
        } else if lookahead > 1 {
            result = try await decodeSpeculative(enc: front.enc, frames: front.frames,
                                                 lookahead: lookahead)
        } else {
            result = try await decode(enc: front.enc, frames: front.frames)
        }
        result.melSeconds = front.melSeconds
        result.encoderSeconds = front.encoderSeconds
        return result
    }

    /// How many encoder frames actually carry audio, for `sampleCount` real samples.
    ///
    /// The encoder consumes a fixed `bucketFrames` mel window and emits `bucketFrames / 8` frames
    /// (2885 → 361), so a short utterance silence-padded up to the bucket still yields 361 frames,
    /// and the TDT loop, which has no length mask, will happily decode the padded tail. On a
    /// long-form chunk that tail is a fraction of a second and harmless; on a 2 s LibriSpeech
    /// utterance it is 27 s of silence, and the model hallucinates across it.
    ///
    /// The margin is deliberate: clipping a frame that still holds speech costs a deletion, which
    /// is worse than decoding a little extra silence. One encoder frame is 8 × 160 samples = 80 ms.
    public func validEncoderFrames(sampleCount: Int, margin: Int = 2) -> Int {
        let melFrames = (sampleCount + MelFrontend.hop - 1) / MelFrontend.hop
        return (melFrames + 7) / 8 + margin
    }

    /// Waveform → tokens, with the per-stage timing split.
    /// `lookahead > 1` selects the speculative joint path.
    ///
    /// `maxFrames` caps how far the TDT loop walks the encoder output. Default `nil` decodes all
    /// of it, which is the shipping behaviour and leaves every earlier measurement untouched.
    public func transcribe(samples: [Float], lookahead: Int = 1,
                           maxFrames: Int? = nil) async throws -> ChunkResult {
        let tMel = Date()
        let mel = frontend.melBucket(samples)
        let melSeconds = -tMel.timeIntervalSinceNow

        let tEnc = Date()
        let (enc, allFrames) = try await encode(mel)
        let frames = min(allFrames, maxFrames ?? allFrames)
        let encSeconds = -tEnc.timeIntervalSinceNow

        var result: ChunkResult
        if batchedJoint != nil && batchK > 0 {
            result = try await decodeBatchedJoint(enc: enc, frames: frames)
        } else if lookahead > 1 {
            result = try await decodeSpeculative(enc: enc, frames: frames, lookahead: lookahead)
        } else {
            result = try await decode(enc: enc, frames: frames)
        }
        result.melSeconds = melSeconds
        result.encoderSeconds = encSeconds
        return result
    }

    public func text(_ tokens: [Int]) -> String { tokenizer.decode(tokens) }
}

/// First-maximum argmax: matches numpy's tie-breaking, which matters on fp16 near-ties.
@inlinable
public func argmax(_ values: [Float]) -> Int {
    var best = 0
    var bestValue = values[0]
    for i in 1..<values.count where values[i] > bestValue {
        bestValue = values[i]
        best = i
    }
    return best
}

/// Same first-maximum rule over one row of a flattened `[K, count]` output. Returns the index
/// *within* the row, so callers do not have to subtract the offset back off.
@inlinable
public func argmax(_ values: [Float], offset: Int, count: Int) -> Int {
    var best = 0
    var bestValue = values[offset]
    for i in 1..<count where values[offset + i] > bestValue {
        bestValue = values[offset + i]
        best = i
    }
    return best
}
