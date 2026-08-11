import Foundation
import ParakeetKit

// parakeet-swift: a native Swift host for the Parakeet-v2 Core AI bundles.
//
//   probe        <bundle.aimodel>          print a graph's signature
//   gate-mel                               Swift mel vs the Python reference mel
//   gate-encoder                           Swift encoder pass vs the Python engine output
//   parity       [--limit N]               all 152 chunks vs the validated token streams
//   leak         [--calls N] [--unit u]    hammer the joint bundle, watch for the IOSurface abort
//   bench        [--plan e/p/j] [--limit]  timed run (RTF + per-stage split)
//
// Every path is resolved from the environment, then from the working directory;
// nothing is compiled in. `transcribe`/`serve` need only the artifacts; the gate,
// parity and bench commands additionally want a corpus and the stored reference
// tensors, which are development inputs and are not part of this repository.

let defaults = Defaults()

struct Defaults {
    /// Env override, else a sibling of the working directory. Set by
    /// `PARAKEET_ARTIFACTS`, or per-run with `--artifacts DIR`.
    let artifacts = Defaults.dir("PARAKEET_ARTIFACTS", default: "artifacts_v2")
    /// Corpus root for `parity`/`bench`: expects `work/<name>/chunks.json` and
    /// `results/<name>/tokens.json` under it. `PARAKEET_BENCH`.
    let benchRoot = Defaults.dir("PARAKEET_BENCH", default: "bench")
    /// Float32 tensors dumped from the Python reference path by
    /// `tools/dump_reference.py`, consumed by the `gate-*` commands.
    /// `PARAKEET_REFERENCE`.
    let reference = Defaults.dir("PARAKEET_REFERENCE", default: "reference")

    /// Two *independent* names, not one: the cut manifest is keyed by recording
    /// (`work/<corpus>/`) while the gold token stream is keyed by recording *and*
    /// the engine that produced it (`results/<set>/`), so one recording can have
    /// several reference streams. Collapsing them into a single name silently
    /// looks for the gold under the manifest's name and fails to find it.
    /// Both commands also accept `--manifest` / `--gold` as full-path overrides.
    static func env(_ key: String, _ fallback: String) -> String {
        ProcessInfo.processInfo.environment[key] ?? fallback
    }
    var manifest: URL {
        benchRoot.appendingPathComponent("work/\(Defaults.env("PARAKEET_CORPUS", "corpus"))/chunks.json")
    }
    var goldTokens: URL {
        benchRoot.appendingPathComponent("results/\(Defaults.env("PARAKEET_GOLD_SET", "gold"))/tokens.json")
    }

    private static func dir(_ variable: String, default fallback: String) -> URL {
        if let value = ProcessInfo.processInfo.environment[variable], !value.isEmpty {
            return URL(fileURLWithPath: (value as NSString).expandingTildeInPath)
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(fallback)
    }
}

// MARK: - tiny argument helpers

let argv = Array(CommandLine.arguments.dropFirst())
func flag(_ name: String) -> Bool { argv.contains("--\(name)") }
func option(_ name: String) -> String? {
    guard let i = argv.firstIndex(of: "--\(name)"), i + 1 < argv.count else { return nil }
    return argv[i + 1]
}
func intOption(_ name: String, _ fallback: Int) -> Int { option(name).flatMap(Int.init) ?? fallback }

/// Which build of the three graphs to load.
///
/// `--graph-ext h16s.aimodelc` (plus `--artifacts <aot dir>`) swaps the JIT-specialized source
/// bundles for the ahead-of-time compiled slice. `--artifacts` alone just relocates the source
/// bundles. Either way the tokenizer/filterbank keep coming from the source tree, since
/// `coreai-build` does not carry them.
func artifactPaths() -> ArtifactPaths {
    let dir = option("artifacts").map { URL(fileURLWithPath: $0) } ?? defaults.artifacts
    // Assets (tokenizer + mel filterbank) live alongside the source bundles and follow
    // `--artifacts` by default. They need to be separable only for the AOT case: `coreai-build`
    // does not carry them, so a run pointed at an AOT slice must still read them from the
    // source tree, hence `--assets` / PARAKEET_ASSETS. Defaulting these to a *fixed* directory
    // rather than to `dir` was a latent bug that only stayed hidden while the fixed default
    // happened to be the same directory everyone passed.
    let assets = option("assets").map { URL(fileURLWithPath: $0) }
        ?? ProcessInfo.processInfo.environment["PARAKEET_ASSETS"].map { URL(fileURLWithPath: $0) }
        ?? dir
    // `--mel-frames L` selects a *differently bucketed* encoder export: it picks the bundle
    // named ...L<L>.aimodel and retargets the mel front end to pad to the same length. The
    // default is the shipped 2885, so omitting the flag reproduces every earlier round exactly.
    var paths = ArtifactPaths(artifactsDirectory: dir,
                              graphExtension: option("graph-ext") ?? "aimodel",
                              assetsDirectory: assets,
                              bucketFrames: intOption("mel-frames",
                                                      MelFrontend.defaultBucketFrames))
    // Per-graph escape hatch, so a run can mix builds (e.g. an AOT-compiled encoder next to
    // JIT-specialized predictor/joint), which is the only mix where AOT is not a regression.
    if let p = option("encoder-path") { paths.encoder = URL(fileURLWithPath: p) }
    if let p = option("predictor-path") { paths.predictor = URL(fileURLWithPath: p) }
    if let p = option("joint-path") { paths.joint = URL(fileURLWithPath: p) }
    return paths
}

/// `--batched-joint <bundle> --batch-k K` selects the re-exported joint that takes
/// `enc_frames [K,640]` and answers for K frames in one dispatch. K must match the bundle's
/// static shape: Core AI graphs are not resizable.
func batchedJointURL() -> URL? { option("batched-joint").map { URL(fileURLWithPath: $0) } }

/// Drive the chunk list with chunk *n+1*'s mel+encoder (GPU) in flight while chunk *n*'s TDT
/// loop (CPU) runs, using a one-deep double buffer.
///
/// The Swift shape worth noting: `Task { ... }` starts running *immediately* and concurrently;
/// `await task.value` is where we join it. So issuing the next front half **before** awaiting
/// the current back half is what creates the overlap. Chunks are still retired in order, and
/// each chunk's decode still sees exactly the encoder output of its own chunk, so the token
/// stream is unchanged: this is a scheduling change, not an approximation.
///
/// `body` is called once per chunk, in order, with that chunk's result.
/// Escape hatch for handing a non-`Sendable` closure to the producer task. Safe here because
/// the closure only reads the immutable, already-loaded WAV.
struct SendableBox<T>: @unchecked Sendable {
    let value: T
    init(_ value: T) { self.value = value }
}

/// A bounded async hand-off queue: the producer blocks once `capacity` items are outstanding,
/// which is what bounds how many encoder passes may be in flight on the GPU at once.
actor BoundedQueue<T: Sendable> {
    private var buffer: [T] = []
    private var takers: [CheckedContinuation<T?, Never>] = []
    private var space: [CheckedContinuation<Void, Never>] = []
    private var finished = false
    private let capacity: Int
    init(capacity: Int) { self.capacity = capacity }

    func put(_ value: T) async {
        while buffer.count >= capacity {
            await withCheckedContinuation { space.append($0) }
        }
        if !takers.isEmpty {
            takers.removeFirst().resume(returning: value)
        } else {
            buffer.append(value)
        }
    }

    func finish() {
        finished = true
        while !takers.isEmpty { takers.removeFirst().resume(returning: nil) }
    }

    func take() async -> T? {
        if !buffer.isEmpty {
            let v = buffer.removeFirst()
            if !space.isEmpty { space.removeFirst().resume() }
            return v
        }
        if finished { return nil }
        return await withCheckedContinuation { takers.append($0) }
    }
}

/// `streamDepth > 0` selects the `ComputeStream` submission path instead of `run()`: chunk
/// encoders are *encoded* onto an explicit Metal queue up to `streamDepth` deep, so the GPU
/// always has queued work and never drains between command buffers. Depth 1 is the honest
/// control for it (encode-then-immediately-await, the same shape as `run()`, different API);
/// depth ≥ 2 is where the cross-chunk gap filling actually happens.
/// Collects out-of-order chunk results so the driver can retire them in order at the end.
/// An actor is the Swift way to get "a mutex around this state" without writing one: every
/// method body runs with exclusive access, and callers `await` their turn.
actor ResultBox {
    private var slots: [Int: ChunkResult] = [:]
    func put(_ index: Int, _ result: ChunkResult) { slots[index] = result }
    func drain() -> [Int: ChunkResult] { slots }
}

func runPipelined(engine: ParakeetEngine,
                  chunks: [ChunkManifest.Chunk],
                  samples: @escaping (ChunkManifest.Chunk) -> [Float],
                  lookahead: Int,
                  pipelined: Bool,
                  streamDepth: Int = 0,
                  decodeWorkers: Int = 1,
                  decoders: [TDTDecoder] = [],
                  noDecode: Bool = false,
                  decodeQoS: TaskPriority? = nil,
                  body: (ChunkManifest.Chunk, ChunkResult) throws -> Void) async throws {
    // ---- Round 6: chunk-parallel decode ------------------------------------------------
    //
    // The encoder was hidden behind the decode loop in Round 5, which left the host-side TDT
    // loop as ~98 % of compute. That loop is strictly sequential *within* a chunk (each joint
    // result decides the next predictor step), but chunks start from zero LSTM state, so
    // different chunks are completely independent. So we widen across chunks, not within one.
    //
    // Three-stage pipeline, and the stage boundaries are chosen to keep every Core AI function
    // touched by exactly one thread:
    //
    //   producer  ──enqueue()──►  [pending queue]  ──►  awaiter  ──►  [front queue]  ──►  N workers
    //   (encoder fn,                                 (encoder fn,                    (own predictor
    //    one thread)                                  same one thread)                + joint each)
    //
    // The awaiter is deliberately NOT folded into the workers: `Pending.take` touches the
    // encoder's `InferenceFunction`, and multi-threaded access to one function is the exact
    // trap that produced garbage in Round 4. Keeping enqueue+take on a single serial path means
    // the encoder side is bit-identical to the parity-verified Round 5 driver, and the only new
    // concurrency is N workers on N *disjoint* predictor/joint pairs.
    if streamDepth > 0 && decodeWorkers > 1 {
        precondition(decoders.count >= decodeWorkers, "need one decoder replica per worker")
        let pendingQueue = BoundedQueue<ParakeetEngine.PendingFront>(capacity: streamDepth)
        // Room for a couple of decoded fronts per worker, so a worker finishing early always has
        // something to pick up, without letting the encoder run arbitrarily far ahead (each
        // FrontHalf holds a [361,640] float array ≈ 0.9 MB).
        let frontQueue = BoundedQueue<(Int, ChunkManifest.Chunk, ParakeetEngine.FrontHalf)>(
            capacity: max(2 * decodeWorkers, 4))
        let ordered = chunks
        let samplesFor = SendableBox(samples)
        let box = ResultBox()

        // `--no-decode` retires fronts without decoding them, which measures the mel+encoder
        // supply rate on its own. That number is the floor the decode side can never beat, so it
        // is what tells us whether a decode optimization still has room to pay.
        let skipDecode = noDecode
        let producer = Task<Void, Error> {
            for chunk in ordered {
                await pendingQueue.put(try engine.enqueueFront(samples: samplesFor.value(chunk)))
            }
            await pendingQueue.finish()
        }
        let awaiter = Task<Void, Error> {
            for (i, chunk) in ordered.enumerated() {
                guard let pending = await pendingQueue.take() else { break }
                await frontQueue.put((i, chunk, try await engine.awaitFront(pending)))
            }
            await frontQueue.finish()
        }
        // `decodeQoS` (transcribe/serve only; bench/parity never pass it) requests a low
        // priority for the worker tasks. NOTE: the request is overridden in practice:
        // `waitForAll()` below escalates the workers to the parent's priority immediately
        // (priority inversion avoidance; escalation propagates through any await). See
        // DecodeShape.decodeQoS for the measurement. `nil` inherits the parent priority.
        try await withThrowingTaskGroup(of: Void.self) { group in
            for w in 0..<decodeWorkers {
                let decoder = decoders[w]
                group.addTask(priority: decodeQoS) {
                    while let (i, _, front) = await frontQueue.take() {
                        if skipDecode {
                            var r = ChunkResult()
                            r.frames = front.frames
                            r.melSeconds = front.melSeconds
                            r.encoderSeconds = front.encoderSeconds
                            await box.put(i, r)
                        } else {
                            await box.put(i, try await engine.backHalf(front, decoder: decoder))
                        }
                    }
                }
            }
            try await group.waitForAll()
        }
        try await producer.value
        try await awaiter.value
        // Retire in chunk order, so `body` sees exactly the sequence the serial driver produced.
        let results = await box.drain()
        for (i, chunk) in ordered.enumerated() {
            guard let r = results[i] else { break }
            try body(chunk, r)
        }
        return
    }
    if streamDepth > 0 {
        // One dedicated producer task does every `encode(to:)`, in chunk order, up to
        // `streamDepth` ahead of the consumer. Two things matter here:
        //
        //  * the encodes stay strictly serialized (one producer), so the stream's command
        //    buffers are appended in chunk order and only ever one `encode` call is live,
        //    which is what keeps us clear of the same-function concurrency trap; and
        //  * the producer runs on a *different* task from the decode, so the ~30 ms of
        //    CPU-side command-buffer building per chunk overlaps the TDT loop instead of
        //    sitting in front of it.
        //
        // The consumer still retires chunks in order, so the token stream is untouched.
        let queue = BoundedQueue<ParakeetEngine.PendingFront>(capacity: streamDepth)
        let ordered = chunks
        let samplesFor = SendableBox(samples)
        let producer = Task<Void, Error> {
            for chunk in ordered {
                await queue.put(try engine.enqueueFront(samples: samplesFor.value(chunk)))
            }
            await queue.finish()
        }
        for chunk in ordered {
            guard let pending = await queue.take() else { break }
            let front = try await engine.awaitFront(pending)
            try body(chunk, try await engine.backHalf(front, lookahead: lookahead))
        }
        try await producer.value
        return
    }
    guard pipelined else {
        for chunk in chunks {
            let front = try await engine.frontHalf(samples: samples(chunk))
            try body(chunk, try await engine.backHalf(front, lookahead: lookahead))
        }
        return
    }
    var inFlight: Task<ParakeetEngine.FrontHalf, Error>?
    for (i, chunk) in chunks.enumerated() {
        // Join chunk n's front half FIRST. Only one front half may be in flight at a time:
        // all three graphs are single `InferenceFunction`s, and two overlapping encoder runs
        // on the same one interleave their output buffers and return garbage (measured:
        // it silently produced 20× the tokens and 57/152 chunk parity).
        let front: ParakeetEngine.FrontHalf
        if let queued = inFlight {
            front = try await queued.value
        } else {
            front = try await engine.frontHalf(samples: samples(chunk))
        }
        // Now launch chunk n+1's GPU work, so it overlaps the CPU decode of chunk n below.
        if i + 1 < chunks.count {
            let s = samples(chunks[i + 1])
            inFlight = Task { try await engine.frontHalf(samples: s) }
        } else {
            inFlight = nil
        }
        try body(chunk, try await engine.backHalf(front, lookahead: lookahead))
    }
}

func parsePlan(_ text: String?) -> ComputePlan {
    // "gpu/gpu/gpu" → encoder/predictor/joint
    guard let parts = text?.split(separator: "/"), parts.count == 3,
          let e = ComputeUnit(rawValue: String(parts[0])),
          let p = ComputeUnit(rawValue: String(parts[1])),
          let j = ComputeUnit(rawValue: String(parts[2])) else { return ComputePlan() }
    return ComputePlan(encoder: e, predictor: p, joint: j)
}

struct GoldChunk { let tokens: [Int]; let text: String }

func loadGold(_ url: URL) throws -> [Int: GoldChunk] {
    let data = try Data(contentsOf: url)
    guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
          let chunks = root["chunks"] as? [[String: Any]] else {
        throw GraphError.message("tokens.json: unexpected shape")
    }
    var out: [Int: GoldChunk] = [:]
    for c in chunks {
        if let i = c["i"] as? Int, let t = c["tokens"] as? [Int] {
            out[i] = GoldChunk(tokens: t, text: (c["text"] as? String) ?? "")
        }
    }
    return out
}

func residentMegabytes() -> Double {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? Double(info.resident_size) / 1_048_576 : -1
}

// MARK: - commands

func cmdProbe() async throws {
    guard argv.count >= 2 else { throw GraphError.message("usage: probe <bundle.aimodel>") }
    let unit = ComputeUnit(rawValue: option("unit") ?? "gpu") ?? .gpu
    // `--default-options` reproduces the documented load failure: no explicit compute-unit
    // preference, so the runtime picks for itself.
    let useDefault = flag("default-options")
    let t0 = Date()
    let g = try await Graph(url: URL(fileURLWithPath: argv[1]), unit: unit,
                            useDefaultOptions: useDefault)
    print(String(format: "loaded %@ on %@%@ in %.2f s", g.url.lastPathComponent,
                 useDefault ? "DEFAULT options" : g.unit.rawValue,
                 useDefault ? "" : " (explicit preference)", -t0.timeIntervalSinceNow))
    for n in g.inputNames { print("  in  \(n) \(g.inputShape(n) ?? [])") }
    for n in g.outputNames { print("  out \(n) \(g.outputShape(n) ?? [])") }
}

func cmdGateMel() throws {
    let paths = artifactPaths()
    // The reference dumps are 2885-only, so this gate is meaningful only at the default bucket.
    let frontend = try MelFrontend(filterbankURL: paths.filterbank, bucketFrames: paths.bucketFrames)
    print("=== mel gate: Swift front-end vs the Python reference (bench/frontend.py) ===")
    for i in 0..<3 {
        let wav = try BinaryIO.readFloat32(defaults.reference.appendingPathComponent("wav_\(i).f32"))
        let want = try BinaryIO.readFloat32(defaults.reference.appendingPathComponent("mel_\(i).f32"))
        let got = frontend.melBucket(wav)
        let maxAbs = Metrics.maxAbsDiff(got, want)
        let cos = Metrics.cosine(got, want)
        let rows = Metrics.perRowCosine(got, want, cols: frontend.bucketFrames)
        print(String(format: "chunk %d  n=%d  maxabs=%.3e  cos=%.9f  per-bin cos mean=%.9f min=%.9f",
                     i, got.count, Double(maxAbs), cos, rows.mean, rows.min))
    }
}

func cmdGateEncoder() async throws {
    let paths = artifactPaths()
    print("=== encoder gate: Swift Core AI pass vs the Python Core AI pass (GPU, same bundle) ===")
    let t0 = Date()
    let encoder = try await Graph(url: paths.encoder, unit: .gpu)
    print(String(format: "encoder loaded in %.2f s", -t0.timeIntervalSinceNow))
    let frontend = try MelFrontend(filterbankURL: paths.filterbank, bucketFrames: paths.bucketFrames)

    for i in 0..<3 {
        let mel = try BinaryIO.readFloat32(defaults.reference.appendingPathComponent("mel_\(i).f32"))
        let want = try BinaryIO.readFloat32(defaults.reference.appendingPathComponent("enc_\(i).f32"))
        // (a) Python mel in, isolates the encoder call itself.
        let gotPy = try await encoder.run(
            ["mel": ndarrayFloat16(mel, shape: [1, MelFrontend.melBins, frontend.bucketFrames])],
            wanting: ["enc_proj"])[0]
        // (b) Swift mel in, the shipping path, end to end.
        let wav = try BinaryIO.readFloat32(defaults.reference.appendingPathComponent("wav_\(i).f32"))
        let swiftMel = frontend.melBucket(wav)
        let gotSwift = try await encoder.run(
            ["mel": ndarrayFloat16(swiftMel, shape: [1, MelFrontend.melBins, frontend.bucketFrames])],
            wanting: ["enc_proj"])[0]

        for (label, got) in [("python-mel", gotPy), ("swift-mel ", gotSwift)] {
            let rows = Metrics.perRowCosine(got, want, cols: 640)
            print(String(format: "chunk %d %@  cos=%.9f  per-frame cos mean=%.9f min=%.9f  maxabs=%.4f",
                         i, label, Metrics.cosine(got, want), rows.mean, rows.min,
                         Double(Metrics.maxAbsDiff(got, want))))
        }
    }
}

func cmdParity() async throws {
    let plan = parsePlan(option("plan"))
    let limit = intOption("limit", Int.max)
    let manifest = try ChunkManifest(url: URL(fileURLWithPath: option("manifest") ?? defaults.manifest.path))
    let gold = try loadGold(URL(fileURLWithPath: option("gold") ?? defaults.goldTokens.path))
    let wav = try PCM16WavFile(url: URL(fileURLWithPath: manifest.wavPath))
    let engine = try await ParakeetEngine(paths: artifactPaths(), plan: plan,
                                          batchedJointURL: batchedJointURL(),
                                          batchK: intOption("batch-k", 0))
    print("=== parity gate: \(plan.label) vs the validated Core AI v2 token streams ===")
    print(String(format: "models loaded in %.2f s; %d chunks, %.1f s of audio",
                 engine.loadSeconds, manifest.chunks.count, manifest.totalSeconds))

    // `--ref-mel-dir` feeds the Python reference mel instead of computing one, which separates
    // "does the Swift runtime host match?" from "does the Swift front-end match?".
    let refMelDir = option("ref-mel-dir").map { URL(fileURLWithPath: $0) }
    if let dir = refMelDir { print("using python reference mels from \(dir.path)") }
    let lookahead = intOption("lookahead", 1)
    if lookahead > 1 { print("speculative joint batching, lookahead K=\(lookahead)") }

    var exactChunks = 0, totalChunks = 0, textMatches = 0
    var agreeTokens = 0, goldTokenCount = 0, gotTokenCount = 0, alignedCommon = 0
    var diverged: [Int] = []
    var textDiverged: [Int] = []
    var audio = 0.0
    let t0 = Date()
    let pipelined = flag("pipeline")
    let streamDepth = intOption("stream-depth", 0)
    let decodeWorkers = max(intOption("decode-workers", 1), 1)
    var decoders: [TDTDecoder] = []
    if decodeWorkers > 1 {
        (decoders, _) = try await engine.makeDecoders(count: decodeWorkers)
    }
    try applyDecodeImpl(engine: engine, decoders: decoders)
    if pipelined { print("pipelined: chunk n decode overlaps chunk n+1 mel+encoder") }
    if streamDepth > 0 { print("ComputeStream submission, queue depth \(streamDepth)") }
    if decodeWorkers > 1 { print("chunk-parallel decode, \(decodeWorkers) workers (own predictor+joint each)") }

    // Per-chunk scoring, shared by both drivers below.
    func score(_ chunk: ChunkManifest.Chunk, _ r: ChunkResult) {
        let want = gold[chunk.index] ?? GoldChunk(tokens: [], text: "")
        totalChunks += 1
        audio += chunk.seconds
        goldTokenCount += want.tokens.count
        gotTokenCount += r.tokens.count
        agreeTokens += zip(r.tokens, want.tokens).reduce(0) { $0 + ($1.0 == $1.1 ? 1 : 0) }
        alignedCommon += r.tokens == want.tokens ? want.tokens.count : lcsLength(r.tokens, want.tokens)
        let exact = r.tokens == want.tokens
        if exact { exactChunks += 1 } else { diverged.append(chunk.index) }
        // The tokenizer is gated separately: our Swift decode of OUR tokens must equal the
        // stored `tokenizers.Tokenizer.decode` text for the same chunk.
        if engine.text(r.tokens) == want.text { textMatches += 1 } else { textDiverged.append(chunk.index) }
        if chunk.index % 20 == 0 || !exact {
            print(String(format: "  chunk %3d  %4d tok (gold %4d) %@  %.2fs",
                         chunk.index, r.tokens.count, want.tokens.count,
                         exact ? "exact" : "DIFF ",
                         r.melSeconds + r.encoderSeconds + r.loopSeconds))
        }
    }

    let selected = manifest.chunks.filter { $0.index < limit }
    if let dir = refMelDir {
        // Python reference mel in, isolates the runtime host from the Swift front-end.
        for chunk in selected {
            let mel = try BinaryIO.readFloat32(dir.appendingPathComponent("mel_\(chunk.index).f32"))
            let (enc, frames) = try await engine.encode(mel)
            let r: ChunkResult
            if engine.batchK > 0 {
                r = try await engine.decodeBatchedJoint(enc: enc, frames: frames)
            } else if lookahead > 1 {
                r = try await engine.decodeSpeculative(enc: enc, frames: frames, lookahead: lookahead)
            } else {
                r = try await engine.decode(enc: enc, frames: frames)
            }
            score(chunk, r)
        }
    } else {
        // Exactly the driver `bench` uses, so the gate validates the path we time.
        try await runPipelined(engine: engine,
                               chunks: selected,
                               samples: { wav.samples(from: $0.start, to: $0.end) },
                               lookahead: lookahead,
                               pipelined: pipelined,
                               streamDepth: streamDepth,
                               decodeWorkers: decodeWorkers,
                               decoders: decoders,
                               body: score)
    }

    let elapsed = -t0.timeIntervalSinceNow
    print("")
    print("chunks token-exact : \(exactChunks)/\(totalChunks)")
    print(String(format: "tokens agreeing    : %d/%d (%.4f%%)   emitted %d  [positional]",
                 agreeTokens, goldTokenCount,
                 100.0 * Double(agreeTokens) / Double(max(goldTokenCount, 1)), gotTokenCount))
    print(String(format: "tokens agreeing    : %d/%d (%.4f%%)  %d differing  [aligned/LCS]",
                 alignedCommon, goldTokenCount,
                 100.0 * Double(alignedCommon) / Double(max(goldTokenCount, 1)),
                 goldTokenCount - alignedCommon))
    print("chunks text-exact  : \(textMatches)/\(totalChunks)")
    if !diverged.isEmpty { print("divergent chunks   : \(diverged)") }
    if !textDiverged.isEmpty { print("text-diff chunks   : \(textDiverged)") }
    print(String(format: "wall %.1f s for %.1f s audio (%.1fx realtime incl. load)",
                 elapsed, audio, audio / (elapsed + engine.loadSeconds)))
}

/// Re-run one chunk and print exactly how it differs from the stored stream. Divergences on
/// this port are fp16 near-ties, so the honest report is "which token, how many, is it stable",
/// not a pass/fail.
func cmdInspect() async throws {
    let plan = parsePlan(option("plan"))
    let index = intOption("chunk", 45)
    let repeats = intOption("repeat", 3)
    let manifest = try ChunkManifest(url: defaults.manifest)
    let gold = try loadGold(defaults.goldTokens)
    let wav = try PCM16WavFile(url: URL(fileURLWithPath: manifest.wavPath))
    guard let chunk = manifest.chunks.first(where: { $0.index == index }),
          let want = gold[index] else { throw GraphError.message("no chunk \(index)") }
    let engine = try await ParakeetEngine(paths: artifactPaths(), plan: plan,
                                          batchedJointURL: batchedJointURL(),
                                          batchK: intOption("batch-k", 0))
    let samples = wav.samples(from: chunk.start, to: chunk.end)
    // `--ref-mel` swaps our Swift mel for the Python reference mel of the same chunk. If a
    // divergence disappears under it, the cause is the front-end's last-ulp, not the loop.
    let refMel: [Float]? = flag("ref-mel")
        ? try BinaryIO.readFloat32(defaults.reference.appendingPathComponent("mel_\(index).f32"))
        : nil
    print("=== inspect chunk \(index) (\(plan.label)), \(repeats) repeats"
          + (refMel == nil ? "" : ", python reference mel") + " ===")
    var runs: [[Int]] = []
    for _ in 0..<repeats {
        if let mel = refMel {
            let (enc, frames) = try await engine.encode(mel)
            runs.append(try await engine.decode(enc: enc, frames: frames).tokens)
        } else {
            runs.append(try await engine.transcribe(samples: samples).tokens)
        }
    }
    print("deterministic across repeats: \(Set(runs.map { $0.description }).count == 1)")
    let got = runs[0]
    print("ours \(got.count) tokens, gold \(want.tokens.count) tokens")
    // First position where they part company, plus the surrounding context.
    var i = 0
    while i < min(got.count, want.tokens.count) && got[i] == want.tokens[i] { i += 1 }
    let lo = max(0, i - 4)
    print("first difference at index \(i)")
    print("  ours: \(Array(got[lo..<min(got.count, i + 6)]))")
    print("  gold: \(Array(want.tokens[lo..<min(want.tokens.count, i + 6)]))")
    // Longest-common-subsequence length gives an alignment-aware agreement figure.
    let common = lcsLength(got, want.tokens)
    print(String(format: "aligned agreement: %d/%d common tokens (%.4f%%), %d differing",
                 common, want.tokens.count,
                 100.0 * Double(common) / Double(want.tokens.count), want.tokens.count - common))
    print("ours text: \(engine.text(got))")
    print("gold text: \(want.text)")
}

/// Classic dynamic-programming LCS length (two rolling rows, since the sequences are ~150 long).
func lcsLength(_ a: [Int], _ b: [Int]) -> Int {
    var previous = [Int](repeating: 0, count: b.count + 1)
    var current = previous
    for i in 1...max(a.count, 1) where !a.isEmpty {
        for j in 1...max(b.count, 1) where !b.isEmpty {
            current[j] = a[i - 1] == b[j - 1] ? previous[j - 1] + 1 : max(previous[j], current[j - 1])
        }
        swap(&previous, &current)
    }
    return previous[b.count]
}

func cmdBench() async throws {
    let plan = parsePlan(option("plan"))
    let limit = intOption("limit", Int.max)
    let manifest = try ChunkManifest(url: URL(fileURLWithPath: option("manifest") ?? defaults.manifest.path))
    let wav = try PCM16WavFile(url: URL(fileURLWithPath: manifest.wavPath))

    let tLoad = Date()
    let engine = try await ParakeetEngine(paths: artifactPaths(), plan: plan,
                                          batchedJointURL: batchedJointURL(),
                                          batchK: intOption("batch-k", 0))
    // `--decode-workers N` gives each of N chunk-parallel workers its own predictor+joint pair.
    let decodeWorkers = max(intOption("decode-workers", 1), 1)
    var decoders: [TDTDecoder] = []
    var replicaLoadSeconds = 0.0
    if decodeWorkers > 1 {
        (decoders, replicaLoadSeconds) = try await engine.makeDecoders(count: decodeWorkers)
    }
    let loadSeconds = -tLoad.timeIntervalSinceNow

    let lookahead = intOption("lookahead", 1)
    var mel = 0.0, enc = 0.0, loop = 0.0, audio = 0.0
    var jointCalls = 0, predictorCalls = 0, tokens = 0
    var wastedCalls = 0, usefulSpeculations = 0
    var firstChunkSeconds = 0.0
    let t0 = Date()
    var n = 0
    try await runPipelined(engine: engine,
                           chunks: manifest.chunks.filter { $0.index < limit },
                           samples: { wav.samples(from: $0.start, to: $0.end) },
                           lookahead: lookahead,
                           pipelined: flag("pipeline"),
                           streamDepth: intOption("stream-depth", 0),
                           decodeWorkers: decodeWorkers,
                           decoders: decoders,
                           noDecode: flag("no-decode")) { chunk, r in
        if n == 0 { firstChunkSeconds = r.melSeconds + r.encoderSeconds + r.loopSeconds }
        mel += r.melSeconds; enc += r.encoderSeconds; loop += r.loopSeconds
        audio += chunk.seconds
        jointCalls += r.jointCalls; predictorCalls += r.predictorCalls; tokens += r.tokens.count
        wastedCalls += r.speculativeJointCalls; usefulSpeculations += r.usefulSpeculations
        n += 1
    }
    let compute = -t0.timeIntervalSinceNow
    let total = compute + loadSeconds
    print("=== bench \(plan.label) \(engine.batchK > 0 ? "batched-joint K=\(engine.batchK)" : "lookahead=\(lookahead)") ===")
    print(String(format: "chunks %d  audio %.1f s  tokens %d", n, audio, tokens))
    print(String(format: "load           %7.2f s  (encoder %.2f  predictor %.2f  joint %.2f%@)",
                 loadSeconds, engine.encoderLoadSeconds,
                 engine.predictorLoadSeconds, engine.jointLoadSeconds,
                 decodeWorkers > 1
                    ? String(format: "  +%d replicas %.2f", decodeWorkers - 1, replicaLoadSeconds)
                    : ""))
    if decodeWorkers > 1 {
        print("decode workers \(decodeWorkers) (own predictor+joint each); "
              + "stage sums below are CPU-time across workers, not wall")
    }
    print(String(format: "mel            %7.2f s  (%.1f%%)", mel, 100 * mel / compute))
    print(String(format: "encoder        %7.2f s  (%.1f%%)", enc, 100 * enc / compute))
    print(String(format: "TDT loop       %7.2f s  (%.1f%%)%@", loop, 100 * loop / compute,
                 decodeWorkers > 1
                    ? String(format: "   [%.2fx concurrency]", loop / compute) : ""))
    print(String(format: "compute total  %7.2f s", compute))
    print(String(format: "RTF excl. load %7.1fx", audio / compute))
    print(String(format: "RTF incl. load %7.1fx", audio / total))
    print(String(format: "cold chunk 1   %7.2f s   warm avg %.2f s",
                 firstChunkSeconds, (compute - firstChunkSeconds) / Double(max(n - 1, 1))))
    print(String(format: "engine calls   joint %d  predictor %d  (%.0f/chunk)",
                 jointCalls, predictorCalls, Double(jointCalls + predictorCalls) / Double(max(n, 1))))
    print(String(format: "per-call       joint+pred %.3f ms", 1000 * loop / Double(jointCalls + predictorCalls)))
    if lookahead > 1 {
        print(String(format: "speculation    %d extra joint calls, %d consumed without a predictor step (%.1f%% useful)",
                     wastedCalls, usefulSpeculations,
                     100.0 * Double(usefulSpeculations) / Double(max(wastedCalls, 1))))
    }
}

/// **Batch transcription of a labelled utterance list**: the public-benchmark path
/// (LibriSpeech test-clean / test-other), as opposed to `bench`/`parity`, which drive the
/// long-form audiobook corpus.
///
/// The shape is deliberately the *same* `ChunkManifest` + `PCM16WavFile` the verified drivers
/// use: the prep step concatenates every utterance of a split into one mono 16-bit PCM wav and
/// emits a manifest whose chunks carry `start`/`end` sample offsets plus the utterance `id`.
/// That reuse is the point: one utterance becomes one chunk, and a chunk is already exactly
/// "silence-pad to the bucket, one mel, one encoder pass, TDT greedy from zero LSTM state".
/// There is no VAD anywhere on this path, and no new audio handling to re-validate.
///
/// Two deliberate differences from `bench`:
///
///  * **Serial, always.** Accuracy runs do not care about wall clock, and the serial driver is
///    the narrowest path to defend. (The chunk-parallel driver is token-identical per the parity
///    gate, but there is no reason to widen the surface for a number that has to be citable.)
///  * **A failing utterance is recorded, not fatal.** Each chunk is wrapped in its own
///    `do`/`catch`; a throw becomes an `error` field on that row and the run continues. Silently
///    dropping a row would quietly shrink the denominator, which is exactly the kind of thing
///    that makes a published WER wrong.
///
/// Utterances longer than the bucket (28.85 s) are cut by the *prep* step into consecutive
/// bucket-sized chunks that share one `id`; the rows are rejoined here in chunk order. The
/// front end would otherwise silently truncate them (`padToBucket` slices), which would show up
/// as a wall of deletions.
func cmdTranscribeList() async throws {
    guard let manifestPath = option("manifest") else {
        throw GraphError.message("usage: transcribe-list --manifest <json> --out <jsonl>")
    }
    let outPath = option("out") ?? "/dev/stdout"
    let plan = parsePlan(option("plan"))
    let limit = intOption("limit", Int.max)

    let maskPadding = flag("mask-padding")

    let manifest = try ChunkManifest(url: URL(fileURLWithPath: manifestPath))
    let wav = try PCM16WavFile(url: URL(fileURLWithPath: manifest.wavPath))
    let engine = try await ParakeetEngine(paths: artifactPaths(), plan: plan)
    // transcribe-list runs everything through `primaryDecoder`, so wiring `--decode-impl`
    // here only needs the engine itself; no worker replicas to touch.
    try applyDecodeImpl(engine: engine, decoders: [])
    let selected = Array(manifest.chunks.prefix(limit))

    FileHandle.standardError.write(Data(String(
        format: "transcribe-list: %d chunks, %.1f s audio, plan %@, bucket %d frames%@\n",
        selected.count, manifest.totalSeconds, plan.label, engine.frontend.bucketFrames,
        maskPadding ? ", decode masked to the audio length" : "").utf8))

    // Hypotheses accumulate per utterance id, in first-seen order, so an utterance that had to be
    // split across chunks comes back out as one row with its pieces joined in order.
    var order: [String] = []
    var pieces: [String: [String]] = [:]
    var failures: [String: String] = [:]
    var totalTokens = 0
    var done = 0

    for chunk in selected {
        let id = chunk.id ?? "chunk_\(chunk.index)"
        if pieces[id] == nil { order.append(id); pieces[id] = [] }
        do {
            let samples = wav.samples(from: chunk.start, to: chunk.end)
            // `--mask-padding` stops the TDT loop at the last encoder frame that carries audio.
            // OFF by default: the shipping path decodes the whole bucket, and that is the number
            // the benchmark reports. This is the diagnostic that separates "the port is wrong"
            // from "the host decodes its own silence padding".
            let cap = maskPadding ? engine.validEncoderFrames(sampleCount: samples.count) : nil
            let r = try await engine.transcribe(samples: samples, maxFrames: cap)
            pieces[id]?.append(engine.text(r.tokens))
            totalTokens += r.tokens.count
        } catch {
            // Record and carry on; the scorer needs to see this row as a failure, not an absence.
            failures[id] = "\(error)"
            pieces[id]?.append("")
        }
        done += 1
        if done % 100 == 0 {
            FileHandle.standardError.write(Data("  \(done)/\(selected.count)\n".utf8))
        }
    }

    // JSONL out: one row per utterance id, in manifest order.
    var out = ""
    for id in order {
        let text = (pieces[id] ?? []).joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        var row: [String: Any] = ["id": id, "text": text]
        if let e = failures[id] { row["error"] = e }
        let data = try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])
        out += String(decoding: data, as: UTF8.self) + "\n"
    }
    try out.write(to: URL(fileURLWithPath: outPath), atomically: true, encoding: .utf8)

    FileHandle.standardError.write(Data(String(
        format: "wrote %d utterances (%d chunks, %d tokens) to %@%@\n",
        order.count, selected.count, totalTokens, outPath,
        failures.isEmpty ? "" : "  (\(failures.count) FAILED: \(failures.keys.sorted()))").utf8))
}

func cmdLeak() async throws {
    let calls = intOption("calls", 25_000)
    let unit = ComputeUnit(rawValue: option("unit") ?? "gpu") ?? .gpu
    let paths = artifactPaths()
    print("=== IOSurface leak test: \(calls) joint calls on \(unit.rawValue) ===")
    print("(the Python bindings abort at ~8000 calls: two outputs/call, ~16k surfaces)")
    let joint = try await Graph(url: paths.joint, unit: unit)
    let H = 640
    var dec = [Float](repeating: 0, count: H)
    var encFrame = [Float](repeating: 0, count: H)
    for i in 0..<H { dec[i] = Float(i % 7) * 0.01; encFrame[i] = Float(i % 11) * 0.01 }

    let t0 = Date()
    var checkpoint = Date()
    for i in 1...calls {
        _ = try await joint.run(["dec_out": ndarray(dec, shape: [1, H]),
                                 "enc_frame": ndarray(encFrame, shape: [1, H])],
                                wanting: ["token_logits", "dur_logits"])
        if i % 1000 == 0 {
            let dt = -checkpoint.timeIntervalSinceNow
            checkpoint = Date()
            print(String(format: "  %6d calls   rss %7.1f MB   last 1000: %.2f s (%.3f ms/call)",
                         i, residentMegabytes(), dt, dt))
            fflush(stdout)
        }
    }
    print(String(format: "survived %d calls in %.1f s, rss %.1f MB: NO abort",
                 calls, -t0.timeIntervalSinceNow, residentMegabytes()))
}

/// **Contention probe.** Does the Core AI runtime let two *independent* `InferenceFunction`s
/// execute at the same time, or does it funnel them through one lane?
///
/// This is the question the whole chunk-parallel-decode idea rests on, and it is worth asking in
/// isolation rather than inferring it from a full pipeline where starvation looks identical to
/// serialization. N threads, N separately-loaded copies of the same bundle, each hammering its
/// own copy with back-to-back `run()` calls. If aggregate throughput scales with N the runtime is
/// concurrent; if it stays flat, every worker is queueing behind the same lock and no amount of
/// pipeline engineering on our side will help.
func cmdContend() async throws {
    let threads = max(intOption("threads", 1), 1)
    let calls = intOption("calls", 3000)
    let unit = ComputeUnit(rawValue: option("unit") ?? "cpu") ?? .cpu
    let which = option("graph") ?? "joint"
    let paths = artifactPaths()
    let url = which == "predictor" ? paths.predictor : paths.joint
    print("=== contention probe: \(threads) threads x \(calls) calls on \(which) (\(unit.rawValue)) ===")
    print("each thread owns a SEPARATELY LOADED copy of the bundle")

    var graphs: [Graph] = []
    let tLoad = Date()
    for _ in 0..<threads { graphs.append(try await Graph(url: url, unit: unit)) }
    print(String(format: "loaded %d copies in %.2f s", threads, -tLoad.timeIntervalSinceNow))

    let H = 640, L = 2
    var dec = [Float](repeating: 0, count: H)
    var encFrame = [Float](repeating: 0, count: H)
    for i in 0..<H { dec[i] = Float(i % 7) * 0.01; encFrame[i] = Float(i % 11) * 0.01 }
    let h = [Float](repeating: 0, count: L * H), c = h
    let decBox = SendableBox(dec), encBox = SendableBox(encFrame)
    let hBox = SendableBox(h), cBox = SendableBox(c)

    let t0 = Date()
    try await withThrowingTaskGroup(of: Double.self) { group in
        for g in graphs {
            group.addTask {
                let tt = Date()
                for _ in 0..<calls {
                    if which == "predictor" {
                        _ = try await g.run(["token": ndarray([Int32(1024)], shape: [1, 1]),
                                             "h": ndarray(hBox.value, shape: [L, 1, H]),
                                             "c": ndarray(cBox.value, shape: [L, 1, H])],
                                            wanting: ["dec_out", "h_out", "c_out"])
                    } else {
                        _ = try await g.run(["dec_out": ndarray(decBox.value, shape: [1, H]),
                                             "enc_frame": ndarray(encBox.value, shape: [1, H])],
                                            wanting: ["token_logits", "dur_logits"])
                    }
                }
                return -tt.timeIntervalSinceNow
            }
        }
        var perThread: [Double] = []
        for try await v in group { perThread.append(v) }
        let wall = -t0.timeIntervalSinceNow
        let total = threads * calls
        print(String(format: "wall %.2f s   aggregate %.0f calls/s   %.3f ms/call effective",
                     wall, Double(total) / wall, 1000 * wall / Double(total)))
        print(String(format: "per-thread wall  min %.2f  max %.2f s   (%.3f ms/call within a thread)",
                     perThread.min() ?? 0, perThread.max() ?? 0,
                     1000 * (perThread.max() ?? 0) / Double(calls)))
        print(String(format: "rss %.1f MB", residentMegabytes()))
    }
}

// MARK: - transcribe / serve (word-timestamp JSON output)

/// One chunk of a transcription with everything the JSON output needs.
struct TranscribedSegment {
    var start: Double          // absolute seconds in the input file
    var end: Double
    var text: String
    var words: [WordTimestamp]
    var tokens: [Int]          // kept for validation (serve-vs-oneshot, parity checks)
}

struct TranscriptionOutput {
    var segments: [TranscribedSegment]
    var transcript: String
    var audioSeconds: Double

    /// The machine-readable shape: Whisper-style segments with word-level times.
    /// Times are `NSDecimalNumber` built from fixed 3-decimal strings so the JSON carries
    /// clean millisecond values instead of double artifacts like `0.24000000000000002`.
    var json: [String: Any] {
        func num(_ t: Double) -> NSDecimalNumber { NSDecimalNumber(string: String(format: "%.3f", t)) }
        return [
            "text": transcript,
            "segments": segments.map { s in
                ["start": num(s.start), "end": num(s.end), "text": s.text,
                 "words": s.words.map { ["word": $0.word, "start": num($0.start), "end": num($0.end)] }]
            },
        ]
    }
}

/// The decode shape `transcribe` and `serve` drive `runPipelined` with. Default is the fast,
/// parity-verified driver: ComputeStream submission depth 2 + 4 chunk-parallel decode workers
/// (each on its own predictor+joint replica). `--stream-depth` / `--decode-workers` (the same
/// flag names `bench`/`parity` use) select any other shape; `--stream-depth 0` with one
/// worker is the plain serial path.
struct DecodeShape {
    var streamDepth: Int
    var decodeWorkers: Int
    var decoders: [TDTDecoder]
    /// Task priority for the decode worker tasks; `nil` inherits (the `default` setting,
    /// and the default). A lower priority here is INERT in practice: the task group's
    /// `waitForAll()` escalates the workers back to the parent's priority before they run
    /// a single chunk (measured 2026-08-06: requested utility=17, effective=21 on every
    /// worker; 3×2 interleaved quiet-machine A/B showed identical wall, CPU, and cluster
    /// residency). Swift escalates through any await by design; pinning decode to
    /// E-cores would need thread-level QoS (dispatch), not task priority. The flag stays
    /// as an honest A/B lever, not because it does anything today.
    var decodeQoS: TaskPriority?
    var decodeQoSLabel: String

    /// Reads the flags and loads the worker replicas (once; serve reuses this for every
    /// request, preserving the model-loaded-once property).
    static func fromFlags(engine: ParakeetEngine) async throws -> DecodeShape {
        let streamDepth = intOption("stream-depth", 2)
        let decodeWorkers = max(intOption("decode-workers", 4), 1)
        let qosLabel = option("decode-qos") ?? "default"
        let decodeQoS: TaskPriority?
        switch qosLabel {
        case "default": decodeQoS = nil
        case "utility": decodeQoS = .utility
        case "background": decodeQoS = .background
        default:
            throw GraphError.message("--decode-qos must be default|utility|background, got '\(qosLabel)'")
        }
        // Replicas only matter on the parallel branch (streamDepth > 0 AND workers > 1);
        // every other shape decodes on the engine's primary decoder.
        let decoders = streamDepth > 0 && decodeWorkers > 1
            ? try await engine.makeDecoders(count: decodeWorkers).decoders
            : [engine.primaryDecoder!]
        try applyDecodeImpl(engine: engine, decoders: decoders)
        return DecodeShape(streamDepth: streamDepth, decodeWorkers: decodeWorkers,
                           decoders: decoders, decodeQoS: decodeQoS, decodeQoSLabel: qosLabel)
    }
}

/// `--decode-impl coreai|accel-joint|accel`: dispatch elimination, in two stages:
/// `accel-joint` swaps only the joint graph for `AccelJoint` (the first spike, kept as an
/// A/B lever); `accel` replaces the whole per-token loop (joint + predictor) with plain
/// Accelerate arithmetic: zero graph calls in the hot path. Same weights either way; one
/// shared instance of each across all decoders (immutable weights, reentrant BLAS).
///
/// `accel` is the DEFAULT: byte-identical to the graph decode on the 152-chunk gate, the
/// full audiobook chapter, the blank-collapse fixture, and LibriSpeech test-clean +
/// test-other (5559 utterances), at −44% decode-loop CPU. `--decode-impl coreai` keeps
/// the graph path around as the A/B reference.
func applyDecodeImpl(engine: ParakeetEngine, decoders: [TDTDecoder]) throws {
    let impl = option("decode-impl") ?? "accel"
    guard impl != "coreai" else { return }
    guard impl == "accel" || impl == "accel-joint" else {
        throw GraphError.message("--decode-impl must be coreai|accel-joint|accel, got '\(impl)'")
    }
    let dir = engine.paths.joint.deletingLastPathComponent()
    let aj = try AccelJoint(artifactsDirectory: dir, config: engine.config)
    let ap = impl == "accel"
        ? try AccelPredictor(artifactsDirectory: dir, config: engine.config) : nil
    for d in [engine.primaryDecoder!] + decoders {
        d.accelJoint = aj
        d.accelPredictor = ap
    }
    FileHandle.standardError.write(Data(
        "decode-impl \(impl): \(impl == "accel" ? "joint+predictor" : "joint") on Accelerate\n".utf8))
}

/// Decode one window `[startSample, endSample)`; if it blank-collapses (zero tokens on > 4 s
/// of audio, see the collapse note in `transcribeFile`), halve and recurse. The 4 s floor
/// bounds the recursion (28.8 → 14.4 → 7.2 → 3.6 s stops) and keeps genuine silence cheap:
/// a short window with no tokens is silence, not a collapse. Word times are absolute.
func decodeWindowWithRetry(engine: ParakeetEngine, wav: PCM16WavFile,
                           startSample: Int, endSample: Int,
                           sr: Double) async throws -> (tokens: [Int], words: [WordTimestamp], subWindows: Int) {
    let r = try await engine.transcribe(samples: wav.samples(from: startSample, to: endSample))
    let start = Double(startSample) / sr, end = Double(endSample) / sr
    if !r.tokens.isEmpty || endSample - startSample <= 4 * Int(sr) {
        return (r.tokens,
                WordTimestamps.words(tokens: r.tokens, frameIndices: r.frameIndices,
                                     tokenizer: engine.tokenizer,
                                     chunkStart: start, chunkEnd: end), 1)
    }
    let mid = startSample + (endSample - startSample) / 2
    let a = try await decodeWindowWithRetry(engine: engine, wav: wav,
                                            startSample: startSample, endSample: mid, sr: sr)
    let b = try await decodeWindowWithRetry(engine: engine, wav: wav,
                                            startSample: mid, endSample: endSample, sr: sr)
    let (tokens, words) = joinAcrossSeam(a: (a.tokens, a.words), b: (b.tokens, b.words),
                                         tokenizer: engine.tokenizer)
    return (tokens, words, a.subWindows + b.subWindows)
}

/// Concatenate two decoded windows. The final text is `decode(tokensA + tokensB)`, which
/// glues the halves seamlessly when the cut landed mid-word (half B's first token carries no
/// ▁), so in that case half B's first word is merged into half A's last, keeping the word
/// list consistent with the text.
func joinAcrossSeam(a: (tokens: [Int], words: [WordTimestamp]),
                    b: (tokens: [Int], words: [WordTimestamp]),
                    tokenizer: ParakeetTokenizer) -> (tokens: [Int], words: [WordTimestamp]) {
    var words = a.words
    var bWords = b.words
    if let firstB = b.tokens.first, !tokenizer.startsWord(firstB),
       !words.isEmpty, !bWords.isEmpty {
        let joined = bWords.removeFirst()
        words[words.count - 1].word += joined.word
        words[words.count - 1].end = joined.end
    }
    return (a.tokens + b.tokens, words + bWords)
}

/// Wav → chunks → decode → segments with word times. This is the one transcription path both
/// `transcribe` and `serve` call, so serve responses are one-shot runs by construction.
///
/// The decode runs through `runPipelined` (the SAME driver `bench` and `parity` gate) in
/// the shape carried by `DecodeShape`. The parallel driver decodes chunks out of order but
/// *retires* them in chunk order (its drain loop walks `ordered.enumerated()`), so `body`
/// below sees chunk 0, 1, 2, … regardless of worker scheduling, and each result arrives with
/// its own chunk's sample offsets; words are stamped against their own chunk's absolute
/// times, never a neighbour's. The precondition pins both properties.
func transcribeFile(engine: ParakeetEngine, path: String, label: String,
                    shape: DecodeShape) async throws -> TranscriptionOutput {
    let wav = try PCM16WavFile(url: URL(fileURLWithPath: path))
    guard wav.sampleRate == 16000 else {
        throw GraphError.message("\(path): expected 16 kHz mono PCM16 (got \(wav.sampleRate) Hz); resample first")
    }
    let sr = Double(wav.sampleRate)
    // Everything reads through the wav's mmap on demand: the whole-file `[Float]` this used
    // to materialise was 253 MB on a 66-minute chapter, and the chunker's prefix-sum array
    // another 507 MB of dirty pages. Streaming both flattened the audio-prep footprint;
    // cuts and per-chunk samples are byte-identical (see cutPoints(sampleCount:) parity note).
    let cuts = LongFormChunker.cutPoints(sampleCount: wav.frameCount,
                                         sampleRate: wav.sampleRate) { wav.samples(from: $0, to: $1) }
    let chunks = cuts.enumerated().map { i, cut in
        ChunkManifest.Chunk(index: i, start: cut.start, end: cut.end,
                            seconds: Double(cut.end - cut.start) / sr)
    }
    FileHandle.standardError.write(Data(String(
        format: "%@: %.1f s audio -> %d chunks (stream-depth %d, decode-workers %d, decode-qos %@)\n",
        label, Double(wav.frameCount) / sr, cuts.count, shape.streamDepth, shape.decodeWorkers,
        shape.decodeQoSLabel).utf8))

    var results: [(chunk: ChunkManifest.Chunk, r: ChunkResult)] = []
    try await runPipelined(engine: engine,
                           chunks: chunks,
                           samples: { wav.samples(from: $0.start, to: $0.end) },
                           lookahead: 1,
                           pipelined: false,
                           streamDepth: shape.streamDepth,
                           decodeWorkers: shape.decodeWorkers,
                           decoders: shape.decoders,
                           decodeQoS: shape.decodeQoS) { chunk, r in
        results.append((chunk, r))
    }
    precondition(results.count == chunks.count &&
                 zip(results, chunks).allSatisfy { $0.chunk.index == $1.index },
                 "driver must retire every chunk, in chunk order")

    // ---- blank-collapse retry -----------------------------------------------------------
    //
    // The TDT loop can decode a whole ~28 s window to ZERO tokens: it takes a blank with
    // duration 4 at every frame and walks off the end. This is a **model + fixed-bucket
    // framing** failure, not a port bug; `bench/probe_collapse.py` showed it is
    // window-bound (shifting or halving the window recovers the full text) and that the
    // PyTorch oracle collapses on exactly the same windows; the validated Python driver
    // ships the same mitigation (`coreai_run.py --collapse-retry`). Narrator/content
    // dependent: LibriVox head/tail windows hit it, the audiobook corpus never does.
    //
    // Mitigation, matching the Python driver: re-decode the collapsed chunk as two halves.
    // Halving loses no audio and stays inside the chunk, so neighbouring chunks (and every
    // chunk that decoded normally) are byte-untouched. Runs serially on the primary decoder
    // after the parallel drain; cost is two extra chunk decodes per collapsed window only.
    // Chunks ≤ 4 s are exempt (same threshold as Python): a short window with no tokens is
    // far more likely to be genuine silence than a collapse.
    var segments: [TranscribedSegment] = []
    var retried: [Int] = []
    for (chunk, r) in results {
        let start = Double(chunk.start) / sr, end = Double(chunk.end) / sr
        if r.tokens.isEmpty && (chunk.end - chunk.start) > 4 * wav.sampleRate {
            // The chunk collapsed: re-decode it as two halves, each of which may recurse
            // once more if it collapses too (measured: some LibriVox head windows collapse
            // at BOTH the 28 s and the 14 s framing but decode fine at 7 s: the mel
            // normalisation spans the padded window, so every window length is a different
            // framing of the same audio).
            let mid = chunk.start + (chunk.end - chunk.start) / 2
            let a = try await decodeWindowWithRetry(engine: engine, wav: wav,
                                                    startSample: chunk.start, endSample: mid, sr: sr)
            let b = try await decodeWindowWithRetry(engine: engine, wav: wav,
                                                    startSample: mid, endSample: chunk.end, sr: sr)
            let (tokens, words) = joinAcrossSeam(a: (a.tokens, a.words), b: (b.tokens, b.words),
                                                 tokenizer: engine.tokenizer)
            segments.append(TranscribedSegment(start: start, end: end,
                                               text: engine.text(tokens),
                                               words: words, tokens: tokens))
            retried.append(chunk.index)
            let note = "\(label): chunk \(chunk.index) blank-collapse retry -> "
                + "\(tokens.count) tokens in \(a.subWindows + b.subWindows) sub-windows\n"
            FileHandle.standardError.write(Data(note.utf8))
        } else {
            segments.append(TranscribedSegment(
                start: start, end: end,
                text: engine.text(r.tokens),
                words: WordTimestamps.words(tokens: r.tokens, frameIndices: r.frameIndices,
                                            tokenizer: engine.tokenizer,
                                            chunkStart: start, chunkEnd: end),
                tokens: r.tokens))
        }
    }
    if !retried.isEmpty {
        FileHandle.standardError.write(Data(
            "\(label): blank-collapse retry on chunks \(retried)\n".utf8))
    }
    let transcript = segments.map(\.text)
        .filter { !$0.isEmpty }
        .joined(separator: " ")
        .trimmingCharacters(in: .whitespacesAndNewlines)
    return TranscriptionOutput(segments: segments, transcript: transcript,
                               audioSeconds: Double(wav.frameCount) / sr)
}

/// `transcribe <wav> [--json PATH]`: plain transcript on stdout, progress on stderr,
/// and (with `--json`) the segments/words JSON written to PATH.
func cmdTranscribe() async throws {
    guard argv.count >= 2, !argv[1].hasPrefix("--") else {
        throw GraphError.message("usage: transcribe <audio_16k_mono.wav> [--json PATH]")
    }
    let engine = try await ParakeetEngine(paths: artifactPaths(), plan: parsePlan(option("plan")))
    let shape = try await DecodeShape.fromFlags(engine: engine)
    let out = try await transcribeFile(engine: engine, path: argv[1], label: "transcribe",
                                       shape: shape)
    if let jsonPath = option("json") {
        var data = try JSONSerialization.data(withJSONObject: out.json, options: [.sortedKeys])
        data.append(Data("\n".utf8))
        try data.write(to: URL(fileURLWithPath: jsonPath))
        FileHandle.standardError.write(Data("wrote \(jsonPath)\n".utf8))
    }
    print(out.transcript)
}

/// `serve` starts a long-running JSON-lines worker: one request per stdin line
/// (`{"audio": "/path.wav"}`), one response object per stdout line, model loaded once.
/// stdout carries ONLY response JSON; every log line goes to stderr. A bad request or a
/// failed file answers `{"error": ...}` on its line; the process never exits on a request.
func cmdServe() async throws {
    let plan = parsePlan(option("plan"))
    let engine = try await ParakeetEngine(paths: artifactPaths(), plan: plan)
    // Worker replicas load once, here, alongside the engine: every request reuses them.
    let shape = try await DecodeShape.fromFlags(engine: engine)
    FileHandle.standardError.write(Data(String(
        format: "serve: models loaded in %.2f s (%@, stream-depth %d, decode-workers %d, decode-qos %@), ready\n",
        engine.loadSeconds, plan.label, shape.streamDepth, shape.decodeWorkers,
        shape.decodeQoSLabel).utf8))

    while let line = readLine(strippingNewline: true) {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty { continue }
        var response: [String: Any]
        do {
            guard let request = try JSONSerialization.jsonObject(with: Data(trimmed.utf8)) as? [String: Any],
                  let audio = request["audio"] as? String else {
                throw GraphError.message("expected {\"audio\": \"/path.wav\"}")
            }
            let t0 = Date()
            let out = try await transcribeFile(engine: engine, path: audio,
                                               label: "serve \(audio)", shape: shape)
            FileHandle.standardError.write(Data(String(
                format: "serve: %@ done in %.2f s (%.1f s audio)\n",
                audio, -t0.timeIntervalSinceNow, out.audioSeconds).utf8))
            response = out.json
        } catch {
            FileHandle.standardError.write(Data("serve: request failed: \(error)\n".utf8))
            response = ["error": "\(error)"]
        }
        var data = try JSONSerialization.data(withJSONObject: response, options: [.sortedKeys])
        data.append(Data("\n".utf8))
        FileHandle.standardOutput.write(data)
    }
}

// MARK: - dispatch

let command = argv.first ?? "help"
do {
    switch command {
    case "probe": try await cmdProbe()
    case "gate-mel": try cmdGateMel()
    case "gate-encoder": try await cmdGateEncoder()
    case "parity": try await cmdParity()
    case "inspect": try await cmdInspect()
    case "bench": try await cmdBench()
    case "transcribe-list": try await cmdTranscribeList()
    case "transcribe": try await cmdTranscribe()
    case "serve": try await cmdServe()
    case "leak": try await cmdLeak()
    case "contend": try await cmdContend()
    default:
        print("""
        usage: parakeet-swift <command> [options]
          probe <bundle.aimodel>
          gate-mel
          gate-encoder
          parity  [--plan gpu/gpu/gpu] [--limit N] [--stream-depth D] [--decode-workers N]
          bench   [--plan gpu/gpu/gpu] [--limit N] [--stream-depth D] [--decode-workers N]
                  [--no-decode]   retire fronts undecoded: measures the mel+encoder floor
                                  (ONLY honoured with --decode-workers > 1; check `tokens 0`)
          common  [--artifacts DIR] [--mel-frames L]
                  --mel-frames selects a differently bucketed encoder (…_L<L>.aimodel) and
                  retargets the mel front end to match. Default 2885. NOTE: the bucket length
                  changes the mel *values* (normalisation spans the padded window), so a new L
                  invalidates the stored token gold; see §7i of the port notes.
          transcribe-list --manifest <json> --out <jsonl> [--limit N] [--plan ...]
                  batch-transcribe a labelled utterance list (LibriSpeech). One chunk = one
                  utterance, silence-padded to the bucket, no VAD; serial; a failing utterance
                  is written with an "error" field rather than aborting the run.
          transcribe <wav> [--json PATH] [--plan ...] [--stream-depth D] [--decode-workers N]
                  [--decode-qos default|utility|background]
                  --decode-qos (default: default) requests a decode WORKER task priority.
                  Known inert under the streamed driver: the group's waitForAll escalates
                  the workers back to the parent's priority (measured; kept as an A/B
                  lever). The encoder submission path always stays at default QoS.
                  --decode-impl coreai|accel-joint|accel (default: accel); accel runs the
                  whole per-token loop (joint + predictor) as direct Accelerate arithmetic
                  (AccelJoint/AccelPredictor) instead of the compiled graphs: same weights,
                  byte-identical output, -44% decode-loop CPU (each graph call was ~123 us
                  of dispatch around ~tens of us of math). Needs the joint_head_* and
                  pred_* f32 blobs next to the bundles (export_{joint,predictor}_weights.py).
                  coreai = the all-graphs reference path; accel-joint = first-stage A/B lever.
                  transcribe one 16 kHz mono PCM16 wav: chunked with the audio_prep.py cut
                  policy, decoded on the streamed chunk-parallel driver (default depth 2,
                  4 workers; --stream-depth 0 --decode-workers 1 = the serial path); plain
                  transcript on stdout, and with --json a
                  {"text","segments":[{start,end,text,words:[{word,start,end}]}]} file with
                  absolute times in seconds (words on the 80 ms encoder-frame grid).
          serve   [--plan ...] [--stream-depth D] [--decode-workers N]
                  [--decode-qos default|utility|background]
                  JSON-lines worker: one {"audio":"/path.wav"} request per stdin line, one
                  response (same shape as --json) per stdout line; model + worker replicas
                  loaded once; decode shape as for transcribe; all logging on stderr.
          leak    [--calls N] [--unit gpu|cpu|ane]
          contend [--threads N] [--calls N] [--graph joint|predictor] [--unit cpu|gpu]
                  N threads on N separately-loaded copies: does the runtime run them concurrently?
        """)
        exit(2)
    }
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
