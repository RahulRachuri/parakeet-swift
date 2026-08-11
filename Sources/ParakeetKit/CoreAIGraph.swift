import CoreAI
import Foundation

/// Which compute unit a bundle should be specialized for.
///
/// The zoo's rule: parity runs may use `cpuOnly`, but anything we *time* must use a
/// real preference (`.gpu` / `.cpu`), never the `cpu_only` parity mode.
public enum ComputeUnit: String, Sendable, CaseIterable {
    case gpu
    case cpu
    case ane

    var kind: ComputeUnitKind {
        switch self {
        case .gpu: return .gpu
        case .cpu: return .cpu
        case .ane: return .neuralEngine
        }
    }
}

public enum GraphError: Error, CustomStringConvertible {
    case message(String)

    public var description: String {
        switch self { case .message(let m): return m }
    }
}

/// A thin, synchronous-feeling wrapper around one Core AI `.aimodel` bundle with a single
/// `main` graph (which is what all three Parakeet bundles are).
///
/// Two Swift things worth noting for a C++ reader:
///
///  * `NDArray` is a value-ish struct whose storage the runtime owns. We never manage that
///    memory by hand; we hand the runtime shapes and it hands back buffers.
///  * `run` is `async` because the runtime dispatches to the GPU and awaits completion. The
///    `await` is a suspension point, not a thread: the TDT loop stays single-threaded and
///    strictly ordered, exactly like the Python reference loop.
public final class Graph: @unchecked Sendable {
    public let url: URL
    public let unit: ComputeUnit
    private let model: AIModel
    /// Internal (not private) so the zero-allocation decode loop can call
    /// `function.run(inputs:outputViews:)` directly: the `outputViews` path hands the runtime
    /// caller-owned buffers, and wrapping it here would just re-hide the lifetime the
    /// `~Escapable` view types need to see.
    let function: InferenceFunction
    public let descriptor: InferenceFunctionDescriptor
    /// An explicit Metal command stream for the `encode(to:)` submission path (see `enqueue`).
    /// `run()` does not use this: it manages its own submission internally.
    private let stream = ComputeStream()

    /// - Parameter useDefaultOptions: load with the framework's `SpecializationOptions()`
    ///   default instead of an explicit preference. This is the documented failure mode: it is
    ///   a diagnostic knob, not something the engine ever uses.
    public init(url: URL, unit: ComputeUnit, useDefaultOptions: Bool = false) async throws {
        self.url = url
        self.unit = unit
        // Explicit compute-unit preference. `AIModel(contentsOf:)` with `.default` options
        // picks the ANE for these graphs and fails to load them (see the zoo's swift-runtime
        // notes), so we always state the preference, the same thing the Python gate does with
        // `SpecializationOptions.from_preferred_compute_unit_kind(...)`.
        // Note: `SpecializationOptions` has no zero-argument initialiser: the only ways to
        // build one are the two statics (`.default`, `.cpuOnly`) and
        // `init(preferredComputeUnitKind:)`. So "default options" means the static.
        let options = useDefaultOptions
            ? SpecializationOptions.default
            : SpecializationOptions(preferredComputeUnitKind: unit.kind)
        self.model = try await AIModel(contentsOf: url, options: options)
        guard let d = model.functionDescriptor(for: "main") else {
            throw GraphError.message("\(url.lastPathComponent): no 'main' graph")
        }
        self.descriptor = d
        guard let fn = try model.loadFunction(named: "main") else {
            throw GraphError.message("\(url.lastPathComponent): could not load 'main'")
        }
        self.function = fn
    }

    public var inputNames: [String] { descriptor.inputNames }
    public var outputNames: [String] { descriptor.outputNames }

    /// Shape the graph declares for an input (dynamic dims come back negative).
    public func inputShape(_ name: String) -> [Int]? {
        guard case .ndArray(let d)? = descriptor.inputDescriptor(of: name) else { return nil }
        return d.shape
    }

    public func outputShape(_ name: String) -> [Int]? {
        guard case .ndArray(let d)? = descriptor.outputDescriptor(of: name) else { return nil }
        return d.shape
    }

    /// Run the graph once. Inputs are already-built `NDArray`s; outputs come back flattened to
    /// `[Float]` in row-major order (the runtime may emit fp16, so we always widen).
    public func run(_ inputs: [String: NDArray]) async throws -> [String: [Float]] {
        var outputs = try await function.run(inputs: inputs)
        var result: [String: [Float]] = [:]
        for name in descriptor.outputNames {
            guard let nd = outputs.remove(name)?.ndArray else {
                throw GraphError.message("missing output '\(name)'")
            }
            result[name] = flattenAsFloat(nd)
        }
        return result
    }

    /// Run the graph once, returning only the named outputs (avoids flattening ones we discard).
    public func run(_ inputs: [String: NDArray], wanting names: [String]) async throws -> [[Float]] {
        var outputs = try await function.run(inputs: inputs)
        return try names.map { name in
            guard let nd = outputs.remove(name)?.ndArray else {
                throw GraphError.message("missing output '\(name)'")
            }
            return flattenAsFloat(nd)
        }
    }

    // MARK: - streamed submission

    /// Outputs of an `enqueue`d run that has been *encoded* but not necessarily *executed*.
    ///
    /// For a C++ reader: think of these as futures over GPU work that is already sitting in a
    /// command queue. Nothing here blocks until you call `take`.
    public struct Pending: @unchecked Sendable {
        fileprivate let values: [String: InferenceFunction.AsyncValue]
        fileprivate let outputNames: [String]

        /// Await the named outputs, flattened to `[Float]`. This is the only blocking point.
        public func take(_ names: [String]) async throws -> [[Float]] {
            var out: [[Float]] = []
            out.reserveCapacity(names.count)
            for name in names {
                guard let value = values[name], let nd = try await value.ndArray else {
                    throw GraphError.message("missing output '\(name)'")
                }
                out.append(flattenAsFloat(nd))
            }
            return out
        }
    }

    /// Encode one invocation onto this graph's `ComputeStream` and return immediately.
    ///
    /// This is the whole point of the experiment: `run()` is `async` and only lets one
    /// invocation's worth of command buffers exist at a time, so the GPU drains between
    /// submissions. `encode(to:)` is *synchronous*: it appends this invocation's command
    /// buffers to the stream and hands back futures, so the host can queue chunk N+1's work
    /// while chunk N is still executing, and the GPU never sees an empty queue.
    public func enqueue(_ inputs: [String: NDArray]) throws -> Pending {
        let async_ = inputs.mapValues { InferenceFunction.AsyncValue($0) }
        let values = try function.encode(inputs: async_, to: stream)
        return Pending(values: values, outputNames: descriptor.outputNames)
    }

    /// Wait until everything currently encoded on this graph's stream has finished.
    public func drain() async { await stream.currentWorkCompleted() }
}

// MARK: - raw buffer helpers (zero-allocation decode)

/// Zero a float32 NDArray's contents in place.
func zeroFloats(_ nd: inout NDArray, count: Int) {
    var v = nd.mutableView(as: Float.self)
    v.withUnsafeMutablePointer { p, _, _ in p.update(repeating: 0, count: count) }
}

/// memcpy between two float32 NDArrays the runtime owns (no Swift array hop).
func copyFloats(from src: inout NDArray, to dst: inout NDArray, count: Int) {
    var sv = src.mutableView(as: Float.self)
    var dv = dst.mutableView(as: Float.self)
    sv.withUnsafeMutablePointer { sp, _, _ in
        dv.withUnsafeMutablePointer { dp, _, _ in dp.update(from: sp, count: count) }
    }
}

/// First-maximum argmax read straight out of a runtime buffer: the same tie-breaking rule as
/// `argmax(_ values: [Float])`, without materialising the logit vector as a Swift array first.
/// (In the baseline marshalling that allocation-and-copy happened once per joint call to
/// produce two integers.)
@inline(__always)
func argmaxNDArray(_ nd: inout NDArray, count: Int) -> Int {
    var v = nd.mutableView(as: Float.self)
    return v.withUnsafeMutablePointer { p, _, _ in
        var best = 0
        var bestValue = p[0]
        for i in 1..<count where p[i] > bestValue {
            bestValue = p[i]
            best = i
        }
        return best
    }
}

// MARK: - NDArray construction / reading

/// Build a float32 NDArray from a Swift array.
public func ndarray(_ values: [Float], shape: [Int]) -> NDArray {
    NDArray(scalars: values, shape: shape)
}

/// Build a float16 NDArray from float32 values (the encoder takes fp16 mel).
public func ndarrayFloat16(_ values: [Float], shape: [Int]) -> NDArray {
    NDArray(scalars: values.map { Float16($0) }, shape: shape)
}

public func ndarray(_ values: [Int32], shape: [Int]) -> NDArray {
    NDArray(scalars: values, shape: shape)
}

/// Flatten any float NDArray output to `[Float]`, row-major, widening fp16 if needed.
/// (Apple ships an equivalent helper inside `CoreAIShared`; this is the same logic, written
/// out so the package depends on nothing but the system framework.)
public func flattenAsFloat(_ array: NDArray) -> [Float] {
    switch array.scalarType {
    case .float16: return flatten(array, as: Float16.self)
    case .float32: return flatten(array, as: Float.self)
    default: preconditionFailure("unsupported output scalar type \(array.scalarType)")
    }
}

private func flatten<T: BinaryFloatingPoint & BitwiseCopyable>(_ array: NDArray, as _: T.Type) -> [Float] {
    let shape = array.shape
    let total = shape.reduce(1, *)
    var out = [Float](repeating: 0, count: total)
    array.view(as: T.self).withUnsafePointer { ptr, shp, strides in
        // Fast path: contiguous row-major, which every Parakeet output is.
        var expected = 1
        var contiguous = true
        for d in stride(from: shp.count - 1, through: 0, by: -1) {
            if strides[d] != expected { contiguous = false; break }
            expected *= shp[d]
        }
        if contiguous {
            for i in 0..<total { out[i] = Float(ptr[i]) }
            return
        }
        var idx = [Int](repeating: 0, count: shp.count)
        for i in 0..<total {
            var off = 0
            for d in 0..<shp.count { off += idx[d] * strides[d] }
            out[i] = Float(ptr[off])
            var d = shp.count - 1
            while d >= 0 {
                idx[d] += 1
                if idx[d] < shp[d] { break }
                idx[d] = 0
                d -= 1
            }
        }
    }
    return out
}
