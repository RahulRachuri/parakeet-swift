import Accelerate
import Foundation

/// Log-mel front-end for Parakeet's fixed 30 s encoder bucket.
///
/// This is a literal port of the algorithm the Python gate validated: a Swift-semantics
/// simulation scored 82/82 token-exact against the reference front end. The pipeline is:
///
///   1. **silence-pad** the *waveform* to exactly 2885 × 160 = 461_600 samples (28.85 s),
///      or truncate. This is the single most important line in the file: padding the *mel
///      matrix* with zeros instead scores 22/82: the encoder has no length mask, so a
///      plateau of exact zeros is out-of-distribution, whereas digital silence still produces
///      a finite log-mel floor.
///   2. pre-emphasis y[t] = x[t] − 0.97·x[t−1] (y[0] = x[0]).
///   3. a 400-sample periodic=false Hann window placed *centred* inside a 512-point frame,
///      with 256 zeros of constant padding on each side of the signal (`center=True`,
///      `pad_mode="constant"`).
///   4. a one-sided 512-point DFT, computed as two real matrix multiplies against precomputed
///      cos/sin tables (the same "manual DFT" the gate simulated, just handed to BLAS).
///   5. power → 128-bin Slaney mel filterbank → log(x + 2⁻²⁴).
///   6. per-mel-bin normalisation over the whole padded window, **unbiased** (N−1) variance,
///      divided by (std + 1e-5).
///
/// Output is `[128 × 2885]` row-major (mel-major), which is exactly the `[1,128,2885]` the
/// encoder graph wants.
public struct MelFrontend: Sendable {
    public static let nFFT = 512
    public static let winLength = 400
    public static let hop = 160
    public static let melBins = 128
    public static let freqBins = nFFT / 2 + 1        // 257
    /// The shipped bucket: 2885 frames = 461_600 samples = 28.85 s.
    ///
    /// Not 30 s, despite the name the port has carried since Round 1, and the *chunker* is
    /// built around it: `Chunker` cuts at silence minima inside a [24.0 s, 28.8 s] band
    /// precisely so that no chunk can exceed this bucket. That is why the bucket cannot
    /// usefully be shortened: see `bucketFrames` below.
    public static let defaultBucketFrames = 2885
    public static let logGuard = Float(exp2(-24.0))
    public static let normEpsilon: Float = 1e-5

    /// Mel frames in the encoder's static input, i.e. which encoder bundle this front end
    /// feeds. Instance state, not a constant, only so that a *shorter* bucket can be measured
    /// (Round 7); the default is the shipped 2885 and nothing in the default path changes.
    ///
    /// Note this is baked into the features, not just the shape: step (6) below normalises each
    /// mel bin over the whole padded window, so two bucket lengths give *different* mel values
    /// for the same audio. A mel computed at 2885 and truncated is not the mel a 2880 pipeline
    /// would see.
    public let bucketFrames: Int
    public var bucketSamples: Int { bucketFrames * Self.hop }

    /// [128 × 257] Slaney-normalised mel filterbank, as dumped into the bundle assets.
    private let filterbank: [Float]
    /// [257 × 512] DFT tables.
    private let cosTable: [Float]
    private let sinTable: [Float]
    /// [512] Hann window, zero-padded and centred.
    private let window: [Float]

    public init(filterbankURL: URL, bucketFrames: Int = MelFrontend.defaultBucketFrames) throws {
        self.bucketFrames = bucketFrames
        let fb = try BinaryIO.readFloat32(filterbankURL)
        guard fb.count == Self.melBins * Self.freqBins else {
            throw GraphError.message(
                "mel filterbank: expected \(Self.melBins * Self.freqBins) floats, got \(fb.count)")
        }
        self.filterbank = fb

        // Hann(400, periodic: false) == 0.5 − 0.5·cos(2πn / 399), centred in the 512 frame.
        var w = [Float](repeating: 0, count: Self.nFFT)
        let offset = (Self.nFFT - Self.winLength) / 2      // 56
        for n in 0..<Self.winLength {
            w[offset + n] = Float(0.5 - 0.5 * cos(2.0 * Double.pi * Double(n) / Double(Self.winLength - 1)))
        }
        self.window = w

        var cosT = [Float](repeating: 0, count: Self.freqBins * Self.nFFT)
        var sinT = [Float](repeating: 0, count: Self.freqBins * Self.nFFT)
        for k in 0..<Self.freqBins {
            for n in 0..<Self.nFFT {
                let angle = 2.0 * Double.pi * Double(k) * Double(n) / Double(Self.nFFT)
                cosT[k * Self.nFFT + n] = Float(cos(angle))
                sinT[k * Self.nFFT + n] = Float(sin(angle))
            }
        }
        self.cosTable = cosT
        self.sinTable = sinT
    }

    /// Silence-pad (or truncate) a chunk to the encoder's fixed bucket.
    public func padToBucket(_ samples: [Float]) -> [Float] {
        if samples.count >= bucketSamples { return Array(samples[0..<bucketSamples]) }
        return samples + [Float](repeating: 0, count: bucketSamples - samples.count)
    }

    /// Waveform (16 kHz mono) → `[128 × bucketFrames]` normalised log-mel, row-major.
    public func melBucket(_ samples: [Float]) -> [Float] {
        let frames = bucketFrames
        let x = padToBucket(samples)

        // (2) pre-emphasis, then (3) constant-pad by nFFT/2 on each side.
        let pad = Self.nFFT / 2
        var padded = [Float](repeating: 0, count: x.count + 2 * pad)
        padded[pad] = x[0]
        for t in 1..<x.count { padded[pad + t] = x[t] - 0.97 * x[t - 1] }

        // (3) windowed frame matrix, laid out [nFFT × frames] so BLAS sees column-per-frame.
        var windowed = [Float](repeating: 0, count: Self.nFFT * frames)
        windowed.withUnsafeMutableBufferPointer { out in
            padded.withUnsafeBufferPointer { src in
                for n in 0..<Self.nFFT {
                    let w = window[n]
                    let row = n * frames
                    for t in 0..<frames { out[row + t] = src[t * Self.hop + n] * w }
                }
            }
        }

        // (4) one-sided DFT as two [257×512]·[512×frames] real matmuls.
        var re = [Float](repeating: 0, count: Self.freqBins * frames)
        var im = [Float](repeating: 0, count: Self.freqBins * frames)
        matmul(cosTable, windowed, into: &re, m: Self.freqBins, k: Self.nFFT, n: frames)
        matmul(sinTable, windowed, into: &im, m: Self.freqBins, k: Self.nFFT, n: frames)

        // power = re² + im² (the sign convention of `im` cancels here, so a plain DFT and an
        // FFT agree bit-for-bit in intent).
        var power = [Float](repeating: 0, count: re.count)
        vDSP.multiply(re, re, result: &power)
        var imsq = [Float](repeating: 0, count: im.count)
        vDSP.multiply(im, im, result: &imsq)
        vDSP.add(power, imsq, result: &power)

        // (5) mel projection + log.
        var mel = [Float](repeating: 0, count: Self.melBins * frames)
        matmul(filterbank, power, into: &mel, m: Self.melBins, k: Self.freqBins, n: frames)
        var guardValue = Self.logGuard
        vDSP_vsadd(mel, 1, &guardValue, &mel, 1, vDSP_Length(mel.count))
        var count = Int32(mel.count)
        vvlogf(&mel, mel, &count)

        // (6) per-mel-bin normalisation over all `frames` columns, unbiased variance.
        mel.withUnsafeMutableBufferPointer { buf in
            let n = Float(frames)
            for b in 0..<Self.melBins {
                let row = buf.baseAddress! + b * frames
                var mean: Float = 0
                vDSP_meanv(row, 1, &mean, vDSP_Length(frames))
                var sumsq: Float = 0
                vDSP_svesq(row, 1, &sumsq, vDSP_Length(frames))
                // unbiased variance = (Σx² − n·mean²) / (n − 1)
                let variance = max(0, (sumsq - n * mean * mean) / (n - 1))
                var scale = 1.0 / (variance.squareRoot() + Self.normEpsilon)
                var shift = -mean
                vDSP_vsadd(row, 1, &shift, row, 1, vDSP_Length(frames))
                vDSP_vsmul(row, 1, &scale, row, 1, vDSP_Length(frames))
            }
        }
        return mel
    }

    /// C[m×n] = A[m×k] · B[k×n], all row-major float32.
    ///
    /// The `numericCast`es are not decoration. CBLAS's index type is `Int` when Accelerate's
    /// ILP64 headers are selected (`ACCELERATE_LAPACK_ILP64`, which the macOS package sets) and
    /// `Int32` otherwise, and the iOS SDK ships only the 32-bit-index declaration. `numericCast`
    /// is a generic `BinaryInteger -> BinaryInteger` conversion whose *result* type is inferred
    /// from the parameter it is being passed to, so one spelling compiles against both headers.
    /// (A C++ reader can read it as a `static_cast` whose target type the compiler picks.)
    private func matmul(_ a: [Float], _ b: [Float], into c: inout [Float], m: Int, k: Int, n: Int) {
        a.withUnsafeBufferPointer { pa in
            b.withUnsafeBufferPointer { pb in
                c.withUnsafeMutableBufferPointer { pc in
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                numericCast(m), numericCast(n), numericCast(k),
                                1.0, pa.baseAddress!, numericCast(k),
                                pb.baseAddress!, numericCast(n),
                                0.0, pc.baseAddress!, numericCast(n))
                }
            }
        }
    }
}
