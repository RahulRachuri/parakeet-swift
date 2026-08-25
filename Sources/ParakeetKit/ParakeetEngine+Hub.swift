//
// ParakeetEngine+Hub.swift — the package-style entry point.
//
// `ParakeetEngine(paths:)` takes an `ArtifactPaths` and asks no questions about where the
// files came from; that is the right seam for a CLI run against `PARAKEET_ARTIFACTS`.
// This adds the other way in, for an app that has no artifacts directory: call it, get an
// engine. Everything downstream is unchanged.
//

import Foundation

extension ParakeetEngine {

    /// Resolve the pinned bundle repository and build an engine from it.
    ///
    ///     let engine = try await ParakeetEngine.fromHub()
    ///     let result = try await engine.transcribe(samples: samples)
    ///
    /// About 1.27 GB, nearly all of it the fp16 encoder, fetched once into the caches
    /// directory and reused. Provenance and conversion notes in the repository are
    /// skipped — only what the host actually reads is fetched, plus `config.json`, which
    /// names the repository the bundles came from.
    ///
    /// The Accelerate decode path is opt-in and unaffected: it reads its blobs from the
    /// same directory, reachable as `engine.paths.joint.deletingLastPathComponent()`.
    public static func fromHub(
        plan: ComputePlan = ComputePlan(),
        config: ParakeetConfig = ParakeetConfig(),
        bucketFrames: Int = MelFrontend.defaultBucketFrames,
        cacheDirectory: URL? = nil,
        progress: (@Sendable (HubProgress) -> Void)? = nil
    ) async throws -> ParakeetEngine {
        let directory = try await HubStore.ensure(
            .artifacts,
            where: { !isProvenanceOnly($0) },
            cacheDirectory: cacheDirectory,
            progress: progress)

        let paths = ArtifactPaths(artifactsDirectory: directory, bucketFrames: bucketFrames)
        return try await ParakeetEngine(paths: paths, plan: plan, config: config)
    }

    /// Files published for provenance and review rather than for loading: the conversion
    /// patch, the gate transcript, the card. Skipping them is about honesty as much as
    /// bytes — they are evidence for a reader, not inputs to a run.
    static func isProvenanceOnly(_ path: String) -> Bool {
        path == "README.md" || path == ".gitattributes"
            || path == "gates_v2.txt" || path.hasSuffix(".patch")
    }
}
