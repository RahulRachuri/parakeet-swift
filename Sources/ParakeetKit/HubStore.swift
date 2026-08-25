//
// HubStore.swift — resolving a pinned Hugging Face repository into a local directory.
//
// This repository distributes no weights and no bundles; it downloads them. Unlike the
// pocket-tts host, everything parakeet needs lives in one repository — the three graph
// bundles, the tokenizer and mel filterbank under `bundle_assets/`, and the plain-float
// blobs the Accelerate predictor and joint read — so a single `ensure` produces a
// directory that `ArtifactPaths(artifactsDirectory:)` can be pointed straight at.
//
// The repository is pinned to an immutable revision rather than a branch: a later push
// cannot change what a pinned consumer receives. The Hub reports a SHA-256 for every
// LFS-stored file, so each payload is checked against it on the way in.
//
// No third-party dependencies: URLSession for transport, CryptoKit for the digest.
//

import Foundation
import CryptoKit

/// A Hugging Face repository pinned to one immutable revision.
public struct HubRepo: Sendable {
    public let id: String
    public let revision: String

    public init(id: String, revision: String) {
        self.id = id
        self.revision = revision
    }

    /// The Core AI bundles and host-side blobs this port runs.
    public static let artifacts = HubRepo(
        id: "rahulrachuri/parakeet-tdt-0.6b-v2-coreai",
        revision: "6fa37b9a2ef99ff67e5cf868b9f2ab5d9540aa60")
}

/// One file in a repository listing, as the Hub reports it.
public struct HubFile: Sendable {
    public let path: String
    public let size: Int
    /// SHA-256 of the content, present for LFS-stored files — which is every payload
    /// large enough to be worth verifying. Small files are plain git blobs and carry a
    /// SHA-1 of the blob instead, which this does not check.
    public let sha256: String?
}

/// Progress for one `ensure` call: bytes so far and the total it intends to fetch.
public struct HubProgress: Sendable {
    public let path: String
    public let completedBytes: Int64
    public let totalBytes: Int64
    public var fraction: Double {
        totalBytes > 0 ? Double(completedBytes) / Double(totalBytes) : 0
    }
}

public enum HubStore {

    // MARK: - Cache location

    /// `~/Library/Caches/parakeet-swift/hub/<repo>/<revision>/`. On iOS this is the app's
    /// own caches directory, which is the correct home for re-downloadable data: the
    /// system may evict it, and `ensure` will simply fetch it again.
    ///
    /// The `hub/` component is not decoration. On macOS the Core AI runtime keeps its own
    /// compiled-bundle cache under a sibling path, so resolved repositories are kept in
    /// their own subtree rather than interleaved with a cache this package does not own.
    public static func defaultCacheDirectory() throws -> URL {
        let caches = try FileManager.default.url(
            for: .cachesDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        return caches.appendingPathComponent("parakeet-swift", isDirectory: true)
            .appendingPathComponent("hub", isDirectory: true)
    }

    static func directory(for repo: HubRepo, under root: URL) -> URL {
        root.appendingPathComponent(repo.id.replacingOccurrences(of: "/", with: "--"),
                                    isDirectory: true)
            .appendingPathComponent(repo.revision, isDirectory: true)
    }

    // MARK: - Listing

    /// Every file in the repository at the pinned revision, with size and digest.
    ///
    /// One request, and it is the ground truth: unlike a manifest committed to the
    /// repository, a listing cannot drift from what is actually there.
    public static func list(_ repo: HubRepo) async throws -> [HubFile] {
        var comps = URLComponents(
            string: "https://huggingface.co/api/models/\(repo.id)/tree/\(repo.revision)")!
        comps.queryItems = [URLQueryItem(name: "recursive", value: "true")]
        guard let url = comps.url else {
            throw GraphError.message("could not form a listing URL for \(repo.id)")
        }

        let (data, response) = try await URLSession.shared.data(from: url)
        guard let http = response as? HTTPURLResponse else {
            throw GraphError.message("no HTTP response listing \(repo.id)")
        }
        guard http.statusCode == 200 else {
            throw GraphError.message("listing \(repo.id) at \(repo.revision) failed: HTTP \(http.statusCode)")
        }

        guard let raw = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw GraphError.message("unexpected listing shape for \(repo.id)")
        }
        return raw.compactMap { entry in
            guard (entry["type"] as? String) == "file",
                  let path = entry["path"] as? String else { return nil }
            let size = (entry["size"] as? Int) ?? 0
            let oid = (entry["lfs"] as? [String: Any])?["oid"] as? String
            return HubFile(path: path, size: size, sha256: oid)
        }
    }

    // MARK: - Fetching

    /// Download every file matching `predicate` and return the directory holding them.
    ///
    /// The returned directory mirrors the repository layout, so it can be handed
    /// straight to `TTSPipeline(assetsDir:)` or used to build a `WeightsLayout`.
    ///
    /// Already-present files of the right size are skipped, so a second call is cheap
    /// and an interrupted one resumes at file granularity. Downloads are sequential on
    /// purpose: in both repositories a single bundle dominates the total, so
    /// parallelism would buy little and makes progress reporting a lie.
    @discardableResult
    public static func ensure(
        _ repo: HubRepo,
        where predicate: (String) -> Bool = { _ in true },
        cacheDirectory: URL? = nil,
        progress: (@Sendable (HubProgress) -> Void)? = nil
    ) async throws -> URL {
        let root = try cacheDirectory ?? defaultCacheDirectory()
        let dest = directory(for: repo, under: root)
        try FileManager.default.createDirectory(at: dest, withIntermediateDirectories: true)

        let wanted = try await list(repo).filter { predicate($0.path) }
        guard !wanted.isEmpty else {
            throw GraphError.message("nothing in \(repo.id) at \(repo.revision) matched the requested files")
        }

        let total = wanted.reduce(Int64(0)) { $0 + Int64($1.size) }
        var done: Int64 = 0

        for file in wanted {
            let local = dest.appendingPathComponent(file.path)
            if isPresent(local, size: file.size) {
                done += Int64(file.size)
                progress?(HubProgress(path: file.path, completedBytes: done, totalBytes: total))
                continue
            }
            try FileManager.default.createDirectory(
                at: local.deletingLastPathComponent(), withIntermediateDirectories: true)

            let base = done
            try await download(file, from: repo, to: local) { written in
                progress?(HubProgress(path: file.path,
                                      completedBytes: base + written,
                                      totalBytes: total))
            }
            done += Int64(file.size)
        }
        return dest
    }

    private static func isPresent(_ url: URL, size: Int) -> Bool {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              let have = attrs[.size] as? Int else { return false }
        // Size alone: the digest was checked when the file was written, and rehashing a
        // gigabyte on every launch to re-answer a settled question is not worth it.
        return size == 0 || have == size
    }

    private static func download(
        _ file: HubFile,
        from repo: HubRepo,
        to destination: URL,
        onProgress: @escaping @Sendable (Int64) -> Void
    ) async throws {
        let encoded = file.path
            .addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? file.path
        guard let url = URL(
            string: "https://huggingface.co/\(repo.id)/resolve/\(repo.revision)/\(encoded)") else {
            throw GraphError.message("could not form a download URL for \(file.path)")
        }

        // Staged next to the destination rather than in the system temp directory: a
        // move across volumes is a copy, and these files are large.
        let staging = destination.appendingPathExtension("part")
        try await BundleDownload(staging: staging, onProgress: onProgress).run(url, describing: file.path)

        if let expected = file.sha256 {
            let actual = try sha256(of: staging)
            guard actual == expected.lowercased() else {
                try? FileManager.default.removeItem(at: staging)
                throw GraphError.message(
                    "\(file.path) does not match the digest the Hub reports for \(repo.id)@\(repo.revision) "
                    + "(expected \(expected.prefix(16))…, got \(actual.prefix(16))…)")
            }
        }

        // Move into place only once verified, so an interrupted or corrupt download can
        // never be mistaken for a complete one by the size check above.
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: staging, to: destination)
    }

    /// Streaming digest — these files do not fit in memory.
    static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 4 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

/// One download, on its own session.
///
/// `URLSession.shared` silently ignores task-specific delegates, so progress reported
/// through it never arrives — the download succeeds and the caller sees nothing. A
/// session that owns its delegate is the only way to get byte counts out, so each
/// download gets one and invalidates it afterwards to break the delegate retain cycle.
///
/// Unchecked because the delegate callbacks arrive on the session's queue and the
/// mutable state is guarded by `lock`.
private final class BundleDownload: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let staging: URL
    private let onProgress: @Sendable (Int64) -> Void
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Error>?
    private var moveFailure: Error?
    private var label = ""

    init(staging: URL, onProgress: @escaping @Sendable (Int64) -> Void) {
        self.staging = staging
        self.onProgress = onProgress
    }

    func run(_ url: URL, describing path: String) async throws {
        label = path
        let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
            lock.lock(); continuation = c; lock.unlock()
            session.downloadTask(with: url).resume()
        }
    }

    private func finish(_ result: Result<Void, Error>) {
        lock.lock()
        let c = continuation
        continuation = nil
        lock.unlock()
        c?.resume(with: result)
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        onProgress(totalBytesWritten)
    }

    /// The temporary file is deleted as soon as this returns, so the move is synchronous.
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        do {
            try? FileManager.default.removeItem(at: staging)
            try FileManager.default.moveItem(at: location, to: staging)
        } catch {
            lock.lock(); moveFailure = error; lock.unlock()
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error { return finish(.failure(error)) }
        if let http = task.response as? HTTPURLResponse, http.statusCode != 200 {
            try? FileManager.default.removeItem(at: staging)
            return finish(.failure(GraphError.message("downloading \(label) failed: HTTP \(http.statusCode)")))
        }
        lock.lock(); let failure = moveFailure; lock.unlock()
        finish(failure.map { .failure($0) } ?? .success(()))
    }
}
