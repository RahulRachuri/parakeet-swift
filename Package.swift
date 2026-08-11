// swift-tools-version: 6.0
//
// parakeet-swift: a native Swift host for the Parakeet-TDT-0.6B-v2 Core AI port.
//
// Build (Xcode 27 beta toolchain, macOS 27 SDK):
//     export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
//     swift build -c release
//
// `CoreAI` is a *system* framework in the macOS 27 SDK, so there is no package
// dependency to resolve; we just tell the linker to link it.
import PackageDescription

let package = Package(
    name: "parakeet-swift",
    platforms: [.macOS("27.0")],
    targets: [
        .target(
            name: "ParakeetKit",
            swiftSettings: [
                // Opt into Accelerate's current (ILP64-capable) CBLAS headers; without this
                // `cblas_sgemm` is flagged deprecated.
                .unsafeFlags(["-Xcc", "-DACCELERATE_NEW_LAPACK=1", "-Xcc", "-DACCELERATE_LAPACK_ILP64=1"])
            ],
            linkerSettings: [.linkedFramework("CoreAI")]
        ),
        .executableTarget(
            name: "parakeet-swift",
            dependencies: ["ParakeetKit"],
            linkerSettings: [.linkedFramework("CoreAI")]
        ),
    ]
)
