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
    // `ParakeetKit` is exported so an app can depend on this package directly and resolve
    // its own bundles through `ParakeetEngine.fromHub()`, rather than building the CLI and
    // pointing `PARAKEET_ARTIFACTS` at a directory it downloaded by hand.
    products: [
        .library(name: "ParakeetKit", targets: ["ParakeetKit"]),
        .executable(name: "parakeet-swift", targets: ["parakeet-swift"]),
    ],
    targets: [
        // Accelerate's current CBLAS interface is selected by two preprocessor defines, and
        // they have to reach the headers as they are parsed. Passed as raw compiler flags
        // they land in SwiftPM's `unsafeFlags`, and a package carrying those cannot be
        // depended on by version -- only by branch or path. Declared as `.define` on a C
        // target they are a setting SwiftPM recognises, and the restriction does not apply.
        // That is the whole reason this target exists; see its header.
        .target(
            name: "CParakeetBLAS",
            cSettings: [
                .define("ACCELERATE_NEW_LAPACK", to: "1"),
                .define("ACCELERATE_LAPACK_ILP64", to: "1"),
            ],
            linkerSettings: [.linkedFramework("Accelerate")]
        ),
        .target(
            name: "ParakeetKit",
            dependencies: ["CParakeetBLAS"],
            linkerSettings: [.linkedFramework("CoreAI")]
        ),
        .executableTarget(
            name: "parakeet-swift",
            dependencies: ["ParakeetKit"],
            linkerSettings: [.linkedFramework("CoreAI")]
        ),
    ]
)
