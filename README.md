# parakeet-swift

A macOS command-line speech-to-text tool. It runs NVIDIA's Parakeet automatic speech
recognition model end to end in Swift on Apple's Core AI runtime, with no Python in the
loop at run time. Point it at a 16 kHz mono WAV file and it prints a transcript,
optionally with word-level timestamps.

On an M4 Pro it transcribes a 66.5 minute audiobook chapter in about 14 seconds of wall
clock, model load included, which is roughly 280 times faster than real time. Word error
rate on LibriSpeech is 1.97 percent on test-clean and 4.29 percent on test-other.

How it got that fast is its own story: ten optimization rounds from a 41x baseline to
291x peak, a Neural Engine that lost to the GPU, and a bug that produced 406,000 garbage
tokens while every API reported success. It is written up in
[A 10-hour audiobook, transcribed on an iPhone in 5 minutes](https://rachuri.me/blog/parakeet-apple-silicon/).

## What the names mean

The vocabulary here is borrowed from four different places, so it is worth pinning down
once.

**Parakeet, TDT, 0.6B, v2.** The model is
[`nvidia/parakeet-tdt-0.6b-v2`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), an
English speech recognition model with 0.6 billion parameters. TDT stands for
Token-and-Duration Transducer, which is the decoding architecture: instead of emitting
one symbol per audio frame, it emits a symbol plus a jump length and skips ahead, which
is why the decode loop runs far fewer steps than there are frames. **The `v2` is NVIDIA's
name for that model release, not a version of this repository.** It appears in this
codebase for two reasons only: the model constants in `ParakeetConfig` (vocabulary size
1025, blank token id 1024) are those of that release, and the converted model files land
in a directory conventionally named `artifacts_v2/`. NVIDIA has since published a `v3`
release, which this host does not target.

**Core AI.** Apple's on-device inference runtime, introduced in the macOS 27 and iOS 27
SDKs. It is a *system framework*, so there is no dependency to fetch or vendor:
`Package.swift` declares `.linkedFramework("CoreAI")` and that is the whole story.

**Bundles, graphs, artifacts.** Core AI runs compiled computation graphs packaged as
`.aimodel` bundles. This host loads three of them: the **encoder**, which turns mel
features into acoustic embeddings and is by far the most expensive; the **predictor**, a
small two-layer LSTM over the tokens emitted so far; and the **joint**, which combines
the two and scores the next token and its duration. Those three plus a tokenizer and a
mel filterbank are what an artifacts directory holds. **This repository distributes no
weights and no bundles**, see [Setup](#setup).

**Gate, parity, RTF.** A *gate* is a pass-or-fail numerical comparison of one stage
against a stored reference tensor. *Parity* is the same idea over a whole corpus: run
every chunk and compare the emitted token stream, and the decoded text, against a stored
reference produced by a Python driver for the same bundles, itself checked against
NVIDIA's original PyTorch model. *RTF*, real-time factor, is audio duration divided by
processing time, so 280x means one hour of audio in about 13 seconds.

## Setup

### What you need

- An Apple silicon Mac. Everything below was measured on an M4 Pro.
- macOS 27.0 and the Xcode 27 beta toolchain, because `CoreAI` ships in that SDK. The
  exact combination that was used, verbatim: macOS 27.0 (26A5388g), Xcode 27.0 beta
  (27A5194q), Swift 6.4, target `arm64-apple-macos27.0`.
- About 1.1 GB of disk for the converted bundles, and roughly 1.7 GB of peak resident
  memory when transcribing a chapter-length file.
- The converted model bundles, which are not in this repository.

### Build

```bash
export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
xcrun swift build -c release
```

There is nothing to resolve and no other build step. The binary lands at
`.build/release/parakeet-swift`.

### Get the model

NVIDIA publishes the checkpoint and Apple's `coreai-build` toolchain compiles it into
`.aimodel` bundles. The conversion route this host expects is the recipe published by
[coreai-model-zoo](https://github.com/john-rocky/coreai-model-zoo); run that, then point
`PARAKEET_ARTIFACTS` at the directory it produces. Converted bundles are also published at
[rahulrachuri/parakeet-tdt-0.6b-v2-coreai](https://huggingface.co/rahulrachuri/parakeet-tdt-0.6b-v2-coreai);
an app can skip this step entirely and let the package fetch them, see
[Use as a package](#use-as-a-package).

The host resolves filenames inside that directory, so the layout has to match:

```
artifacts_v2/
  parakeet_encoder_float16_L2885.aimodel   the encoder, at the 2885-frame mel bucket
  parakeet_predict_float32.aimodel         the predictor
  parakeet_joint_float32.aimodel           the joint
  bundle_assets/
    tokenizer.json                         HuggingFace tokenizer, read directly
    mel_filters_128x257_f32.bin            the Slaney mel filterbank
  joint_head_w_f32.bin, joint_head_b_f32.bin      \  the joint and predictor weights as
  pred_embed_f32.bin, pred_proj_{w,b}_f32.bin      | flat float32, for the default
  pred_{w,b}_{ih,hh}{0,1}_f32.bin                 /  Accelerate decode path
```

The `L2885` in the encoder filename is the mel bucket length in frames, which the host
reads back out of the name so the front end pads to exactly the length the graph expects.
The flat float32 blobs in the last group are needed only for the default
`--decode-impl accel`; `--decode-impl coreai` runs entirely on the three graphs.

### Configure paths

Nothing is compiled in. Every path comes from the environment, with a relative fallback,
and can be overridden per run:

| variable | default | used by |
|---|---|---|
| `PARAKEET_ARTIFACTS` | `./artifacts_v2` | everything, also `--artifacts` |
| `PARAKEET_ASSETS` | the artifacts directory | tokenizer and filterbank, also `--assets` |
| `PARAKEET_BENCH` | `./bench` | `parity`, `bench` |
| `PARAKEET_CORPUS` | `corpus` | selects `bench/work/<name>/chunks.json` |
| `PARAKEET_GOLD_SET` | `gold` | selects `bench/results/<name>/tokens.json` |
| `PARAKEET_REFERENCE` | `./reference` | `gate-mel`, `gate-encoder` |

Everything the tool touches is read-only, and nothing is written outside the paths you
pass on the command line.

### What a clean clone can run

Transcription needs only the converted artifacts, so `transcribe`, `serve` and
`transcribe-list` work as soon as you have the model. So do the runtime probes `probe`,
`leak` and `contend`, which need a bundle and nothing else.

The verification commands `gate-mel`, `gate-encoder`, `parity` and `inspect` compare
against stored reference tensors and token streams covering a long-form audiobook corpus.
Those are large development inputs and are not published, so those four commands cannot
run from a clean clone. `tools/dump_reference.py` is what regenerates the `reference/`
tensors if you have the Python path set up.

## Quick start

```bash
export PARAKEET_ARTIFACTS=/path/to/artifacts_v2
.build/release/parakeet-swift transcribe recording_16k_mono.wav
```

The transcript goes to stdout. Add `--json out.json` to also get segments and word-level
timestamps in absolute seconds, in the shape Whisper emits:

```json
{"text": "...", "segments": [{"start": 0.0, "end": 4.8, "text": "...",
                              "words": [{"word": "the", "start": 0.32, "end": 0.44}]}]}
```

Input must be 16 kHz mono PCM16 WAV; convert anything else first, for example with
`ffmpeg -i in.m4a -ar 16000 -ac 1 -c:a pcm_s16le out.wav`.

For repeated transcription, `serve` avoids reloading the model. It reads one
`{"audio": "/path.wav"}` request per line on stdin and writes one response per line on
stdout in the `--json` shape. Stdout stays clean JSON-lines, all logging goes to stderr,
and a failing request is answered inline rather than being fatal.

### How a file becomes chunks

The encoder has one fixed input length, a bucket of 2885 mel frames, which is 28.85
seconds of audio. Longer input is cut into chunks and shorter input is padded with
silence to fill the bucket. The cut policy looks 24 to 28.8 seconds ahead and cuts at the
quietest 200 ms window in that band, which on narrated speech lands in a pause between
sentences. Chunks start from zero predictor state, so they are independent and decode
concurrently.

Two consequences are worth knowing before comparing output against another
implementation. Because the mel normalisation spans the whole padded bucket, the same
audio in a different-length window produces slightly different mel values, so the framing
is part of the result and not just a shape detail. And the greedy TDT loop can
occasionally decode an entire window to zero tokens, emitting blank at the maximum
duration on every frame and walking off the end. That is a property of the model at a
fixed framing rather than a fault in this host, since the PyTorch model collapses on the
same windows. `transcribe` and `serve` mitigate it by re-decoding a zero-token window as
two halves, recursing while the halves keep collapsing and stay longer than 4 seconds,
and logging each retry to stderr.

## Use as a package

`ParakeetKit` is a library product, so an app can depend on it and let it resolve its own
bundles — no checkout, no `PARAKEET_ARTIFACTS`, no manual download.

```swift
.package(url: "https://github.com/RahulRachuri/parakeet-swift", branch: "main")
// target dependency: .product(name: "ParakeetKit", package: "parakeet-swift")
```

Track `main`: `v0.1.0` predates this entry point.

```swift
import ParakeetKit

let engine = try await ParakeetEngine.fromHub()
let result = try await engine.transcribe(samples: samples)   // 16 kHz mono float
let text = try ParakeetTokenizer(tokenizerJSON: engine.paths.tokenizer).decode(result.tokens)
```

`fromHub` resolves [the bundle repository](https://huggingface.co/rahulrachuri/parakeet-tdt-0.6b-v2-coreai)
at a pinned, immutable revision, so a later push cannot change what a pinned consumer
receives. It is about 1.27 GB, nearly all of it the fp16 encoder, fetched once into
`~/Library/Caches/parakeet-swift/hub/` and reused; a warm start pays a cache check
rather than the download. Files published for provenance rather than loading — the conversion patch, the
gate transcript, the card — are evidence for a reader, not inputs to a run, and are
skipped.

Every LFS-stored payload is checked against the SHA-256 the Hub reports and moved into
place only once verified, so an interrupted download cannot be mistaken for a complete one.

Pass `progress:` for a closure to drive a loading UI — it is `@Sendable` and called from
URLSession's queue, so hop to the main actor before touching UI. `cacheDirectory:` puts the
cache elsewhere. The Accelerate decode path is unaffected and still opt-in; it reads its
blobs from the same directory, reachable as `engine.paths.joint.deletingLastPathComponent()`.

## Commands

Transcription:

```
transcribe <wav> [--json PATH]        one WAV to a transcript, plus optional timestamp JSON
serve                                 JSON-lines worker, model loaded once
transcribe-list --manifest <json> --out <jsonl> [--limit N]
                                      batch-transcribe a labelled utterance list such as
                                      LibriSpeech: one utterance per chunk, no voice
                                      activity detection, always serial, and a failing
                                      utterance is recorded with an "error" field rather
                                      than aborting the run and quietly shrinking the
                                      scored denominator
```

Verification. All four need the reference data described above:

```
gate-mel                              the Swift mel front end vs the Python reference
gate-encoder                          one Swift encoder pass vs the Python one, both mel sources
parity [--limit N] [--ref-mel-dir D]  every chunk of the corpus vs the stored token streams
       [--manifest PATH] [--gold PATH]
inspect --chunk N [--ref-mel]         the anatomy of a single divergence
        [--repeat R]
```

Measurement and runtime probing, which need only the artifacts:

```
bench [--limit N] [--manifest PATH]   RTF plus a per-stage time split
      [--no-decode]                   retire chunks undecoded, to measure the mel and
                                      encoder supply rate on its own. Only honoured with
                                      --decode-workers > 1; check the reported tokens is 0
probe <bundle> [--unit u]             print one graph's input and output signature
      [--default-options]             load with no compute-unit preference, which reproduces
                                      the runtime's own choice and its load failure
leak [--calls N] [--unit gpu|cpu|ane] hammer one bundle and watch resident memory
contend [--threads N] [--calls N]     N threads on N separately loaded copies of one graph,
        [--graph joint|predictor]     to ask whether the runtime executes them concurrently.
        [--unit cpu|gpu]              It does: joint 8238 to 16569 calls/s, 1 to 4 threads
```

## Configuration

Which bundles to load, shared by `parity`, `bench` and `inspect`:

```
--artifacts DIR            where the three graphs live (default: ./artifacts_v2)
--assets DIR               tokenizer and filterbank, if not beside the graphs
--graph-ext EXT            "aimodel", the just-in-time specialised source bundle and the
                           default, or e.g. "h16s.aimodelc" for an ahead-of-time build
--encoder-path PATH        override one graph, to mix build kinds in a single run
--predictor-path PATH
--joint-path PATH
--mel-frames L             load a differently bucketed encoder export and retarget the mel
                           front end to match (default 2885). This changes every mel value,
                           because normalisation spans the padded window, so it invalidates
                           any stored token reference: 21 of 152 chunks flipped at L=2880
--batched-joint PATH       the re-exported joint taking enc_frames [K,640]
--batch-k K                must match that bundle's static shape
```

How it runs:

```
--plan e/p/j               compute unit per graph, each gpu, cpu or ane, e.g. gpu/cpu/cpu.
                           Maps to SpecializationOptions(preferredComputeUnitKind:), which
                           is a preference and not the strict cpuOnly parity mode
--stream-depth D           submit encoders through ComputeStream, queued D deep
                           (default 2 for transcribe and serve, 0 elsewhere)
--decode-workers N         concurrent decode workers, each with its own predictor and joint
                           replica (default 4 for transcribe and serve, 1 elsewhere)
--pipeline                 one-deep overlap of decode with the next chunk's mel and encoder,
                           an alternative to --stream-depth on the serial submission path
--decode-impl IMPL         coreai | accel-joint | accel (default: accel)
--decode-qos Q             default | utility | background (default: default)
--lookahead K              speculative joint batching, a refuted idea kept as a lever
```

Three of those have earned an explanation.

**`--stream-depth` and `--decode-workers`** select the driver shape. `transcribe` and
`serve` default to depth 2 with 4 workers, which is the shape every performance number
below was measured on; `--stream-depth 0 --decode-workers 1` is the plain serial path and
produces byte-identical output. Encoder submissions stay serialised through a single
producer task whatever the shape, because two overlapping invocations of one
`InferenceFunction` corrupt each other, see [Runtime notes](#runtime-notes).

**`--decode-impl`** picks the per-token loop implementation. `coreai` runs the joint and
predictor as compiled graphs. The default, `accel`, runs them as direct Accelerate
arithmetic instead, with no graph dispatch in the hot path at all: same weights, and
byte-identical output on the 152-chunk parity gate, on a full chapter compared by hash,
on the blank-collapse fixture, and on LibriSpeech test-clean plus test-other at 5559
utterances. It uses 44 percent less decode-loop CPU, 14.0 against 24.9 CPU-seconds per
chapter, with system time at 3.1 against 9.0 seconds because graph dispatch is
syscall-heavy, and costs 22 MB of resident memory for the float32 weight blobs.
`accel-joint` converts the joint only and keeps the predictor on its graph.

**`--decode-qos`** requests a task priority for the decode workers and is known to be
inert. A quiet-machine interleaved A/B measured no difference in wall time (14.86 against
14.78 s), CPU, or per-cluster residency, and a priority probe showed why: the task group's
`waitForAll()` escalates every worker from the requested priority back to the parent's
before the first chunk, because Swift propagates priority through any await by design.
Expressing "run this on the efficiency cores while I wait" needs thread-level QoS, not
task priority. The flag stays as an honest A/B lever.

## Results

Measured on an M4 Pro. The long-form corpus is an audiobook chapter: 66.5 minutes of
narrated speech, 3989.9 seconds, cut by the policy above into 152 chunks.

### Accuracy

| benchmark | word error rate |
|---|---|
| LibriSpeech test-clean | 1.97 % |
| LibriSpeech test-other | 4.29 % |

Measured over both full splits, 5559 utterances, on the serial driver. Every decode
implementation inherits these by byte-identity rather than by re-measurement.

The encoder emits its full frame count whatever the input length, and the decode loop has
no length mask, so a 2 second utterance is decoded across 27 seconds of silence padding.
`--mask-padding` stops the loop at the last frame carrying audio. It cuts insertions
materially, 136 down to 85 on test-clean, but moves word error rate only to 1.91 and 4.23
percent, and leaves substitutions, deletions and all 33 all-blank utterances untouched.
So tail hallucination is the minor term; the dominant short-utterance cost is upstream, in
the mel normalisation spanning the padded window. It is off by default.

### Speed

The default shape, `--stream-depth 2 --decode-workers 4 --decode-impl accel`:

| measurement | value |
|---|---|
| `transcribe` wall clock, model load included | 14.1 s, about 280x real time |
| process CPU for the same run | 14.0 CPU-s |
| mel and encoder supply rate alone (`bench --no-decode`) | 13.42 s, 297.3x |
| peak resident memory | 1679 MB |

The last two lines are the whole story: the encoder floor is 13.42 seconds and the
finished transcript takes 14.1, so decode is essentially free and the encoder is the
entire remaining budget. Peak memory is flat in file length, because both the WAV read and
the chunker's running sums go through the file's memory map on demand rather than
materialising the audio.

Under ambient system load the parallel decode shape is also the robust one: the serial
path degrades to 219x while two workers hold around 260x.

As a like-for-like yardstick, FluidAudio's published ~145x is a short-form number. On this
same M4 Pro and the same 340 minute corpus, their CoreML int8-on-ANE build does 452x, so
this host is meaningfully behind, and the entire gap is the encoder. This port stays at
full precision by policy: no quantization, fp16 encoder, fp32 predictor and joint as
exported.

### Numerical gates

| gate | result |
|---|---|
| mel front end vs Python reference | `cos = 1.000000000`, per-bin cosine min 1.000000000 |
| encoder, Python mel in | `cos = 1.000000000`, `maxabs = 0.0000`, bit-identical |
| encoder, Swift mel in | `cos 0.999998`, the same band as the conversion's own export gate |

### Parity over the full corpus

| configuration | chunks token-exact | tokens (aligned) | text-exact |
|---|---|---|---|
| Swift host, Python reference mel | 152/152 | 20376/20376 (100.0000 %) | 152/152 |
| Swift host, Swift mel (shipping) | 151/152 | 20374/20376 (99.9902 %) | 151/152 |

Given identical mel input, this host is exactly equivalent to the Python engine, down to
character-for-character text. The single divergence on the shipping path, chunk 45 and
deterministic, is one word-initial casing flip caused by the last unit in the last place
of the front end: float32 BLAS here against float64 torch there flips one fp16 near-tie on
the first frame. That is two tokens in 20376, over 66 minutes.

### Runtime notes

Three things about the Core AI runtime that cost time to find.

**One invocation of a function at a time.** Two overlapping runs of the same
`InferenceFunction` interleave their output buffers and return garbage, silently. Launching
the next chunk's encoder before joining the current one produced 406130 tokens instead of
20375, with no error reported anywhere. Every driver here keeps encoder submission on a
single task for this reason.

**`run()` leaves the GPU idle between invocations.** It is async and allows only one
invocation's command buffers to exist at a time, so the GPU drains between them, costing
roughly 19 percent of encoder time. The synchronous `encode(inputs:to:)` path appends
command buffers to an explicit Metal queue and returns futures, which takes the encoder
pass from 16.45 to 12.16 seconds over 152 buckets, 108.2 down to 80.0 ms each. Building
those command buffers costs about 30 ms of synchronous CPU per chunk, so it belongs in a
dedicated producer task ahead of the consumer rather than inline.

**Independent functions do run concurrently.** Separately loaded copies of one bundle
execute in parallel: the joint goes from 8238 to 16569 calls per second between 1 and 4
threads, the predictor from 2241 to 6113. `contend` is the probe.

**The Python bindings leak IOSurfaces; the runtime does not.** Same bundle, same GPU
compute unit, same call shape: the Python bindings abort somewhere between call 8000 and
9000, while the Swift `CoreAI` framework survives 120000 calls with flat resident memory,
25.0 to 25.7 MB, and flat throughput.

### Compute unit placement

Where the predictor and joint run matters enormously when they run as graphs, which is
what `--plan` selects and what the numbers below isolate. Measured on the serial driver
with `--decode-impl coreai`, encoder pinned to the GPU throughout.

| plan (enc/pred/joint) | mel | encoder | TDT loop | compute | RTF excl. load | joint+pred per call |
|---|---|---|---|---|---|---|
| gpu / gpu / gpu | 0.39 s | 16.36 s | 31.97 s | 48.77 s | 81.8x | 0.739 ms |
| gpu / ane / ane | 0.39 s | 16.31 s | 27.12 s | 43.86 s | 91.0x | 0.627 ms |
| gpu / cpu / gpu | 0.39 s | 16.31 s | 25.61 s | 42.35 s | 94.2x | 0.592 ms |
| gpu / gpu / cpu | 0.39 s | 16.20 s | 20.66 s | 37.29 s | 107.0x | 0.478 ms |
| gpu / cpu / cpu | 0.38 s | 16.37 s | 12.77 s | 29.56 s | 135.0x | 0.295 ms |

Moving the predictor and joint off the GPU is a 1.65x end-to-end win. They are tiny fp32
graphs invoked about 284 times per chunk, and on the GPU each call pays a dispatch cost
that dwarfs the arithmetic; the Neural Engine sits in between and is not competitive. The
default `--decode-impl accel` takes the argument further by removing those two graphs from
the runtime altogether, which is why `--plan` only affects the encoder in the default
configuration. The mel front end is 1.3 percent of compute throughout, so the BLAS
discrete Fourier transform was never a bottleneck.

### Optimisations that did not work

- **Speculative joint lookahead loses at every K.** 89.7 percent of joint calls emit a
  token, so a prefetched window is invalidated almost every time it fills. K=2 already
  drops gpu/cpu/cpu from 136.5x to 134.0x, and it gets worse from there.
- **A batched joint loses too.** Re-exporting the joint to take `enc_frames [K,640]` in
  one dispatch gated token-exact and then measured slower, 140.8x down to 121.2x at K=4.
  The emission rate means the dispatch count only falls 7 percent while every call does K
  times the arithmetic.
- **Ahead-of-time compilation is not worth it on Mac.** Steady-state encoder time is
  identical to just-in-time, the only win is the once-per-install first-ever load, 4.74
  down to 0.48 seconds, warm just-in-time loads are actually faster at 0.24 seconds, and
  the compiled encoder doubles on-disk size from 1.1 to 2.3 GB.
- **Low-QoS decode workers save nothing**, because task priority cannot express what was
  wanted. See `--decode-qos` above.
- **The Neural Engine is unreachable at full precision through this toolchain.** The
  `.neuralEngine` preference specialises for 26.8 seconds and then runs 2.4x slower than
  the GPU, and `coreai-build` 3600.67.5.8.1 emits no ANE delegate at all, for any graph,
  on either platform: even a trivial Linear and ReLU pair compiles to MPSGraph only.
  Reaching the ANE would additionally require re-authoring the encoder on the iOS export
  track, with BC1S layout, 1x1 convolutions, per-head attention and fp16 throughout. A
  re-authored export does reach it, and then loses to the GPU at fp16 on an M4 Pro anyway,
  a result an independent CoreML control confirms is the silicon rather than the compiler.

## File map

```
Package.swift                          swift-tools 6.0, macOS 27, links the CoreAI system framework
Sources/ParakeetKit/
  CoreAIGraph.swift                    load one .aimodel with an explicit compute-unit
                                       preference, run or stream it, flatten outputs
  MelFrontend.swift                    the log-mel front end: silence-pad to the bucket,
                                       pre-emphasis, centred Hann in 512, BLAS DFT,
                                       Slaney mel, log, per-bin unbiased normalisation
  Tokenizer.swift                      Metaspace/BPE decode straight off tokenizer.json
  ParakeetEngine.swift                 artifact paths, compute plan, the TDT greedy loop
  HubStore.swift                       pinned-revision Hub resolution: list, fetch,
                                       digest-check against the Hub's SHA-256, cache
  ParakeetEngine+Hub.swift             ParakeetEngine.fromHub(), the package entry point
  AccelJoint.swift, AccelPredictor.swift  the graph-free decode hot path (SGEMV/vDSP/vForce)
  Chunker.swift                        the audio cut policy
  AudioInput.swift                     chunks.json manifest reader, mmap'd PCM16 wav reader
  BinaryIO.swift                       flat float32 I/O and the cosine/LCS gate metrics
  Timestamps.swift                     word-level timestamp assembly on the 80 ms frame grid
Sources/parakeet-swift/main.swift      the CLI and the drivers
tools/dump_reference.py                regenerates reference/ from the Python path
tools/export_joint_batched.py          exports the batched-joint variant used to refute it
reference/                             mel/encoder/wav tensors for a few chunks (gitignored)
```

`ParakeetKit` also builds for iOS.

## Limits

The encoder is the entire remaining budget, and at full precision on the GPU there is no
lever left in the host: everything else has been measured down to noise. Closing the gap
to a quantized int8-on-ANE build would mean giving up the no-quantization policy, and
reaching the Neural Engine at all is blocked on the toolchain emitting an ANE delegate.
The remaining speedups are runtime-side, so re-measuring on each new beta is worthwhile.

## Licence and attribution

This host, meaning everything under `Sources/` and `tools/`, is licensed under the Apache
Licence, Version 2.0. See [LICENSE](LICENSE).

`Sources/ParakeetKit/VocabularyRescoring/` is adapted from
[FluidAudio](https://github.com/FluidInference/FluidAudio) (Apache-2.0), which is also
where the custom-vocabulary design comes from. See [NOTICE](NOTICE) for the upstream
revision and the per-file details.

The models are not covered by that licence. The transcription model is NVIDIA's
[`parakeet-tdt-0.6b-v2`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) and the
custom-vocabulary verifier is NVIDIA's
[`parakeet-tdt_ctc-110m`](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m) CTC head.
Both are released under CC-BY-4.0, and any weights or converted bundles you run through
this host stay under that licence and carry its attribution requirement. No weights are
distributed in this repository.

The conversion route from NVIDIA's checkpoint to Core AI bundles follows the recipe
published by [coreai-model-zoo](https://github.com/john-rocky/coreai-model-zoo) (BSD
3-Clause, © 2026 Daisuke Majima). No source from it is vendored or imported here.
