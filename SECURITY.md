# Security

This repository is a native Swift port of Parakeet-TDT: the mel front end, the greedy TDT
decode loop, an Accelerate decode path that runs the predictor and joint outside the Core
AI graphs, and the long-form handling around them. It distributes no weights and no
bundles, and the conversion recipe that produces the bundles is published separately.

So there are two questions worth asking, not one. Whether this code decodes correctly,
and whether the artifacts it loaded are the ones that were gated. This page answers both,
including where the answer is no.

## Reporting

Open a [security advisory](../../security/advisories/new) for anything you would rather
not say in public, such as a bundle that behaves unlike its source model, or a repository
that looks tampered with. Ordinary bugs belong in [issues](../../issues). This is a
personal project, so expect a reply in days rather than hours.

## What the integrity story actually is

**A pinned revision, not a signature.** Two published repositories matter here:

- [rahulrachuri/parakeet-tdt-0.6b-v2-coreai](https://huggingface.co/rahulrachuri/parakeet-tdt-0.6b-v2-coreai)
  holds the Core AI bundles this host runs, along with the tokenizer assets, the mel
  filterbank, and the gate transcript they were accepted on.
- [rahulrachuri/parakeet-tdt-0.6b-v2](https://huggingface.co/rahulrachuri/parakeet-tdt-0.6b-v2)
  holds an HF-layout conversion of NVIDIA's checkpoint, which exists because NVIDIA
  publishes v2 as a `.nemo` only. It is the reference the bundles were gated against, not
  something this host loads at runtime.

Pin a revision rather than tracking a branch. A later push cannot then change what you
receive.

**What is not done.** The bundles are not code-signed, and there is no checksum manifest
beyond what Hugging Face itself stores. Conversion is not byte-deterministic, so a
checksum of your own rebuild will not match the published one even when the rebuild is
correct. Integrity rests on the revision pin and on Hugging Face's storage, not on a
signature you can verify offline.

**A `.aimodel` bundle is data that a runtime executes.** Treat one from any source the way
you would treat a binary dependency.

**The flat weight blobs deserve separate attention.** The default decode path,
`--decode-impl accel`, does not run the predictor and joint as graphs. It reads their
weights as raw float32 arrays from the artifacts directory: `joint_head_w_f32.bin`,
`pred_embed_f32.bin`, `pred_proj_{w,b}_f32.bin` and the LSTM gate blobs. These are the
least self-describing artifacts in the set. They carry no shapes, no metadata and no
provenance of their own beyond `pred_manifest.json` and `joint_head_manifest.json`
alongside them, and the graph gates in `gates_v2.txt` do not cover them, because that path
does not use the graphs. Swapping a blob is the cheapest way to change what this host
outputs without touching a bundle. Run `--decode-impl coreai` to stay entirely on the
three gated graphs, and compare the two paths if you want the blobs checked against
something.

## Checking the bundles yourself

The gate transcript is published beside the bundles as `gates_v2.txt`. It records what
each graph was checked against and what it scored, including the end-to-end result of
82/82 tokens exact against the Hugging Face `ParakeetForTDT` reference.

The exporters and gates that produce the bundles live in the
[Core AI model zoo](https://github.com/john-rocky/coreai-model-zoo) under
`conversion/parakeet/`, with the exact configuration recorded in
`models/parakeet-v2/recipe.toml`. Re-running them reproduces the bundles and re-derives
every number in the transcript.

Those gates check the bundles, not this host. To check the host itself,
`tools/dump_reference.py` dumps reference tensors so this Swift implementation can be
compared against the Python one stage by stage, which is the check that catches a decode
bug rather than a tampered artifact.

One gate is worth singling out, because it is the failure most likely to look fine. Mel
padding is load-bearing: feeding per-clip mel with zero padding produces 85 tokens against
82 gold and agrees on only 22 of them, while the oracle-style path is exact. A host that
gets the padding convention wrong does not crash. It returns plausible, wrong text.

## Model weights are upstream's

The model is NVIDIA's
[`parakeet-tdt-0.6b-v2`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), CC-BY-4.0.
What the model says, what it was trained on, and its licence are upstream's concern.

## Scope

In scope: this host, and the published bundles and their provenance. Out of scope: the
upstream model's content, Hugging Face's own storage, and Apple's Core AI runtime. If you
are shipping something you have to support, mirror the bundles you depend on into storage
you control rather than fetching a personal Hugging Face namespace at runtime, and
re-verify after any OS or toolchain bump. The `coreai-core` wheel is OS-coupled, and a
beta bump has already invalidated previously exported bundles once.
