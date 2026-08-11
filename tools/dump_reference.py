"""Dump reference tensors from the VALIDATED Python Core AI path, for the Swift gates.

Read-only against its inputs; writes only into the reference/ directory.

Needs a Python environment with coreai.runtime + torch + librosa + soundfile, plus
PARAKEET_BENCH and PARAKEET_ARTIFACTS pointing at the corpus harness (which
supplies `bench.frontend`) and the converted Core AI bundles. Neither ships here.

    PARAKEET_BENCH=… PARAKEET_ARTIFACTS=… python tools/dump_reference.py --chunks 0 1 2

Per chunk i it writes:
    reference/mel_<i>.f32   [1,128,2885] float32: bench/frontend.py mel_bucket()
    reference/enc_<i>.f32   [1,361,640]  float32: encoder .aimodel on GPU, fp16 mel in
    reference/wav_<i>.f32   [n]          float32: the raw 16 kHz chunk samples
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def _required_dir(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise SystemExit(
            f"{variable} is not set. It must point at "
            + ("the corpus harness providing bench.frontend."
               if variable == "PARAKEET_BENCH"
               else "the converted Core AI artifacts directory.")
        )
    return Path(value).expanduser()


BENCH = _required_dir("PARAKEET_BENCH")
ART = _required_dir("PARAKEET_ARTIFACTS")
OUT = Path(os.environ.get("PARAKEET_REFERENCE")
           or Path(__file__).resolve().parent.parent / "reference").expanduser()

sys.path.insert(0, str(BENCH))
from bench.frontend import mel_bucket, BUCKET  # noqa: E402

import coreai.runtime as rt  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--all", action="store_true", help="every chunk in the manifest")
    ap.add_argument("--mel-only", action="store_true", help="skip the encoder pass")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--manifest", default=str(BENCH / "work" / os.environ.get("PARAKEET_CORPUS", "corpus_ch1") / "chunks.json"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    man = json.loads(Path(args.manifest).read_text())
    keep = None if args.all else set(args.chunks)
    wanted = {c["i"]: c for c in man["chunks"] if keep is None or c["i"] in keep}

    if args.mel_only:
        for i in sorted(wanted):
            ch = wanted[i]
            wav, sr = sf.read(man["path"], dtype="float32", start=ch["start"], stop=ch["end"])
            mel_bucket(wav).astype(np.float32).tofile(out_dir / f"mel_{i}.f32")
        print(f"wrote {len(wanted)} mel tensors to {out_dir}")
        return

    m = await rt.AIModel.load(
        str(ART / f"parakeet_encoder_float16_L{BUCKET}.aimodel"),
        rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    enc_fn = m.load_function("main")

    for i in sorted(wanted):
        ch = wanted[i]
        wav, sr = sf.read(man["path"], dtype="float32", start=ch["start"], stop=ch["end"])
        assert sr == man["sr"]
        mel = mel_bucket(wav)                       # [1,128,2885] float32
        r = await enc_fn({"mel": rt.NDArray(mel.astype(np.float16))})
        enc = r["enc_proj"].numpy().astype(np.float32)   # [1,361,640]
        wav.astype(np.float32).tofile(OUT / f"wav_{i}.f32")
        mel.astype(np.float32).tofile(OUT / f"mel_{i}.f32")
        enc.tofile(OUT / f"enc_{i}.f32")
        print(f"chunk {i}: wav {wav.shape} mel {mel.shape} enc {enc.shape} "
              f"| enc absmax {np.abs(enc).max():.4f}")


if __name__ == "__main__":
    asyncio.run(main())
