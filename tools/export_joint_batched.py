"""Export a BATCHED Parakeet-v2 joint: K encoder frames per dispatch instead of one.

Why this exists
---------------
The sequential TDT loop calls the joint once per encoder frame: ~22 700 calls for 66 minutes of
audio.  Loop-level speculation was already measured and rejected: 89.7 % of joint calls emit a
token, so a lookahead window is thrown away almost every time it is filled, and issuing K
*separate* dispatches to save 10 % of the round trips is a straight loss.

The version that could still pay is a joint graph that takes `enc_frames [K,640]` and returns
`token_logits [K,1025]` / `dur_logits [K,5]` in ONE dispatch.  Then the trade stops being
"K× the work to skip 10 % of the round trips" and becomes "one dispatch instead of K".  The
current bundle cannot express that: its `enc_frame` is statically `[1,640]`, hence this export.

The maths is unchanged: `head(relu(enc_frame + dec_out))`, with `dec_out [1,640]` broadcasting
across the K rows.  That broadcast is exactly why one predictor state can serve K frames.

Gate (run here, before any Swift work touches it):
  * fp32 oracle: the eager batched module vs the eager single-frame module on the golden
    `enc_proj`, cos >= 0.999 AND argmax-derived (token, duration) pairs identical for every frame;
  * the exported bundle vs the same oracle, through `coreai.runtime`, same two bars.

Needs an export environment with torch + coreai_models. COREAI_MODEL_ZOO only locates the
stored fp32 oracle (`oracle_v2_30s.npz`); no code is imported from it. Override the oracle
directly with --oracle to drop the variable entirely.

    COREAI_MODEL_ZOO=/path/to/coreai-model-zoo python tools/export_joint_batched.py --k 8 --k 4
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Optional: only used to locate the default oracle. --oracle overrides it, so an
# unset COREAI_MODEL_ZOO is fine as long as the path is passed explicitly.
_zoo_root = os.environ.get("COREAI_MODEL_ZOO")
ZOO = Path(_zoo_root).expanduser() / "conversion" / "parakeet" if _zoo_root else None
HF_ID = os.environ.get("PARAKEET_HF_ID", "nvidia/parakeet-tdt-0.6b-v2")
HID = 640


class JointBatched(nn.Module):
    """`dec_out [1,640]` + `enc_frames [K,640]` -> `token_logits [K,V]`, `dur_logits [K,D]`.

    Identical arithmetic to the shipped single-frame joint; only the leading dimension moves.
    """

    def __init__(self, vocab: int, ndur: int):
        super().__init__()
        self.vocab = vocab
        self.head = nn.Linear(HID, vocab + ndur, bias=True)

    def forward(self, dec_out, enc_frames):
        logits = self.head(torch.relu(enc_frames + dec_out))   # broadcast [1,640] over [K,640]
        return logits[:, :self.vocab], logits[:, self.vocab:]


def load_head(joint: JointBatched) -> None:
    from safetensors import safe_open
    with safe_open(str(Path(HF_ID) / "model.safetensors"), framework="pt") as f:
        joint.head.weight.data.copy_(f.get_tensor("joint.head.weight"))
        joint.head.bias.data.copy_(f.get_tensor("joint.head.bias"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, action="append", default=None, help="repeatable")
    ap.add_argument("--oracle", default=str(ZOO / "oracle_v2_30s.npz") if ZOO else None,
                    help="stored fp32 oracle .npz (defaults under COREAI_MODEL_ZOO if set)")
    ap.add_argument("--artifacts", default=os.environ.get("PARAKEET_BATCHED_OUT", "artifacts_batched"),
                    help="output dir for the batched joint bundles (PARAKEET_BATCHED_OUT)")
    args = ap.parse_args()
    if not args.oracle:
        ap.error("--oracle is required when COREAI_MODEL_ZOO is unset")
    ks = args.k or [8]

    d = np.load(args.oracle)
    enc_proj = torch.from_numpy(d["enc_proj"]).float()           # [T,640] golden
    vocab, blank = int(d["vocab_size"]), int(d["blank_id"])
    durations = d["durations"].tolist()
    T = int(d["T_valid"])
    print(f"[oracle] T={T} vocab={vocab} blank={blank} durations={durations}")

    joint = JointBatched(vocab, len(durations)).eval()
    load_head(joint)

    # A realistic dec_out to gate against: the shipped predictor's very first state (blank in,
    # zero LSTM state) is the one every chunk actually starts from.
    torch.manual_seed(0)
    dec = torch.randn(1, HID) * 0.5

    with torch.no_grad():
        ref_tl, ref_dl = joint(dec, enc_proj[:T])                # [T,V] / [T,D] single pass
    ref_pairs = [(int(ref_tl[t].argmax()), durations[int(ref_dl[t].argmax())]) for t in range(T)]

    import coreai.runtime as rt
    from coreai_models.export.macos import export_to_coreai

    art = Path(args.artifacts)
    art.mkdir(parents=True, exist_ok=True)

    for K in ks:
        prog = export_to_coreai(
            joint,
            {"dec_out": torch.zeros(1, HID), "enc_frames": torch.zeros(K, HID)},
            dynamic_shapes=None,
            input_names=("dec_out", "enc_frames"),
            output_names=("token_logits", "dur_logits"),
            state_names=None,
            externalize_modules=[])
        prog.optimize()
        path = art / f"parakeet_joint_batched_k{K}_float32.aimodel"
        shutil.rmtree(path, ignore_errors=True)
        meta = rt.AIModelAssetMetadata()
        meta.license = "cc-by-4.0"
        prog.save_asset(path, meta)
        sz = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
        print(f"[save] {path.name} ({sz:.1f} MB)")

        async def gate() -> tuple[float, int, int]:
            m = await rt.AIModel.load(
                str(path),
                rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
            fn = m.load_function("main")
            cos_sum, n_win, pairs_ok, n_frames = 0.0, 0, 0, 0
            # Walk the golden encoder output in K-frame windows: the exact access pattern the
            # Swift speculative path will use. Well under the ~8k-call Python binding ceiling.
            for lo in range(0, T, K):
                block = torch.zeros(K, HID)
                hi = min(T, lo + K)
                block[: hi - lo] = enc_proj[lo:hi]
                r = await fn({"dec_out": rt.NDArray(dec.numpy()),
                              "enc_frames": rt.NDArray(block.numpy())})
                tl = torch.from_numpy(r["token_logits"].numpy().astype(np.float32))
                dl = torch.from_numpy(r["dur_logits"].numpy().astype(np.float32))
                for j in range(hi - lo):
                    cos_sum += float(torch.nn.functional.cosine_similarity(
                        tl[j], ref_tl[lo + j], dim=-1))
                    n_win += 1
                    got = (int(tl[j].argmax()), durations[int(dl[j].argmax())])
                    pairs_ok += int(got == ref_pairs[lo + j])
                    n_frames += 1
            return cos_sum / max(n_win, 1), pairs_ok, n_frames

        cos, ok, n = asyncio.run(gate())
        verdict = "PASS" if cos >= 0.999 and ok == n else "FAIL"
        print(f"[gate K={K}] token_logits cos mean {cos:.6f}  greedy (token,dur) pairs "
              f"{ok}/{n} identical -> {verdict}")


if __name__ == "__main__":
    main()
