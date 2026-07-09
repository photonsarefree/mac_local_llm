#!/usr/bin/env python3
"""Build MTP speculative-decoding sidecars for the Qwen3.6 models.

Rapid-MLX's --spec-decode mtp needs the model's MTP head weights, but the
unsloth 8-bit MLX conversions strip them. The raw Qwen HF weights don't work as
a 1-step drafter (0% acceptance in testing). The weights that DO work are the
mlx-community retrained drafters — this converts them into the quantized sidecar
format Rapid-MLX's injector expects (norm +1.0 shift, packed-expert split for
the MoE, 8-bit gs=64 to match the base).

Output: <out-dir>/27b/model-mtp.safetensors  (~450 MB)
        <out-dir>/35b/model-mtp.safetensors  (~900 MB)

Usage (run with the rapid-mlx venv's python):
    python build_mtp_sidecars.py [--out ~/.rapid-mlx/mtp_sidecars] [--only 27b|35b]
"""

import argparse
import os

import mlx.core as mx
from huggingface_hub import snapshot_download

mx.set_default_device(mx.cpu)

NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    "norm.weight",                  # also matches pre_fc_norm_* below
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
)
BITS, GS = 8, 64

JOBS = {
    "27b": "mlx-community/Qwen3.6-27B-MTP-bf16",
    "35b": "mlx-community/Qwen3.6-35B-A3B-MTP-bf16",
}


def build(repo: str, dst: str) -> None:
    print(f"=== {repo} ===", flush=True)
    import glob as g

    d = snapshot_download(repo)
    weights = {}
    for f in g.glob(os.path.join(d, "*.safetensors")):
        weights.update(mx.load(f))
    print(f"  {len(weights)} tensors loaded", flush=True)

    weights = {(k[len("mtp."):] if k.startswith("mtp.") else k): v
               for k, v in weights.items()}

    # MoE packed-expert split (mirrors mlx-vlm / mlx-lm sanitize)
    for prefix in [k[: -len(".experts.gate_up_proj")] for k in list(weights)
                   if k.endswith(".experts.gate_up_proj")]:
        gate_up = weights.pop(f"{prefix}.experts.gate_up_proj")
        gate_w, up_w = mx.split(gate_up, 2, axis=-2)
        weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_w
        weights[f"{prefix}.switch_mlp.up_proj.weight"] = up_w
        weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
            f"{prefix}.experts.down_proj")
        print(f"  split experts under {prefix}", flush=True)

    out = {}
    for k, v in sorted(weights.items()):
        if any(k.endswith(s) for s in NORM_SUFFIXES) and v.ndim == 1:
            out["mtp." + k] = v + 1.0     # bake mlx-vlm's load-time norm shift
            continue
        if v.ndim == 1 or v.shape[-1] % GS != 0:
            out["mtp." + k] = v
            continue
        qw, qs, qb = mx.quantize(v, group_size=GS, bits=BITS)
        mx.eval(qw, qs, qb)
        base = "mtp." + (k[: -len(".weight")] if k.endswith(".weight") else k)
        out[base + ".weight"] = qw
        out[base + ".scales"] = qs
        out[base + ".biases"] = qb

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    mx.save_safetensors(dst, out)
    mb = sum(v.nbytes for v in out.values()) / 1e6
    print(f"  saved {len(out)} tensors ({mb:.0f} MB) -> {dst}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/.rapid-mlx/mtp_sidecars"))
    ap.add_argument("--only", choices=["27b", "35b"], help="build just one")
    args = ap.parse_args()

    keys = [args.only] if args.only else list(JOBS)
    for k in keys:
        build(JOBS[k], os.path.join(args.out, k, "model-mtp.safetensors"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
