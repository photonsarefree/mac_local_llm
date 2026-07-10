# The recipe: Qwen3.6-35B-A3B at full speed on Apple Silicon

The solidified, end-to-end configuration this repo installs — every choice
below is A/B-measured on an M4 Max (128 GB); numbers are decode tok/s on
real completions unless noted. TL;DR: **one 8-bit MoE model + its own MTP
head as drafter + depth-adaptive speculation + int-quantized KV** ≈ 90+
tok/s on greedy coding, ~85 at temperature 0.6, on a laptop.

## The stack, layer by layer

| layer | choice | why (measured) |
|---|---|---|
| weights | `unsloth/Qwen3.6-35B-A3B-MLX-8bit` | MoE 3B-active = fastest family; 8-bit passes the 2-tool-call gate that 4-bit fails; 6-bit is +7-13% and ~8 GB lighter if RAM-tight (`--6bit`) |
| engine | [Rapid-MLX fork `qwen36-mtp-tuned`](https://github.com/photonsarefree/Rapid-MLX/tree/qwen36-mtp-tuned) | MLX is the most bandwidth-efficient Qwen3.6 engine we measured (~375 GB/s effective vs llama.cpp ~322); the fork makes MTP actually work on these checkpoints and upgrades it (below) |
| drafter | rebuilt MTP sidecar (`scripts/build_mtp_sidecars.py`) | the model's own MTP head, re-quantized to match the weights; raw HF tensors give 0% acceptance |
| speculation | MTP, auto-K ≤ 2, trim fix, any temperature | see next section |
| KV cache | int8 (default) / `--turboquant` K8V4 for 16k+ | int8 needle-verified at 100k; K8V4 +7-16% at 16-64k; int4 is slower AND riskier — never |
| thinking | default-on via `RAPID_MLX_DEFAULT_THINKING=1` | stock silently strips reasoning from all tool-calling requests |
| prompt | `--pflash off` for coding | PFlash is lossy prompt compression — wrong trade for code |

## What the fork's speculation upgrade buys

Stock 0.9.13 only speculated on `temperature == 0` requests and only 1 draft
token deep. The fork (commits on the branch, ~300 lines, adversarially
reviewed):

- **Any-temperature speculation** — exact Leviathan–Chen acceptance with
  residual resampling, so temp 0.6/0.95/20 sampling is distribution-identical
  to plain decode, just faster. Seeded requests automatically fall back to
  plain decode to keep per-seed reproducibility. Coding at temp 0.6:
  **77 → ~85 tok/s**.
- **Chain-of-K on the GatedDeltaNet hybrid** — the linear-attention state
  now snapshots at every verify boundary, so multi-token drafts can partially
  accept. Greedy coding: **90 → 92 tok/s** (K=2 + trim fix), verified
  bit-identical outputs across K=1/K=2/auto-K.
- **Depth auto-tuning** — an expected-value controller picks draft depth
  0–2 per round and *parks* the drafter on content where it loses (prose,
  long-context acceptance decay), which is what makes MTP safe to default-on.

Measured α (draft acceptance) explains the design: the recursive MTP head
decays 0.83 → 0.65 → 0.39 by depth, so K=3 regresses and K≤2 + parking is
optimal. A purpose-trained block drafter is the next jump (in progress).

## Bring it to life

```bash
./install.sh --with-models      # venv + fork + sidecars + models (~70 GB)
llm-serve 35b                   # ~90 tok/s greedy code, ~85 @temp 0.6
llm-serve 35b --turboquant      # add K8V4 when you live at 16k+ context
llm-serve 27b                   # dense 27B when quality > speed
```

Point any OpenAI/Anthropic client at `http://127.0.0.1:8642/v1` (configs for
Hermes/OpenCode in `configs/`).

## What is and isn't guaranteed (read this once)

Honest accounting — "lossless" has three different strengths here:

1. **Speculation at temp 0 (greedy)**: outputs are verified **bit-identical**
   across draft depths (fingerprint-tested, plus the fork passes 125/128 of
   upstream's MTP suite; the 3 = 2 tests asserting the old greedy-only gate
   + 1 flake that fails on clean 0.9.13). Caveat: speculative *verify* runs
   positions batched; bf16 non-associativity can flip a near-tie argmax vs
   non-speculative decode — rare, both outputs are valid model samples.
2. **Speculation at temp > 0**: exact **in distribution** (the Leviathan–Chen
   theorem), not per-token reproducible — same guarantee class as any
   sampled decode. Seeded requests bypass speculation entirely.
3. **Quantization is lossy by choice**: 8-bit weights and int8 KV are
   accuracy/speed trades (validated by needle@100k, 2-tool-call formatting,
   and coherence gates — not proofs). Full-precision exists; it's just slow.

Known sharp edges: never place `model-mtp.safetensors` inside a model
snapshot directory (the loader globs it and corrupts norms — sidecars live
outside, the tooling enforces this); vision goes through `llm-vision`
(upstream can't serve hybrid VLMs); MTP metrics are on `/metrics` if you want
to watch acceptance live.
