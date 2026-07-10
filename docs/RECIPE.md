# The recipe: Qwen3.6-35B-A3B at 100+ tokens per second on a MacBook

This page lists only the configuration that ships in this repo. Every lever
here is measured, kept, and reproducible.

Reference hardware: M4 Max, 128 GB unified memory. Numbers are decode tokens
per second on real completions. Greedy outputs are verified byte-identical
against non-speculative decoding.

## Results you should be able to reproduce

| workload | tokens/sec |
|---|---|
| greedy coding prompt | 105 to 109 (anchor bench, thermals move it a few points) |
| greedy coding, 300-token completions | ~110 |
| sampled coding (temperature 0.6) | ~100 |
| prose, sampled | ~81 |

The unoptimized baseline for the same model and weights is 75 tokens/sec.
The gap comes from the lever stack below.

## The stack

| layer | choice | measured effect |
|---|---|---|
| model | `unsloth/Qwen3.6-35B-A3B-MLX-8bit` | MoE with 3B active parameters. 8-bit passes the tool-call formatting gate that 4-bit fails |
| engine | [Rapid-MLX fork, branch `qwen36-mtp-tuned`](https://github.com/photonsarefree/Rapid-MLX/tree/qwen36-mtp-tuned) | contains every code lever below as reviewed commits |
| drafter | rebuilt MTP sidecar (`scripts/build_mtp_sidecars.py`) | the model's own multi-token-prediction head, re-quantized to match the weights. Raw HuggingFace tensors give 0% acceptance |
| speculation | MTP, depth auto-tuned up to 2, any temperature | +20% at depth 1; depth 2 adds ~2.5% greedy and ~6% sampled |
| drafter cache trim fix | `RAPID_MLX_MTP_TRIM_FIX=1` (set by `llm-serve`) | +3.5% at depth 2. Keeps one real context entry per rejected round that the stock code discarded |
| async draft submission | on by default in the fork | +11% greedy and sampled. The GPU runs the drafter while the CPU prepares the verification pass |
| compiled verification pass | `RAPID_MLX_MTP_COMPILED_VERIFY=1` (set by `llm-serve`) | +9 to 12%. The verification forward is compiled once and replayed, removing ~4 ms of per-round graph construction; verified bit-exact, with compile warm-up kept out of the depth controller's cost window |
| sampled-request speculation | on by default in the fork | sampled traffic previously fell back to plain decoding at 77 tok/s. Exact acceptance math keeps the output distribution unchanged. Seeded requests still use plain decoding so seeds stay reproducible |
| KV cache | K8V4 compression over a bf16 base (default; `--no-turboquant` for plain int8) | +7 to 16% at 16k to 64k context, neutral at short context (on/off measured 114.5 vs 114.2 back-to-back), retrieval verified at 16k and 32k; int8 separately verified at 100k |
| thinking mode | default on via `RAPID_MLX_DEFAULT_THINKING=1` | stock builds silently strip reasoning from tool-calling requests |
| prompt handling | `--pflash off` for coding | PFlash is lossy prompt compression. Wrong trade for code |

The MoE gate/up projection fusion (+0.7%, token-exact) now runs inside the
engine fork and is on by default, so every lever above is active out of the
box with `llm-serve`.

## Reproduce it

```bash
./install.sh --with-models      # venv + patched engine + sidecars + models (~70 GB)
llm-serve 35b                   # serves on http://127.0.0.1:8642/v1
```

Point any OpenAI or Anthropic compatible client at the server. Client
configuration examples for Hermes and OpenCode are in `configs/`.

Verify your install reproduces our numbers:

```bash
llm-serve 35b
curl -s http://127.0.0.1:8642/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","messages":[{"role":"user","content":"Implement a thread-safe LRU cache in Python with O(1) get and put."}],"max_tokens":300,"temperature":0}'
# watch the server log for the reported decode rate, or use /metrics
curl -s http://127.0.0.1:8642/metrics | grep spec_decode
```

Expected: acceptance ratio near 0.69 at depth 2 on coding prompts, decode
in the 100 to 110 tokens/sec range on an M4 Max depending on thermals.

## What is and is not guaranteed

1. Greedy speculation is verified byte-identical across draft depths
   (fingerprint-tested; the fork passes 125 of 128 upstream MTP tests, and
   the 3 failures are explained in the fork's commit messages).
2. Sampled speculation preserves the output distribution by the
   rejection-sampling theorem. It does not reproduce token-for-token across
   runs, which is also true of plain sampling. Seeded requests bypass
   speculation entirely.
3. Quantization is a deliberate accuracy trade validated by retrieval and
   tool-formatting gates rather than a proof. We do not quantize below
   8-bit anywhere in this recipe.

Sharp edges: never place `model-mtp.safetensors` inside a model snapshot
directory (the loader corrupts norm weights; sidecars live outside, and the
tooling enforces this). Vision requests go through `llm-vision`.
