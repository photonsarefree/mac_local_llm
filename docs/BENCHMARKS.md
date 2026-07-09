# Benchmarks

Measured on a **MacBook Pro M4 Max, 128GB RAM, macOS 26.3.1**, Rapid-MLX
0.9.13, int8 KV cache, 256-token generations, coding-style prompts built from
real source code. Your numbers will scale with memory bandwidth (M4 Max ≈ 546
GB/s; M3 Ultra ≈ 819 GB/s runs ~1.5× faster; base M-series ~1/3).

## Decode speed (tok/s) vs context length

| model / config | 1k | 16k | 32k | 64k | 100k |
|---|---|---|---|---|---|
| **35B MoE** (int8) | 75 | 71 | 64 | ~45 | 16–18 |
| **35B MoE** + MTP | 89 | 70 | 51 | 38 | 16 |
| **27B dense** (int8) | 12 | 8.6 | 8.3 | 7.8 | 7.3 |
| 27B bf16 KV | 13 | 7.1 | 8.3 | 7.9 | 7.4 |
| 27B int4 KV | 9.4 | – | – | – | – |

The 35B MoE is the daily driver — 3–6× the 27B's decode speed at similar RAM.

## Time-to-first-token (TTFT)

| | 35B cold | 35B warm | 27B cold | 27B warm |
|---|---|---|---|---|
| 1k   | 1.2s  | 0.7s | 5s    | 0.5s |
| 16k  | 12s   | 0.8s | 90s   | 0.6s |
| 32k  | 28s   | 1.2s | 206s  | 0.7s |
| 64k  | 74s   | 1.9s | 470s  | 1.0s |
| 100k | 247s  | 6.4s | ~820s | 1.2s |

**Cold** = empty cache (first time a prefix is seen). **Warm** = prefix cache
hit (same or extended prompt). The huge cold→warm gap is why the prefix cache
matters so much for agentic/multi-turn use: the first touch of a big context is
slow, but every follow-up sharing that prefix is ~1s. Raise `--cache-gb` to
keep more large contexts warm.

## Memory (peak RSS)

Flat across context length because these are **hybrid** models — only the
full-attention layers hold a growing KV cache (16 of 64 layers on the 27B, 10
of 40 on the 35B), so KV is tiny even at 100k (6.1GB / 2.0GB at bf16).

| model | weights + KV @100k |
|---|---|
| 27B | ~32–33 GB |
| 35B | ~35–41 GB |

Both fit comfortably under a 96GB budget on a 128GB machine. `--gpu-util 0.85`
(default) caps Metal at 85% of the GPU working set, leaving headroom for the OS.

## KV dtype

int8 is the shipping default: quality-verified (a planted-constant retrieval
probe passes at 32k **and** 100k) and speed/memory-neutral vs bf16 in practice.
**int4 was measurably slower on the 27B** (9.4 vs 12 tok/s at 1k) with added
quality risk — not recommended. Since KV is small on these hybrid models, KV
quantization is a long-context bandwidth lever, not a memory necessity.

## MTP (speculative decoding)

Draft acceptance is high at short context but **decays as context grows**:

| context | 27B accept | 35B accept |
|---|---|---|
| 1k   | 74% | 86% |
| 16k  | –   | 56% |
| 32k  | –   | 49% |
| 64k+ | –   | ~42% |

So MTP is net-positive on the 27B (slow base decode) and on the 35B **only at
short context** — which is why `llm-serve` defaults MTP **on for 27b, off for
35b**. Toggle with `--mtp` / `--no-mtp`; watch it live with
`curl -s localhost:8642/metrics | grep spec_decode`.

## Vision

Via `llm-vision` (mlx-vlm): a 1920×1080 code screenshot is ~2,080 image tokens,
prefills at ~680 tok/s, generates at ~75 tok/s, ~40GB peak RAM. Correctly reads
code and identifies bugs from screenshots.

## Additional perf-lever A/Bs

Measured on the same M4 Max after the main sweep. (Decode tok/s is the clean
metric here — the disk prefix cache persists across server restarts, so
"cold" TTFT on the *second* config to serve a given model can be contaminated;
decode is unaffected.)

### TurboQuant K8V4 KV — `llm-serve 35b --turboquant`

K8V4 (K 8-bit Hadamard + V 4-bit Lloyd-Max) vs our int8-KV default, 35B decode:

| context | int8 (default) | K8V4 | K8V4 RSS |
|---|---|---|---|
| 1k  | 76.4 | 77.7 | 35.3 GB |
| 16k | 65.8 | 70.5 | 35.7 GB |
| 32k | 62.3 | 63.3 | 36.4 GB |
| 64k | 47.5 | **55.0** | 38.3 GB |

The decode gain grows with context (+2% @1k → +16% @64k) as KV bandwidth starts
to matter — but K8V4 uses **~3 GB more RAM** at 64k (its transform/codebook
buffers outweigh the KV it saves, since KV is already tiny on this hybrid arch).
Needle retrieval still passes at 32k. **Verdict:** a worthwhile opt-in for
sustained long-context (32k+) work, not worth changing the default given the
RAM cost and near-zero benefit at short context.

### prefill-step 8192 vs 2048 (27B) — no effect

Aimed at the 27B's slow cold long-prompt prefill. Result: **null.**

| context | step 2048 (default) | step 8192 |
|---|---|---|
| 32k cold TTFT | 206 s | 205 s |
| 64k cold TTFT | 470 s | 489 s |

Raising the prefill chunk size does not help (marginally worse at 64k). The
27B's cold-prefill cost is compute-bound (hybrid GatedDeltaNet + full-attention
prefill), not chunk-size-limited. Keep the default; the real mitigation for slow
long-prompt cold starts is the prefix cache (warm hits are ~1 s).

### 6-bit vs 8-bit (35B) — `llm-serve 35b --6bit`

`mlx-community/Qwen3.6-35B-A3B-6bit` vs the 8-bit default:

| context | 8-bit decode | 6-bit decode | 6-bit RSS |
|---|---|---|---|
| 1k  | 76.4 | **82.1** (+7%) | 27.1 GB |
| 16k | 65.8 | **74.5** (+13%) | 27.1 GB |
| 32k | 62.3 | **66.6** (+7%) | 27.1 GB |

6-bit is **faster at every context and uses ~8 GB less RAM** (27 vs 35 GB, ~23%
less) — the win comes from lower weight-read bandwidth per token. Quality held up
in testing:

- **Needle retrieval @32k: PASS** (same planted-constant probe as the main sweep).
- **Tool-call formatting: PASS** — the exact 2-tool composition that makes plain
  4-bit ramble 4000+ tokens with no `<tool_call>` produced a valid tool call
  here, even with thinking on. (This is why 6-bit, not 4-bit, is the one to use:
  4-bit breaks strict tool-call output on this model.)
- Coding output: correct on spot checks.

**Verdict:** 6-bit is a strong option — arguably the better daily driver on
RAM-constrained Macs (32–48 GB), where the 8-bit's ~35 GB is tight. The 8-bit
remains the max-quality reference (6-bit quality was spot-checked, not
exhaustively evaluated). Note MTP isn't wired for the 6-bit build (would need a
6-bit-matched sidecar), but 35B defaults MTP off anyway.
