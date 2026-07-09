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
