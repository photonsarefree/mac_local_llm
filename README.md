# mac_local_llm

Run **Qwen3.6** locally on an Apple Silicon Mac with one command — an
OpenAI/Anthropic-compatible server you can point Hermes, OpenCode, KiloCode,
Cline, Cursor, or any OpenAI SDK at. Built on [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX)
(MLX inference for Apple Silicon).

Two models, both 8-bit MLX, both vision-capable, both 262K context:

| key   | model | strengths | decode speed* |
|-------|-------|-----------|---------------|
| `35b` | Qwen3.6-35B-A3B (MoE, 3B active) | **fastest** — daily driver | ~75 tok/s @1k, 64 @32k |
| `27b` | Qwen3.6-27B (dense) | **strongest quality** | ~13 tok/s (MTP on) |

*Measured on an M4 Max, 128GB, int8 KV. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

This repo adds, on top of stock Rapid-MLX 0.9.13:
- **Working MTP speculative decoding** for both Qwen3.6 checkpoints (stock
  rapid-mlx rejects them / no-ops silently — three code fixes + rebuilt drafter
  sidecars are applied automatically).
- **Thinking-by-default** so agent UIs actually see the model's reasoning.
- A single `llm-serve` command with tuned defaults + a full lever guide.
- `llm-vision` for image Q&A (the OpenAI server can't do vision for these
  hybrid models — upstream limitation — so this wraps mlx-vlm).

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4), macOS 13+
- **RAM**: 48GB minimum, 64GB+ recommended (27b needs ~40GB free, 35b ~44GB)
- ~70GB free disk for both models (or ~35GB for one)
- Python 3.10+

## Install

```bash
git clone https://github.com/photonsarefree/mac_local_llm.git
cd mac_local_llm
./install.sh                 # software: venv + rapid-mlx + patches + PATH
# then, to also fetch weights (~66GB) and build MTP sidecars:
./install.sh --with-models   # or: ./install.sh --models 35b   (just one)
```

`install.sh` creates a venv at `~/.rapid-mlx`, installs `rapid-mlx==0.9.13`
(pinned — the patches target this version), applies the patches, and links
`llm-serve` / `llm-vision` into `~/.local/bin`. If you skip `--with-models`,
the first `llm-serve` downloads the model weights automatically (but you'll
want to build MTP sidecars separately — see below).

## Use

```bash
llm-serve 35b            # start the fast MoE model (default)
llm-serve 27b            # start the quality model (MTP on)
llm-serve status         # print connection details again
llm-serve stop           # stop (persists prefix cache)
llm-serve logs           # tail server log
llm-serve --help         # full option + curl reference
```

On start it prints everything a client needs:

```
  OpenAI endpoint:     http://127.0.0.1:8642/v1
  Anthropic endpoint:  http://127.0.0.1:8642  (/v1/messages)
  Auth:                none (use any key string if a client requires one, e.g. "local")
  Model id:            unsloth/Qwen3.6-35B-A3B-MLX-8bit   (or "default")
```

Quick test:

```bash
curl -s http://127.0.0.1:8642/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "default",
  "messages": [{"role":"user","content":"Write a Python LRU cache. Just code."}],
  "max_tokens": 400, "temperature": 0.6
}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
```

### Image Q&A

```bash
llm-vision screenshot.png "What bug does this code have?"
llm-vision --27b diagram.png "Explain this architecture"
```

## Connect your tools

All clients use the same endpoint (`http://127.0.0.1:8642/v1`), no API key
(use `local` if a field is required), model id `unsloth/Qwen3.6-35B-A3B-MLX-8bit`
or `unsloth/Qwen3.6-27B-MLX-8bit`.

- **KiloCode / Cline / Cursor / Continue**: add an "OpenAI Compatible" provider
  with the base URL, key `local`, and the model id above.
- **Hermes Agent**: merge [configs/hermes.custom_providers.yaml](configs/hermes.custom_providers.yaml)
  into `~/.hermes/config.yaml`, restart the app, pick `local-qwen`. For visible
  reasoning also set `display.show_reasoning: true`.
- **OpenCode**: merge [configs/opencode.jsonc](configs/opencode.jsonc) into
  `~/.config/opencode/opencode.jsonc`, restart the app, pick "Local Qwen".
- **Any OpenAI SDK**: `base_url="http://127.0.0.1:8642/v1"`, `api_key="local"`.

## Thinking (reasoning) on/off

Thinking is **on by default** here (agent UIs render it as reasoning). The
reasoning trace comes back in `reasoning_content`, the answer in `content`.
Turn it off per request:

```json
"chat_template_kwargs": {"enable_thinking": false}
```

Or server-wide: `llm-serve 35b --thinking off` (or `--thinking auto` for stock
behavior — off for tool-calls, on otherwise). See `llm-serve --help`.

## Performance levers

`llm-serve` exposes the knobs worth tuning (full details in `--help`):

| flag | default | note |
|------|---------|------|
| `--kv bf16\|int8\|int4` | int8 | int8 is quality-safe to 100k; int4 slower on 27b |
| `--mtp` / `--no-mtp` | 27b on, 35b off | MTP helps 27b + short-context; decays with length |
| `--cache-gb N` | 24 | prefix cache — a cached 100k prefix answers in ~1s vs minutes cold |
| `--prefill-step N` | 2048 | raise (8192) to cut cold long-prompt TTFT |
| `--turboquant` | off | 35b experimental KV compression |
| `--gpu-util 0.x` | 0.85 | fraction of GPU working set (machine-relative) |

## MTP sidecars

MTP (multi-token prediction speculative decoding) needs the model's MTP head
weights, which the unsloth conversions strip. This repo rebuilds working
sidecars from the public `mlx-community/Qwen3.6-*-MTP-bf16` drafters:

```bash
python scripts/build_mtp_sidecars.py --out ~/.rapid-mlx/mtp_sidecars   # both
python scripts/build_mtp_sidecars.py --only 35b                        # one
```

`install.sh --with-models` does this for you. Without a sidecar, `llm-serve`
still runs — it just skips MTP.

## What the patches do

`scripts/apply_patches.py` (run by the installer, idempotent) makes three fixes
to the installed rapid-mlx so Qwen3.6 MTP + thinking work. See
[docs/PATCHES.md](docs/PATCHES.md) for the full explanation. They re-apply
cleanly and only touch anchors present in 0.9.13.

## Credits & license

- Inference engine: [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) (Apache-2.0)
- Models: [Qwen3.6](https://huggingface.co/Qwen) via [unsloth](https://huggingface.co/unsloth) MLX conversions (Apache-2.0)
- MTP drafters: [mlx-community](https://huggingface.co/mlx-community) (Apache-2.0)

This repo: Apache-2.0. It only orchestrates the above — no model weights are
redistributed here.
