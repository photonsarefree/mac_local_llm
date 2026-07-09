# Patches applied to Rapid-MLX 0.9.13

`scripts/apply_patches.py` applies these to the installed `vllm_mlx` package.
They are idempotent (re-running is safe) and anchor-matched (each only fires if
its exact pristine text is present), so they cleanly no-op if already applied
and warn — rather than corrupt — if the installed version differs from 0.9.13.

Why they're needed: stock Rapid-MLX 0.9.13 ships Qwen3.6 MTP support, but it
doesn't actually engage for the released Qwen3.6 checkpoints, and thinking is
force-disabled for the agent (tool-calling) requests that matter most.

### 1. `spec_decode/mtp/detect.py` — MTP eligibility reads `text_config`

The MTP eligibility gate reads `mtp_num_hidden_layers` from the top level of
`config.json`. Qwen3.6 checkpoints are VLM-wrapped, so that field lives under
`text_config`. Stock behavior: `--spec-decode mtp` exits with "model does not
qualify" for every Qwen3.6 model. Fix: fall back to `text_config`.

### 2. `scheduler.py` — MTP surface check resolves the inner model

The injector installs the MTP head (`mtp_forward` / `make_mtp_cache` / `mtp`)
on the **inner** `TextModel` (`model.language_model`), but the scheduler checks
for those surfaces on the **outer** VLM wrapper, doesn't find them, and silently
falls back to plain decode. Fix: resolve `model.language_model` before the
check. (The repo's own tests construct the inner model directly, so CI never
sees the real runtime shape.)

### 3. `service/helpers.py` — `RAPID_MLX_DEFAULT_THINKING` switch

Rapid-MLX has three "auto-disable thinking" gates that inject
`enable_thinking=false` whenever the client doesn't explicitly ask for
thinking — including on all tool-calling requests. So agents (Hermes, OpenCode,
etc.) never receive reasoning tokens, regardless of client settings. Fix: add a
`RAPID_MLX_DEFAULT_THINKING=1` env switch that makes thinking the default for
unpinned requests and bypasses both the tools and casual-chat auto-disable
gates. Per-request `enable_thinking: false` still wins. `llm-serve --thinking on`
(the default) sets this env var.

## A note on the MTP sidecars (not a code patch)

Even with fixes 1–2, the raw Qwen3.6 MTP tensors from the HF base repos give
**0% draft acceptance** as a 1-step drafter (verified across every plausible
input contract). The weights that work are the retrained
`mlx-community/Qwen3.6-*-MTP-bf16` drafters. `scripts/build_mtp_sidecars.py`
converts them into the quantized sidecar format the injector expects (norm +1.0
shift, packed-expert split for the MoE, 8-bit gs=64 to match the base). Result:
~74% acceptance on the 27B, ~86% at short context on the 35B.

## Upgrading rapid-mlx

The patches pin to 0.9.13. If you `pip install --upgrade rapid-mlx`, the anchors
may no longer match and `apply_patches.py` will warn instead of applying — MTP
and thinking-default would then be off, but basic serving still works. Re-pin to
0.9.13 (or update the anchors) to restore them.
