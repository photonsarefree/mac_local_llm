#!/usr/bin/env python3
"""Apply the mac_local_llm patches to an installed Rapid-MLX (vllm_mlx).

Rapid-MLX 0.9.13 ships Qwen3.6 MTP support and thinking handling that don't
quite work for the two Qwen3.6 checkpoints we target. This script applies three
fixes to the installed package, in place. It is idempotent (safe to re-run) and
non-destructive (each patch only fires if its exact pristine anchor is present
and the fix is not already applied).

Patches:
  1. detect.py   — read mtp_num_hidden_layers from text_config (VLM-wrapped
                   Qwen3.6 configs nest it there), so --spec-decode mtp is not
                   rejected at boot.
  2. scheduler.py — resolve the inner TextModel (model.language_model) before
                   checking for MTP surfaces, so the injected head is found.
  3. helpers.py  — add a RAPID_MLX_DEFAULT_THINKING env switch that makes
                   thinking the server-side default for unpinned requests
                   (including agent tool-calls), which agent UIs render as
                   reasoning. Per-request enable_thinking still wins.

Usage:
    python apply_patches.py [--python /path/to/venv/bin/python] [--check]

Without --python, uses the interpreter running this script (so run it with the
venv's python). --check reports status without writing.
"""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

# Each patch: (relative path, human name, sentinel-if-already-applied,
#              pristine find, patched replace)
PATCHES = [
    (
        "spec_decode/mtp/detect.py",
        "MTP eligibility reads text_config",
        "_mtp_raw = config.get(",
        '    num_mtp_layers = _safe_int(config.get("mtp_num_hidden_layers"), 0)\n',
        (
            "    # Qwen3.6 checkpoints are VLM-wrapped: ``mtp_num_hidden_layers``\n"
            "    # lives under ``text_config`` (the injector already reads it from\n"
            "    # there at runtime) — fall back to the nested dict so eligibility\n"
            "    # matches what inject_mtp_support will actually see.\n"
            '    _mtp_raw = config.get("mtp_num_hidden_layers")\n'
            "    if _mtp_raw is None:\n"
            '        text_config = config.get("text_config")\n'
            "        if isinstance(text_config, dict):\n"
            '            _mtp_raw = text_config.get("mtp_num_hidden_layers")\n'
            "    num_mtp_layers = _safe_int(_mtp_raw, 0)\n"
        ),
    ),
    (
        "scheduler.py",
        "MTP surface check resolves inner TextModel",
        "Qwen3.6 checkpoints load as VLM-style wrappers",
        (
            '    if not (\n'
            '        hasattr(model, "mtp_forward")\n'
            '        and hasattr(model, "make_mtp_cache")\n'
            '        and hasattr(model, "mtp")\n'
            '    ):\n'
            '        logger.warning(\n'
            '            "[MTP-vendored] disabled: model lacks mtp_forward / make_mtp_cache / "\n'
        ),
        (
            "    # Qwen3.6 checkpoints load as VLM-style wrappers: qwen3_5_inject\n"
            "    # installs the MTP surfaces on the inner ``TextModel``\n"
            "    # (``model.language_model``), not the outer wrapper the engine\n"
            "    # holds. Resolve the inner model here — the generator's contract\n"
            "    # (``model.layers`` / ``mtp_forward`` / ``__call__(return_hidden=...)``)\n"
            "    # is the inner model's interface.\n"
            '    if not hasattr(model, "mtp_forward"):\n'
            '        _inner_lm = getattr(model, "language_model", None)\n'
            '        if _inner_lm is not None and hasattr(_inner_lm, "mtp_forward"):\n'
            "            model = _inner_lm\n"
            "\n"
            '    if not (\n'
            '        hasattr(model, "mtp_forward")\n'
            '        and hasattr(model, "make_mtp_cache")\n'
            '        and hasattr(model, "mtp")\n'
            '    ):\n'
            '        logger.warning(\n'
            '            "[MTP-vendored] disabled: model lacks mtp_forward / make_mtp_cache / "\n'
        ),
    ),
    (
        "service/helpers.py",
        "thinking default helper",
        "def _default_thinking_on(",
        "def _resolve_enable_thinking(request) -> bool | None:\n",
        (
            "def _default_thinking_on() -> bool:\n"
            '    """Operator override: RAPID_MLX_DEFAULT_THINKING=1 makes thinking the\n'
            "    server-side default for requests that did not pin a preference —\n"
            "    including tool-calling requests (skips the R12-T1F auto-disable).\n"
            '    Per-request ``enable_thinking: false`` still wins."""\n'
            "    import os\n"
            "\n"
            '    return os.environ.get("RAPID_MLX_DEFAULT_THINKING", "").strip().lower() in {\n'
            '        "1",\n'
            '        "true",\n'
            '        "yes",\n'
            '        "on",\n'
            "    }\n"
            "\n"
            "\n"
            "def _resolve_enable_thinking(request) -> bool | None:\n"
        ),
    ),
    (
        "service/helpers.py",
        "thinking default in resolve()",
        "if pinned is None and _default_thinking_on():",
        (
            "    cfg = get_config()\n"
            "    if cfg.no_thinking:\n"
            "        return False\n"
            "    return _extract_thinking_from_request(request)\n"
        ),
        (
            "    cfg = get_config()\n"
            "    if cfg.no_thinking:\n"
            "        return False\n"
            "    pinned = _extract_thinking_from_request(request)\n"
            "    if pinned is None and _default_thinking_on():\n"
            "        return True\n"
            "    return pinned\n"
        ),
    ),
    (
        "service/helpers.py",
        "thinking default skips tools auto-disable",
        "skip the tools\n        # auto-disable entirely",
        (
            '    tools = getattr(request, "tools", None)\n'
            "    if not tools:\n"
            "        return False\n"
        ),
        (
            "    if _default_thinking_on():\n"
            "        # Operator pinned thinking-on server-wide — skip the tools\n"
            "        # auto-disable entirely (they accept the token-budget risk).\n"
            "        return False\n"
            '    tools = getattr(request, "tools", None)\n'
            "    if not tools:\n"
            "        return False\n"
        ),
    ),
    (
        "service/helpers.py",
        "thinking default skips casual-chat auto-disable",
        "skip the casual-chat auto-disable",
        (
            "    load-bearing signal for the route's structured log line.\n"
            '    """\n'
            "    cfg = get_config()\n"
        ),
        (
            "    load-bearing signal for the route's structured log line.\n"
            '    """\n'
            "    if _default_thinking_on():\n"
            "        # Operator pinned thinking-on server-wide (RAPID_MLX_DEFAULT_THINKING)\n"
            "        # -- skip the casual-chat auto-disable, same as the tools gate.\n"
            "        return False\n"
            "    cfg = get_config()\n"
        ),
    ),
]


def find_vllm_mlx(py: str | None) -> Path:
    if py:
        out = subprocess.check_output(
            [py, "-c", "import vllm_mlx, os; print(os.path.dirname(vllm_mlx.__file__))"],
            text=True,
        ).strip()
        return Path(out)
    spec = importlib.util.find_spec("vllm_mlx")
    if spec is None or not spec.origin:
        sys.exit("vllm_mlx not importable — run with --python <venv>/bin/python")
    return Path(spec.origin).parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", help="venv python that has vllm_mlx installed")
    ap.add_argument("--check", action="store_true", help="report only, do not write")
    args = ap.parse_args()

    root = find_vllm_mlx(args.python)
    print(f"vllm_mlx: {root}")

    applied = skipped = failed = 0
    # Group edits per file so we write each file once.
    per_file: dict[str, str] = {}
    for rel, name, sentinel, find, replace in PATCHES:
        path = root / rel
        if rel not in per_file:
            per_file[rel] = path.read_text()
        text = per_file[rel]
        if sentinel in text:
            print(f"  ✓ already applied: {name}")
            skipped += 1
            continue
        if find not in text:
            print(f"  ✗ ANCHOR NOT FOUND: {name} ({rel}) — version drift? skipping")
            failed += 1
            continue
        per_file[rel] = text.replace(find, replace, 1)
        print(f"  + applying: {name}")
        applied += 1

    if not args.check:
        for rel, text in per_file.items():
            (root / rel).write_text(text)

    # Byte-compile to surface any syntax error we introduced.
    if applied and not args.check:
        for rel in {p[0] for p in PATCHES}:
            import py_compile

            try:
                py_compile.compile(str(root / rel), doraise=True)
            except py_compile.PyCompileError as e:
                sys.exit(f"SYNTAX ERROR after patching {rel}: {e}")

    print(f"\n{applied} applied, {skipped} already-present, {failed} failed")
    if failed:
        print("Some patches did not apply — MTP or thinking-default may be off. "
              "This usually means the installed rapid-mlx version differs from 0.9.13.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
