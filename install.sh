#!/usr/bin/env bash
# mac_local_llm installer — local Qwen3.6 inference on Apple Silicon.
#
#   ./install.sh                 # software only (venv + rapid-mlx + patches + PATH)
#   ./install.sh --with-models   # also download both models (~66GB) + build MTP sidecars
#   ./install.sh --models 35b    # download just one model (27b|35b) + its sidecar
#
# Env overrides:
#   RAPID_MLX_VENV   venv location (default ~/.rapid-mlx)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAPID_MLX_VENV="${RAPID_MLX_VENV:-$HOME/.rapid-mlx}"
BIN_DIR="$HOME/.local/bin"
RAPID_MLX_VERSION="0.9.13"   # pinned: the patches target this exact version
SIDECAR_DIR="$RAPID_MLX_VENV/mtp_sidecars"

WITH_MODELS=""; ONLY_MODEL=""
while [ $# -gt 0 ]; do
    case "$1" in
        --with-models) WITH_MODELS=1 ;;
        --models) WITH_MODELS=1; ONLY_MODEL="$2"; shift ;;
        -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
    shift
done

say() { printf '\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ── 1. platform ──────────────────────────────────────────────────────────────
[ "$(uname -s)" = "Darwin" ] || die "macOS only (MLX requires Apple Silicon)."
[ "$(uname -m)" = "arm64" ]  || die "Apple Silicon (M-series) required. Detected: $(uname -m)."

RAM_GB=$(sysctl -n hw.memsize | awk '{printf "%d", $1/1073741824}')
say "Apple Silicon, ${RAM_GB}GB RAM, macOS $(sw_vers -productVersion)"
if [ "$RAM_GB" -lt 48 ]; then
    echo "  WARNING: <48GB RAM. 27b needs ~40GB free, 35b ~44GB. Serving may swap or OOM."
fi

# ── 2. python ────────────────────────────────────────────────────────────────
PYTHON=""
for py in python3.13 python3.12 python3.11 python3.10; do
    if command -v "$py" >/dev/null 2>&1; then PYTHON="$py"; break; fi
done
[ -n "$PYTHON" ] || die "Python 3.10+ not found. Install with: brew install python@3.12"
say "Using $($PYTHON --version)"

# ── 3. venv + rapid-mlx ──────────────────────────────────────────────────────
if [ ! -d "$RAPID_MLX_VENV" ]; then
    say "Creating venv at $RAPID_MLX_VENV"
    "$PYTHON" -m venv "$RAPID_MLX_VENV"
fi
PIP="$RAPID_MLX_VENV/bin/pip"
VENV_PY="$RAPID_MLX_VENV/bin/python"
"$PIP" install --upgrade pip -q

say "Installing rapid-mlx (patched fork, base $RAPID_MLX_VERSION) [vision,dflash,mtp]"
# The fork branch carries the full patch set as reviewed commits — the three
# original fixes (docs/PATCHES.md) PLUS the MTP upgrades that can't be
# expressed as anchor patches: speculation at any temperature (exact
# Leviathan-Chen accept; seeded requests fall back to plain decode),
# chain-of-K on the GatedDeltaNet hybrid (multi-boundary state snapshots,
# lossless-verified), drafter-hidden cascade, vectorized verify sampling.
# Branch commits: https://github.com/photonsarefree/Rapid-MLX/commits/qwen36-mtp-tuned
FORK_REF="git+https://github.com/photonsarefree/Rapid-MLX@qwen36-mtp-tuned"
if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$VENV_PY" "rapid-mlx[vision,dflash,mtp] @ $FORK_REF"
else
    "$PIP" install "rapid-mlx[vision,dflash,mtp] @ $FORK_REF"
fi

# ── 4. patches (legacy PyPI installs only) ───────────────────────────────────
# The fork already contains everything; apply_patches.py anchor-matches
# pristine 0.9.13 text, so on a fork install it cleanly no-ops.
say "Verifying patch state"
"$VENV_PY" "$REPO_DIR/scripts/apply_patches.py" --python "$VENV_PY" \
    || echo "  (some patches did not apply — see message above; basic serving still works)"

# ── 5. PATH wiring ───────────────────────────────────────────────────────────
say "Linking launchers into $BIN_DIR"
mkdir -p "$BIN_DIR"
ln -sf "$REPO_DIR/bin/llm-serve"  "$BIN_DIR/llm-serve"
ln -sf "$REPO_DIR/bin/llm-vision" "$BIN_DIR/llm-vision"
chmod +x "$REPO_DIR/bin/llm-serve" "$REPO_DIR/bin/llm-vision"

if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    SHELL_RC="$HOME/.zshrc"; [ -n "${BASH_VERSION:-}" ] && SHELL_RC="$HOME/.bashrc"
    if ! grep -q "$BIN_DIR" "$SHELL_RC" 2>/dev/null; then
        printf '\n# mac_local_llm\nexport PATH="%s:$PATH"\n' "$BIN_DIR" >> "$SHELL_RC"
    fi
    echo "  Added $BIN_DIR to PATH in $SHELL_RC — restart your shell or: export PATH=\"$BIN_DIR:\$PATH\""
fi

# ── 6. optional: models + sidecars ───────────────────────────────────────────
if [ -n "$WITH_MODELS" ]; then
    HF="$RAPID_MLX_VENV/bin/hf"; [ -x "$HF" ] || HF="$RAPID_MLX_VENV/bin/huggingface-cli"
    dl() { say "Downloading $1 (large)"; "$HF" download "$1"; }
    if [ "$ONLY_MODEL" = "27b" ]; then
        dl unsloth/Qwen3.6-27B-MLX-8bit
        "$VENV_PY" "$REPO_DIR/scripts/build_mtp_sidecars.py" --out "$SIDECAR_DIR" --only 27b
    elif [ "$ONLY_MODEL" = "35b" ]; then
        dl unsloth/Qwen3.6-35B-A3B-MLX-8bit
        "$VENV_PY" "$REPO_DIR/scripts/build_mtp_sidecars.py" --out "$SIDECAR_DIR" --only 35b
    else
        dl unsloth/Qwen3.6-27B-MLX-8bit
        dl unsloth/Qwen3.6-35B-A3B-MLX-8bit
        say "Building MTP sidecars"
        "$VENV_PY" "$REPO_DIR/scripts/build_mtp_sidecars.py" --out "$SIDECAR_DIR"
    fi
fi

# ── done ─────────────────────────────────────────────────────────────────────
say "Installed."
echo ""
echo "  Start a server:   llm-serve 35b        (or 27b)"
echo "  Connection info:  printed on start; llm-serve status re-prints it"
echo "  Image Q&A:        llm-vision <image> \"question\""
echo ""
if [ -z "$WITH_MODELS" ]; then
    echo "  Models not downloaded yet. First 'llm-serve' will pull the weights"
    echo "  automatically, OR fetch + build MTP sidecars now:"
    echo "    ./install.sh --with-models"
    echo ""
fi
echo "  Point any OpenAI-compatible client at the printed base URL (no API key)."
echo "  See README.md for Hermes / OpenCode / KiloCode setup."
