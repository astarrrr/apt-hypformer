#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PIDSMAKER_ROOT="/home/astar/projects"

resolve_pidsmaker_dir() {
    if [ -n "${PIDSMAKER_DIR:-}" ]; then
        printf '%s\n' "$PIDSMAKER_DIR"
        return 0
    fi

    local candidate
    for candidate in \
        "${DEFAULT_PIDSMAKER_ROOT}/PIDSMaker-hyp" \
        "${DEFAULT_PIDSMAKER_ROOT}/PIDSMaker"
    do
        if [ -d "$candidate/config" ] && [ -f "$candidate/pidsmaker/main.py" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}

if ! PIDSMAKER_DIR="$(resolve_pidsmaker_dir)"; then
    echo "[run.sh] Error: could not find PIDSMaker checkout."
    echo "[run.sh] Set PIDSMAKER_DIR or place the repo at one of:"
    echo "  ${DEFAULT_PIDSMAKER_ROOT}/PIDSMaker-hyp"
    echo "  ${DEFAULT_PIDSMAKER_ROOT}/PIDSMaker"
    exit 1
fi

CONFIG_SRC="${SCRIPT_DIR}/configs/hyp_pids.yml"
CONFIG_DST="${PIDSMAKER_DIR}/config/hyp_pids.yml"

# --- Usage ---
if [ $# -lt 1 ]; then
    echo "Usage: $0 DATASET [extra args...]"
    echo ""
    echo "Examples:"
    echo "  $0 THEIA_E5"
    echo "  $0 THEIA_E5 --wandb"
    echo "  $0 THEIA_E5 --force_restart=training"
    echo "  $0 THEIA_E5 --cpu"
    echo "  $0 THEIA_E5 --training.lr=0.001 --training.num_epochs=50"
    exit 1
fi

DATASET="$1"
shift

# --- Symlink config into PIDSMaker ---
if [ -L "$CONFIG_DST" ]; then
    # Already a symlink — update if target changed
    if [ "$(readlink -f "$CONFIG_DST")" != "$(readlink -f "$CONFIG_SRC")" ]; then
        ln -sf "$CONFIG_SRC" "$CONFIG_DST"
        echo "[run.sh] Updated symlink: $CONFIG_DST -> $CONFIG_SRC"
    fi
elif [ -e "$CONFIG_DST" ]; then
    echo "[run.sh] Error: $CONFIG_DST exists and is not a symlink. Remove it manually to proceed."
    exit 1
else
    ln -s "$CONFIG_SRC" "$CONFIG_DST"
    echo "[run.sh] Created symlink: $CONFIG_DST -> $CONFIG_SRC"
fi

# --- Run PIDSMaker ---
# Add PIDSMaker root to PYTHONPATH so `import pidsmaker` resolves correctly
# (running `python pidsmaker/main.py` puts pidsmaker/ on sys.path, not its parent)
cd "$PIDSMAKER_DIR"
export PYTHONPATH="${PIDSMAKER_DIR}:${SCRIPT_DIR}:${PYTHONPATH:-}"
exec python pidsmaker/main.py hyp_pids "$DATASET" "$@"
