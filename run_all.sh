#!/bin/bash
# Complete gem5 neural-network cache analysis pipeline with optional live monitor.

set -euo pipefail

GEM5="${GEM5:-$HOME/gem5/build/RISCV/gem5.opt}"
PROJECT="$(cd "$(dirname "$0")" && pwd)"
VENV_PY="$PROJECT/venv/bin/python"
STATUS_FILE="$PROJECT/results/live_status.json"
MONITOR_PID=""

write_status() {
    local state="$1"
    local step="$2"
    local current_config="${3:-}"
    local details="${4:-}"

    STATE="$state" STEP="$step" CURRENT_CONFIG="$current_config" DETAILS="$details" STATUS_FILE="$STATUS_FILE" "$VENV_PY" - <<'PY'
import json
import os
import time
from pathlib import Path

status_path = Path(os.environ["STATUS_FILE"])
status_path.parent.mkdir(parents=True, exist_ok=True)

started_at = time.time()
if status_path.exists():
    try:
        started_at = json.loads(status_path.read_text()).get("started_at", started_at)
    except Exception:
        pass

payload = {
    "state": os.environ["STATE"],
    "step": os.environ["STEP"],
    "current_config": os.environ["CURRENT_CONFIG"],
    "details": os.environ["DETAILS"],
    "started_at": started_at,
    "updated_at": time.time(),
}
status_path.write_text(json.dumps(payload, indent=2))
PY
}

cleanup() {
    if [ "${KEEP_MONITOR_OPEN:-1}" = "1" ]; then
        return
    fi
    if [ -n "$MONITOR_PID" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
        kill "$MONITOR_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT

can_launch_gui() {
    if [ "${ENABLE_LIVE_MONITOR:-auto}" = "0" ]; then
        return 1
    fi
    if [ "${ENABLE_LIVE_MONITOR:-auto}" = "1" ]; then
        return 0
    fi
    if [ -n "${DISPLAY:-}" ] && command -v xdpyinfo >/dev/null 2>&1; then
        xdpyinfo >/dev/null 2>&1
        return $?
    fi
    return 1
}

if [ ! -x "$VENV_PY" ]; then
    echo "ERROR: virtualenv Python not found at $VENV_PY"
    echo "Create it first, then install: PyQt5 matplotlib numpy pillow requests"
    exit 1
fi

if [ ! -f "$GEM5" ]; then
    echo "ERROR: gem5 not found at $GEM5"
    echo "Build it first: cd ~/gem5 && scons build/RISCV/gem5.opt -j2"
    exit 1
fi

if ! command -v riscv64-linux-gnu-g++ >/dev/null 2>&1; then
    echo "ERROR: riscv64-linux-gnu-g++ is not installed"
    exit 1
fi

cd "$PROJECT"
mkdir -p .mplconfig results/direct results/set4way results/fullassoc plots
export MPLCONFIGDIR="$PROJECT/.mplconfig"

echo "Working dir: $(pwd)"
echo ""
echo "============================================"
echo "  gem5 Neural-Network Cache Analysis"
echo "  Dataset: Pizza / Steak / Sushi"
echo "  Machine: CPU-only (no GPU required)"
echo "============================================"
echo ""

write_status "starting" "Checking Python packages"
"$VENV_PY" - <<'PY'
import importlib
mods = ["PyQt5", "matplotlib", "numpy", "PIL", "requests"]
missing = []
for mod in mods:
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(mod)
if missing:
    raise SystemExit("Missing Python modules in venv: " + ", ".join(missing))
print("Python packages OK:", ", ".join(mods))
PY

if can_launch_gui; then
    write_status "starting" "Launching live monitor"
    "$VENV_PY" live_monitor.py >/tmp/nn_cache_live_monitor.log 2>&1 &
    MONITOR_PID="$!"
    echo "Live monitor started (PID $MONITOR_PID)"
else
    echo "No reachable GUI display detected, so live PyQt5 monitor is skipped."
fi

echo "[1/6] Downloading pizza/steak/sushi dataset..."
write_status "running" "Downloading dataset"
"$VENV_PY" download_dataset.py
echo ""

echo "[2/6] Extracting image matrices and training classifier..."
write_status "running" "Extracting image matrices and training classifier"
"$VENV_PY" extract_matrices.py
echo ""

echo "[3/6] Compiling RISC-V binary..."
write_status "running" "Compiling RISC-V binary"
riscv64-linux-gnu-g++ -O0 -static -o matmul_food matmul_food.cpp
echo "  Binary ready: $PROJECT/matmul_food"
echo ""

run_gem5_case() {
    local name="$1"
    local assoc="$2"
    local outdir="$3"

    rm -f "$outdir/stats.txt" "$outdir/gem5.log"
    echo "  Running $name (assoc=$assoc)..."
    write_status "running" "Running gem5 simulation" "$name" "assoc=$assoc"
    "$GEM5" --listener-mode=off --outdir="$outdir" gem5_food_config.py --assoc="$assoc" >"$outdir/gem5.log" 2>&1
    if [ ! -s "$outdir/stats.txt" ]; then
        echo "ERROR: gem5 did not produce stats for $name"
        tail -n 20 "$outdir/gem5.log" || true
        exit 1
    fi
    local ticks
    ticks="$(awk '/^(sim_ticks|simTicks) / {print $2}' "$outdir/stats.txt" | head -n 1)"
    echo "       Done. sim_ticks=$ticks"
}

echo "[4/6] Running gem5 simulations..."
run_gem5_case "Direct-Mapped" "1" "$PROJECT/results/direct"
run_gem5_case "4-way Set-Associative" "4" "$PROJECT/results/set4way"
run_gem5_case "Fully Associative" "512" "$PROJECT/results/fullassoc"
echo ""

echo "[5/6] Parsing results and generating plots..."
write_status "running" "Generating plots"
"$VENV_PY" parse_and_plot.py --no-show
echo ""

write_status "finished" "Pipeline complete" "" "Plots saved to plots/cache_analysis.png"
echo "[6/6] All done!"
echo ""
echo "  Plot image: $PROJECT/plots/cache_analysis.png"
echo "  Model:      $PROJECT/results/model_metrics.json"
echo "  Stats:      $PROJECT/results/{direct,set4way,fullassoc}/stats.txt"
echo "  GEM5 logs:  $PROJECT/results/{direct,set4way,fullassoc}/gem5.log"
