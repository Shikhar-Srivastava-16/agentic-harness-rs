#!/usr/bin/env bash
#
# run_comparison.sh — Orchestrate a benchmark comparison between the
# Rust `llms` crate and Python LangChain.
#
# Prerequisites:
#   1. Ollama server running (ollama serve)
#   2. Model pulled (e.g. ollama pull qwen3:8b)
#   3. uv installed (https://docs.astral.sh/uv/)
#  4. .env file created from this directory's .env.example
#      (cp .env.example .env) — required when run from benchmarks/.
#
# Usage:
#   ./benchmarks/analysis/run_comparison.sh [all|rust|python|compare]
#
# All settings come from the .env file next to this script.  Individual
# variables can also be overridden via the shell environment.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# --- Load .env (if present) --------------------------------------------------

if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env"
    set +a
fi

# --- Parameters (with fallbacks) ---------------------------------------------

ITERATIONS="${BENCH_ITERATIONS:-10}"
MODEL="${BENCH_MODEL:-gemma4:e4b}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
LLMS_OUTPUT="${LLMS_BENCH_OUTPUT:-$ROOT_DIR/bench_llms.jsonl}"
LANGCHAIN_OUTPUT="${LANGCHAIN_BENCH_OUTPUT:-$ROOT_DIR/bench_langchain.jsonl}"
CHART_OUTPUT="${COMPARE_CHART:-$ROOT_DIR/comparison.png}"

# --- Helpers -----------------------------------------------------------------

check_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "ERROR: '$1' not found. $2"
        exit 1
    }
}

require_ollama() {
    check_cmd ollama "Install from https://ollama.com"
    if ! ollama ps >/dev/null 2>&1; then
        echo "ERROR: Ollama server is not running. Start it with: ollama serve"
        exit 1
    fi
    if ! ollama list 2>/dev/null | grep -q "$MODEL"; then
        echo "ERROR: Model '$MODEL' is not pulled. Run: ollama pull $MODEL"
        exit 1
    fi
}

require_uv() {
    check_cmd uv "Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
}

# --- Step functions ----------------------------------------------------------

run_rust() {
    require_ollama
    echo "=== Running Rust llms harness ==="
    LLMS_BENCH_OUTPUT="$LLMS_OUTPUT" \
        BENCH_ITERATIONS="$ITERATIONS" \
        BENCH_MODEL="$MODEL" \
        OLLAMA_URL="$OLLAMA_URL" \
        cargo run --manifest-path "$SCRIPT_DIR/../rust/Cargo.toml" --release 2>&1 || {
        echo "ERROR: Rust harness failed."
        exit 1
    }
    echo ""
}

run_python() {
    require_ollama
    require_uv
    echo "=== Running Python LangChain harness ==="
    BENCH_ITERATIONS="$ITERATIONS" \
        BENCH_MODEL="$MODEL" \
        OLLAMA_URL="$OLLAMA_URL" \
        LANGCHAIN_BENCH_OUTPUT="$LANGCHAIN_OUTPUT" \
        uv run --project "$SCRIPT_DIR/../python" \
        python "$SCRIPT_DIR/../python/harness_langchain.py" 2>&1 || {
        echo "ERROR: Python harness failed."
        exit 1
    }
    echo ""
}

run_compare() {
    require_uv

    if [ ! -f "$LLMS_OUTPUT" ]; then
        echo "ERROR: Rust harness output not found at $LLMS_OUTPUT"
        echo "       Run './benchmarks/analysis/run_comparison.sh rust' first."
        exit 1
    fi
    if [ ! -f "$LANGCHAIN_OUTPUT" ]; then
        echo "ERROR: Python harness output not found at $LANGCHAIN_OUTPUT"
        echo "       Run './benchmarks/analysis/run_comparison.sh python' first."
        exit 1
    fi

    echo "=== Generating comparison report ==="
    uv run --project "$SCRIPT_DIR/../python" \
        python "$SCRIPT_DIR/compare.py" \
        --llms "$LLMS_OUTPUT" \
        --langchain "$LANGCHAIN_OUTPUT" \
        --output "$CHART_OUTPUT" 2>&1
    echo ""
    echo "============================================"
    echo " Comparison complete!"
    echo "============================================"
    echo "Rust output:    $LLMS_OUTPUT"
    echo "Python output:  $LANGCHAIN_OUTPUT"
    echo "Chart:          $CHART_OUTPUT"
}

# --- Dispatch ----------------------------------------------------------------

STEP="${1:-all}"

case "$STEP" in
    all)
        echo "============================================"
        echo " llms vs LangChain Benchmark Comparison"
        echo "============================================"
        echo "Model:      $MODEL"
        echo "Iterations: $ITERATIONS"
        echo "Ollama URL: $OLLAMA_URL"
        echo "Rust out:   $LLMS_OUTPUT"
        echo "Python out: $LANGCHAIN_OUTPUT"
        echo "Chart:      $CHART_OUTPUT"
        echo ""

        run_rust
        run_python
        run_compare
        ;;
    rust)
        run_rust
        ;;
    python)
        run_python
        ;;
    compare)
        run_compare
        ;;
    *)
        echo "Usage: $0 [all|rust|python|compare]"
        echo ""
        echo "Arguments:"
        echo "  all       Run all three steps (default)"
        echo "  rust      Run only the Rust llms harness"
        echo "  python    Run only the Python LangChain harness"
        echo "  compare   Run only the comparison script"
        exit 1
        ;;
esac
