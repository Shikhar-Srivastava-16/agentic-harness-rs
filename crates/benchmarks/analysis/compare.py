#!/usr/bin/env python3
"""
Compare benchmark results between the Rust `llms` crate and Python LangChain.

Reads two JSONL files and emits:
  1. A text table with per-metric statistics (count, mean, P50, P95, P99)
  2. A PNG bar chart (benchmarks/comparison.png)

Usage:
    python compare.py [--llms LLMS_JSONL] [--langchain LANGCHAIN_JSONL] [--output OUTPUT_PNG]
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_LLMS = os.environ.get("LLMS_BENCH_OUTPUT", "benchmarks/bench_llms.jsonl")
DEFAULT_LANGCHAIN = os.environ.get(
    "LANGCHAIN_BENCH_OUTPUT", "benchmarks/bench_langchain.jsonl"
)
DEFAULT_CHART = os.environ.get("COMPARE_CHART", "benchmarks/comparison.png")


def load_jsonl(path: str, framework: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"WARNING: {path} not found — skipping {framework}", file=sys.stderr)
        return pd.DataFrame()
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["framework"] = framework
            records.append(rec)
    df = pd.DataFrame(records)
    if not df.empty:
        df["duration_ms"] = pd.to_numeric(df["duration_ms"], errors="coerce")
    return df


def percentiles(series: pd.Series) -> dict[str, float]:
    s = series.dropna()
    if len(s) == 0:
        return {"count": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0}
    return {
        "count": len(s),
        "mean": round(float(s.mean()), 3),
        "p50": round(float(s.quantile(0.50)), 3),
        "p95": round(float(s.quantile(0.95)), 3),
        "p99": round(float(s.quantile(0.99)), 3),
    }


def print_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("No data to compare.")
        return

    kinds = ["query", "send", "tool_time", "tool_cycle"]
    print()
    print("=" * 96)
    print("BENCHMARK COMPARISON: llms crate vs Python LangChain")
    print("=" * 96)

    for kind in kinds:
        sub = df[df["kind"] == kind]
        if sub.empty:
            continue

        tools = sub["tool"].dropna().unique() if "tool" in sub.columns else []
        if len(tools) > 0:
            for tool_name in tools:
                tsub = sub[(sub["tool"] == tool_name)]
                _print_metric_table(kind, tsub, label=f" (tool={tool_name})")
        else:
            _print_metric_table(kind, sub)

    print()
    print("Note: LangChain's AgentExecutor batches multiple tool calls into a")
    print("single follow-up LLM call. The `llms` crate sends one LLM call per")
    print("tool. For multi-tool turns, LangChain may produce fewer tool_cycle")
    print("records. Tool_time durations are comparable directly.")
    print("=" * 96)


def _print_metric_table(kind: str, sub: pd.DataFrame, label: str = "") -> None:
    print(f"\n  {kind.upper()}{label}")
    print(f"  {'-' * 70}")
    print(
        f"  {'Framework':<16} {'Count':>6} {'Mean':>10} {'P50':>10} {'P95':>10} {'P99':>10}  (ms)"
    )
    print(f"  {'-' * 70}")
    for fw in ["llms", "langchain"]:
        fw_sub = sub[sub["framework"] == fw]
        if fw_sub.empty:
            print(f"  {fw:<16} {'—':>6} {'—':>10} {'—':>10} {'—':>10} {'—':>10}")
            continue
        stats = percentiles(fw_sub["duration_ms"])
        print(
            f"  {fw:<16} {stats['count']:>6} {stats['mean']:>10.3f} {stats['p50']:>10.3f} "
            f"{stats['p95']:>10.3f} {stats['p99']:>10.3f}"
        )

    # llms_sub = sub[sub["framework"] == "llms"]
    # lc_sub = sub[sub["framework"] == "langchain"]
    # if not llms_sub.empty and not lc_sub.empty:
    #     llms_mean = llms_sub["duration_ms"].mean()
    #     lc_mean = lc_sub["duration_ms"].mean()
    #     if llms_mean > 0:
    #         delta = ((lc_mean - llms_mean) / llms_mean) * 100
    #         faster = "faster" if delta < 0 else "slower"


def generate_chart(df: pd.DataFrame, output: str) -> None:
    if df.empty:
        print("No data for chart.")
        return

    kinds = ["query", "send", "tool_time", "tool_cycle"]
    titles = ["Query (total prompt)", "Send (prompt → HTTP)", "Tool Time", "Tool Cycle"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "llms crate vs Python LangChain — Benchmark Comparison",
        fontsize=14,
        fontweight="bold",
    )

    x = np.arange(2)
    width = 0.35
    colors = {"llms": "#2563eb", "langchain": "#dc2626"}

    for idx, (kind, title) in enumerate(zip(kinds, titles)):
        ax = axes[idx // 2][idx % 2]
        sub = df[df["kind"] == kind]

        tools = sorted(sub["tool"].dropna().unique()) if "tool" in sub.columns else []
        if not tools:
            tools = ["(all)"]

        means = []
        errors = []
        labels = []
        for fw in ["llms", "langchain"]:
            fw_sub = sub[(sub["framework"] == fw)]
            if tools[0] != "(all)":
                fw_sub = fw_sub[fw_sub["tool"] == tools[0]]
            if fw_sub.empty:
                means.append(0)
                errors.append(0)
            else:
                means.append(fw_sub["duration_ms"].mean())
                errors.append(fw_sub["duration_ms"].std() if len(fw_sub) > 1 else 0)
            labels.append(fw)

        bars = ax.bar(
            x,
            means,
            width * 2,
            yerr=errors,
            color=[colors[color] for color in labels],
            capsize=5,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(["llms", "LangChain"], fontsize=11)
        ax.set_ylabel("Duration (ms)", fontsize=10)
        ax.set_title(
            f"{title}" + (f" — tool: {tools[0]}" if tools[0] != "(all)" else ""),
            fontsize=11,
            fontweight="bold",
        )

        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{mean:.1f}ms",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    # Collect all unique tool names for tool_time/tool_cycle subplots
    all_tools = sorted(
        df[df["kind"].isin(["tool_time", "tool_cycle"])]["tool"].dropna().unique()
    )
    if all_tools:
        legend_text = "Tools: " + ", ".join(all_tools)
        fig.text(0.5, 0.01, legend_text, ha="center", fontsize=9, style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"\nChart saved to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare llms vs LangChain benchmarks")
    parser.add_argument(
        "--llms", default=DEFAULT_LLMS, help="Path to llms JSONL output"
    )
    parser.add_argument(
        "--langchain", default=DEFAULT_LANGCHAIN, help="Path to LangChain JSONL output"
    )
    parser.add_argument(
        "--output", default=DEFAULT_CHART, help="Output PNG path for chart"
    )
    args = parser.parse_args()

    df_llms = load_jsonl(args.llms, "llms")
    df_lc = load_jsonl(args.langchain, "langchain")
    df = pd.concat([df_llms, df_lc], ignore_index=True)

    if df.empty:
        print("No benchmark data found. Run both harnesses first.")
        print(f"  llms:      {args.llms}")
        print(f"  langchain: {args.langchain}")
        sys.exit(1)

    print_table(df)
    generate_chart(df, args.output)


if __name__ == "__main__":
    main()
