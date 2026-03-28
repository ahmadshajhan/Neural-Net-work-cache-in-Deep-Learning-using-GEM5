#!/usr/bin/env python3
"""
Generate cache comparison plots from gem5 stats.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from model_utils import load_model_metrics
from stats_utils import DEFAULT_RESULTS, ensure_matplotlib_env, load_results

ensure_matplotlib_env()

import matplotlib.pyplot as plt


COLORS = ["#E74C3C", "#3498DB", "#2ECC71"]


def build_plot(
    results: dict[str, dict[str, float | int | None]],
    model_metrics: dict[str, object] | None,
    output: Path,
) -> None:
    labels = list(results.keys())
    miss_rates = [float(results[label]["miss_rate"]) for label in labels]
    sim_ticks = [int(results[label]["sim_ticks"]) for label in labels]
    ticks_m = [tick / 1e6 for tick in sim_ticks]
    baseline = sim_ticks[0]
    speedups = [baseline / tick if tick else 0.0 for tick in sim_ticks]
    accuracy_labels = ["Train", "Test", "GEM5 Batch"]
    accuracy_values = [0.0, 0.0, 0.0]
    if model_metrics:
        accuracy_values = [
            float(model_metrics.get("train_accuracy", 0.0)) * 100.0,
            float(model_metrics.get("test_accuracy", 0.0)) * 100.0,
            float(model_metrics.get("gem5_subset_accuracy", 0.0)) * 100.0,
        ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    fig.suptitle(
        "Neural-Network Cache and Model Dashboard\nPizza / Steak / Sushi Dataset  |  gem5 + Linear Classifier",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    ax = axes[0]
    bars = ax.bar(labels, miss_rates, color=COLORS, width=0.5, edgecolor="white", linewidth=1.2)
    ax.set_title("L1 Data Cache Miss Rate (%)", fontweight="bold")
    ax.set_ylabel("Miss Rate (%)")
    ax.set_ylim(0, max(miss_rates) * 1.25 if miss_rates else 1)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.tick_params(axis="x", labelsize=9)
    for bar, value in zip(bars, miss_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax = axes[1]
    bars = ax.bar(labels, ticks_m, color=COLORS, width=0.5, edgecolor="white", linewidth=1.2)
    ax.set_title("Simulation Ticks (Millions)", fontweight="bold")
    ax.set_ylabel("Ticks (M)")
    ax.set_ylim(0, max(ticks_m) * 1.25 if ticks_m else 1)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.tick_params(axis="x", labelsize=9)
    label_offset = max(ticks_m) * 0.01 if ticks_m else 0.1
    for bar, value in zip(bars, ticks_m):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            f"{value:.2f}M",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax = axes[2]
    bars = ax.bar(labels, speedups, color=COLORS, width=0.5, edgecolor="white", linewidth=1.2)
    ax.set_title("Speedup vs Direct-Mapped", fontweight="bold")
    ax.set_ylabel("Speedup (x)")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylim(0, max(speedups) * 1.25 if speedups else 1)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.tick_params(axis="x", labelsize=9)
    for bar, value in zip(bars, speedups):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.3f}x",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax = axes[3]
    acc_colors = ["#1F7A8C", "#BF4342", "#6A994E"]
    bars = ax.bar(accuracy_labels, accuracy_values, color=acc_colors, width=0.55, edgecolor="white", linewidth=1.2)
    ax.set_title("Classifier Accuracy (%)", fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, max(accuracy_values) * 1.2 if any(accuracy_values) else 1)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    for bar, value in zip(bars, accuracy_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")


def print_summary(
    results: dict[str, dict[str, float | int | None]],
    model_metrics: dict[str, object] | None,
) -> None:
    labels = list(results.keys())
    sim_ticks = [int(results[label]["sim_ticks"]) for label in labels]
    baseline = sim_ticks[0]

    print("\n" + "=" * 72)
    print(f"{'Cache Config':<22} {'Miss Rate':>10} {'Latency':>10} {'Ticks(M)':>10} {'Speedup':>9}")
    print("-" * 72)
    for label in labels:
        metrics = results[label]
        name = label.replace("\n", " ")
        miss_rate = float(metrics["miss_rate"])
        miss_latency = float(metrics["miss_latency"])
        ticks = int(metrics["sim_ticks"])
        speedup = baseline / ticks if ticks else 0.0
        print(
            f"{name:<22} {miss_rate:>9.2f}% {miss_latency:>10.2f} "
            f"{ticks / 1e6:>10.2f} {speedup:>8.3f}x"
        )
    print("=" * 72)
    if model_metrics:
        print(
            "Model accuracy: "
            f"train={float(model_metrics['train_accuracy']) * 100:.2f}%  "
            f"test={float(model_metrics['test_accuracy']) * 100:.2f}%  "
            f"gem5_batch={float(model_metrics['gem5_subset_accuracy']) * 100:.2f}%"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="plots/cache_analysis.png",
        help="Where to save the generated plot image.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip opening a plot window even when a display is available.",
    )
    args = parser.parse_args()

    results = load_results(DEFAULT_RESULTS)
    model_metrics = load_model_metrics()
    build_plot(results, model_metrics, Path(args.output))
    print_summary(results, model_metrics)

    if not args.no_show and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
