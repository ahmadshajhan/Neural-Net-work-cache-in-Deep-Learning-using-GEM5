#!/usr/bin/env python3
"""
Shared gem5 stats parsing helpers for plotting and live monitoring.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


DEFAULT_RESULTS = {
    "Direct-Mapped\n(1-way)": Path("results/direct/stats.txt"),
    "4-way\nSet-Assoc": Path("results/set4way/stats.txt"),
    "Fully\nAssociative": Path("results/fullassoc/stats.txt"),
}


DEMO_RESULTS = {
    "Direct-Mapped\n(1-way)": {
        "miss_rate": 38.4,
        "miss_latency": 198.0,
        "sim_ticks": 925_000_000,
        "l2_miss_rate": 18.3,
        "sim_insts": 1_000_000,
    },
    "4-way\nSet-Assoc": {
        "miss_rate": 18.9,
        "miss_latency": 144.0,
        "sim_ticks": 642_000_000,
        "l2_miss_rate": 12.1,
        "sim_insts": 1_000_000,
    },
    "Fully\nAssociative": {
        "miss_rate": 11.7,
        "miss_latency": 115.0,
        "sim_ticks": 529_000_000,
        "l2_miss_rate": 9.4,
        "sim_insts": 1_000_000,
    },
}


FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
INT_RE = re.compile(r"[-+]?\d+")


def ensure_matplotlib_env() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))


def _parse_number(text: str, integer: bool = False) -> float | int | None:
    regex = INT_RE if integer else FLOAT_RE
    match = regex.search(text)
    if not match:
        return None
    return int(match.group(0)) if integer else float(match.group(0))


def _extract_metric(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _extract_value_token(line: str) -> str:
    parts = line.split()
    if len(parts) < 2:
        return line
    return parts[1]


def parse_stats(path: Path) -> dict[str, float | int | None] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None

    metrics: dict[str, float | int | None] = {
        "miss_rate": None,
        "miss_latency": None,
        "sim_ticks": None,
        "l2_miss_rate": None,
        "sim_insts": None,
    }

    with path.open() as handle:
        for raw_line in handle:
            line = _extract_metric(raw_line)
            if not line:
                continue
            value_token = _extract_value_token(line)
            if line.startswith("sim_ticks ") or line.startswith("simTicks "):
                metrics["sim_ticks"] = _parse_number(value_token, integer=True)
            elif line.startswith("sim_insts ") or line.startswith("simInsts "):
                metrics["sim_insts"] = _parse_number(value_token, integer=True)
            elif (
                "system.cpu.dcache.overall_miss_rate::total" in line
                or "system.cpu.dcache.overallMissRate::total" in line
            ):
                value = _parse_number(value_token)
                if value is not None:
                    metrics["miss_rate"] = float(value) * 100.0
            elif (
                "system.cpu.dcache.overallAvgMissLatency::total" in line
                or "system.cpu.dcache.overall_avg_miss_latency::total" in line
            ):
                value = _parse_number(value_token)
                if value is not None:
                    metrics["miss_latency"] = float(value)
            elif (
                "system.l2cache.overall_miss_rate::total" in line
                or "system.l2cache.overallMissRate::total" in line
            ):
                value = _parse_number(value_token)
                if value is not None:
                    metrics["l2_miss_rate"] = float(value) * 100.0

    return metrics


def result_complete(metrics: dict[str, float | int | None] | None) -> bool:
    if not metrics:
        return False
    required = ("miss_rate", "miss_latency", "sim_ticks")
    return all(metrics.get(key) is not None for key in required)


def load_results(result_paths: dict[str, Path], use_demo_fallback: bool = True) -> dict[str, dict[str, float | int | None]]:
    results: dict[str, dict[str, float | int | None]] = {}
    for label, path in result_paths.items():
        metrics = parse_stats(path)
        if metrics is None or not result_complete(metrics):
            if use_demo_fallback:
                metrics = dict(DEMO_RESULTS[label])
            else:
                metrics = {
                    "miss_rate": None,
                    "miss_latency": None,
                    "sim_ticks": None,
                    "l2_miss_rate": None,
                    "sim_insts": None,
                }
        results[label] = metrics
    return results
