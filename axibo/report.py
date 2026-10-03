"""Aggregate and print evaluation results.

Shared by `eval_policy.py` (prints at the end of a run) and `eval_report.py` (re-prints a saved run, or compares
several). Both read the same per-trial rows, so the tables cannot drift apart.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

NUMERIC = (
    "trial", "seed", "layout_id", "chunks", "steps", "max_src_z_mm", "lift_threshold_mm", "stack_height_mm",
    "final_xy_offset_mm", "final_dz_error_mm", "final_src_speed_mm_s", "final_settle_disp_mm", "final_gripper_mm",
    "dst_tilt_deg", "dst_moved_mm", "third_moved_mm", "latency_p50_ms", "clipped_steps",
    "release_dz_mm", "release_xy_offset_mm", "release_speed_mm_s",
)
BOOLEAN = ("third_disturbed", "dst_toppled")


def load_run(path: str | Path) -> tuple[list[dict], dict]:
    """(rows, meta) for one eval output directory (or a direct path to its eval_log.csv)."""
    path = Path(path)
    log = path if path.suffix == ".csv" else path / "eval_log.csv"
    meta_path = log.parent / "summary.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    rows = []
    for r in csv.DictReader(log.open()):
        for k in NUMERIC:
            if r.get(k) not in (None, ""):
                r[k] = float(r[k]) if "." in r[k] else int(r[k])
        for k in BOOLEAN:
            r[k] = str(r.get(k)).lower() == "true"
        rows.append(r)
    return rows, meta


def _rate(rows: list[dict]) -> float:
    return sum(r["outcome"] == "success" for r in rows) / max(1, len(rows))


def group_by(rows: list[dict], key_fn) -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault(key_fn(r), []).append(r)
    return out


def summarise(rows: list[dict]) -> dict:
    """Aggregates used both for the printed tables and for summary.json."""
    by_pair = group_by(rows, lambda r: (r["source"], r["destination"]))
    by_layout = group_by(rows, lambda r: int(r["layout_id"]))
    lat = np.array([r["latency_p50_ms"] for r in rows], dtype=float)
    return {
        "trials": len(rows),
        "success_rate": _rate(rows),
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "flags": {
            "third_disturbed": sum(r["third_disturbed"] for r in rows),
            "dst_toppled": sum(r["dst_toppled"] for r in rows),
        },
        "per_pair": {
            f"{s} -> {d}": {
                "n": len(rs),
                "success": sum(r["outcome"] == "success" for r in rs),
                "success_rate": _rate(rs),
                "seeds": sorted({int(r["seed"]) for r in rs}),
                "outcomes": dict(Counter(r["outcome"] for r in rs)),
            }
            for (s, d), rs in by_pair.items()
        },
        "per_layout": {
            str(lid): {
                "n": len(rs),
                "success": sum(r["outcome"] == "success" for r in rs),
                "success_rate": _rate(rs),
                "outcomes": dict(Counter(r["outcome"] for r in rs)),
                "failed_pairs": [f"{r['source']} -> {r['destination']}" for r in rs if r["outcome"] != "success"],
            }
            for lid, rs in sorted(by_layout.items())
        },
        "latency_ms": {"p50": float(np.percentile(lat, 50)), "p99": float(np.percentile(lat, 99))} if len(lat) else {},
    }


def _bar(rate: float, width: int = 10) -> str:
    filled = int(round(rate * width))
    return "#" * filled + "." * (width - filled)


def _table(title: str, groups: dict, label_fn, width: int = 40):
    print(f"\n  {title:{width}} {'n':>3} {'success':>8} {'':12} outcomes")
    for key, rs in sorted(groups.items(), key=lambda kv: (_rate(kv[1]), str(kv[0]))):
        detail = ", ".join(f"{k} {v}" for k, v in Counter(r["outcome"] for r in rs).most_common() if k != "success")
        print(f"  {label_fn(key):{width}} {len(rs):3d} {_rate(rs):7.0%} {_bar(_rate(rs)):12} {detail}")


def print_run(rows: list[dict], title: str = "", meta: dict | None = None):
    """The per-pair and per-layout tables, worst first, plus totals, flags and latency."""
    if not rows:
        print("no trials")
        return
    s = summarise(rows)
    meta = meta or {}
    print(f"\n=== {title or 'eval'} ===")
    if meta:
        seeds = meta.get("seeds", sorted({int(r["seed"]) for r in rows}))
        print(f"  checkpoint {meta.get('checkpoint', '?')}")
        print(f"  layouts {meta.get('layouts', '?')} | pairs {meta.get('pairs_mode', '?')} | seeds {seeds} | "
              f"{meta.get('exec_steps', '?')}/{meta.get('chunk_size', '?')} per chunk, "
              f"max {meta.get('max_chunks', '?')} chunks + {meta.get('settle_s', '?')} s settle")

    _table("pair (worst first)", group_by(rows, lambda r: (r["source"], r["destination"])), lambda k: f"{k[0]} -> {k[1]}")
    _table("layout (worst first)", group_by(rows, lambda r: int(r["layout_id"])), lambda k: f"layout {k}")

    print(f"\n  {'TOTAL':40} {s['trials']:3d} {s['success_rate']:7.0%} {_bar(s['success_rate']):12}")
    print("  outcomes: " + ", ".join(f"{k} {v}" for k, v in Counter(r["outcome"] for r in rows).most_common()))
    print(f"  flags: third_disturbed {s['flags']['third_disturbed']}, dst_toppled {s['flags']['dst_toppled']}")
    if s["latency_ms"]:
        print(f"  latency per chunk: p50 {s['latency_ms']['p50']:.0f} ms, p99 {s['latency_ms']['p99']:.0f} ms")


def print_comparison(runs: list[tuple[str, list[dict]]]):
    """Success rate per pair across several runs, e.g. checkpoints of the same training run."""
    names = [n for n, _ in runs]
    pairs = sorted({f"{r['source']} -> {r['destination']}" for _, rows in runs for r in rows})
    w = max(len(p) for p in pairs) + 2
    print(f"\n=== comparison ===\n  {'pair':{w}}" + "".join(f"{n:>16}" for n in names))
    for pair in pairs:
        cells = []
        for _, rows in runs:
            rs = [r for r in rows if f"{r['source']} -> {r['destination']}" == pair]
            cells.append(f"{_rate(rs):>13.0%}({len(rs)})" if rs else f"{'-':>16}")
        print(f"  {pair:{w}}" + "".join(cells))
    totals = "".join(f"{_rate(rows):>13.0%}({len(rows)})" for _, rows in runs)
    print(f"  {'TOTAL':{w}}" + totals)
