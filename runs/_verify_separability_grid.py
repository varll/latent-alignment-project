"""Standalone (Agg) verifier for separability_grid.py — the Jupyter kernel can't run here.

Exercises the exact plotting/saving logic over the real CSVs, then confirms ALL expected PNGs are
on disk: 4 combined 2x2 grids + 16 standalone per-panel files (sep_grid_<metric>_<dataset>.png).
Prints the present/missing breakdown so the missing-model detection can be checked without a
notebook kernel. Safe to delete.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import separability_grid as G  # noqa: E402

DPI = 300


def main() -> None:
    figdir = G.RUNS / "figs"
    results = G.render_all(dpi=DPI)

    expected = []
    for metric in G.METRIC_FNAME:
        expected.append(figdir / G.METRIC_FNAME[metric])
        for dskey in G.DATASET_ORDER:
            expected.append(figdir / f"sep_grid_{metric}_{G.DATASET_SLUG[dskey]}.png")

    print(f"=== present / missing per (metric, dataset) — expected = {len(G.EXPECTED)} models ===")
    for metric, info in results.items():
        print(f"\n[{metric}] combined -> {Path(info['combined']).name}")
        for dskey in G.DATASET_ORDER:
            p = info["panels"][dskey]
            print(f"   {dskey:16} -> {Path(p['path']).name}")
            print(f"      present ({len(p['present'])}): {p['present']}")
            print(f"      missing ({len(p['missing'])}): {p['missing']}")

    missing_files = [str(p) for p in expected if not p.exists()]
    print(f"\nexpected {len(expected)} PNGs (4 combined + 16 panels); "
          f"on disk: {len(expected) - len(missing_files)}")
    if missing_files:
        raise SystemExit(f"MISSING FILES: {missing_files}")
    print("OK — all 20 PNGs written, no exceptions.")


if __name__ == "__main__":
    main()
