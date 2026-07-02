"""High-accuracy layer positions for PA-CCS — where does the latent signal live?

This quantifies the recurring thesis that the PA-CCS separability signal sits in the
**mid-to-late** layers (never in the very first layers, often a long plateau to the top).

For each model in :data:`analysis.MODELS` (loaded via :func:`analysis.load_ccs`) we define a
per-layer **corrected accuracy** ``max(acc, 1 - acc)`` (CCS is sign-ambiguous) and a model's
**high-accuracy band**: every layer within :data:`HIGH_DELTA` of that model's best corrected
accuracy. The table reports how big that band is, how deep it sits, and how accurate it is.

Pure pandas/numpy so it can be unit-tested and re-run offline. Run as a script to print the
table and write ``runs/summary_high_layers.csv``.
"""

from __future__ import annotations

from pathlib import Path

import analysis as A
import numpy as np
import pandas as pd

RUNS = Path(__file__).resolve().parent

# A layer counts as "high-accuracy" if its corrected accuracy is within this much of the
# model's best corrected accuracy. Named so the notebook/FINDINGS can quote it once.
HIGH_DELTA = 0.03


def high_layer_row(run: A.ModelRun, df: pd.DataFrame, *, delta: float = HIGH_DELTA) -> dict:
    """One model's high-accuracy-layer summary from its per-layer PA-CCS table.

    ``df`` is the output of :func:`analysis.load_ccs` (must carry ``layer`` and the
    sign-corrected ``acc_corrected`` column). The high band is every layer with
    ``acc_corrected >= best - delta``; fractional depth is ``layer / (n_layers - 1)``.
    """
    d = df.copy()
    if "acc_corrected" not in d.columns:
        d["acc_corrected"] = np.maximum(d["accuracy"], 1 - d["accuracy"])
    n = int(len(d))
    denom = max(n - 1, 1)
    best = d.loc[d["acc_corrected"].idxmax()]
    best_acc = float(best["acc_corrected"])
    best_layer = int(best["layer"])

    high = d[d["acc_corrected"] >= best_acc - delta]
    high_frac = high["layer"].astype(float) / denom
    n_high = int(len(high))

    return {
        "model": run.name,
        "params": run.params,
        "n_layers": n,
        "best_acc": round(best_acc, 3),
        "best_layer": best_layer,
        "best_layer_frac": round(best_layer / denom, 3),
        "mean_acc": round(float(d["acc_corrected"].mean()), 3),
        "n_high": n_high,
        "mean_acc_across_best": round(float(high["acc_corrected"].mean()), 3),
        "n_high_percentage": int(round(n_high / n * 100)),
        "high_frac_mean": round(float(high_frac.mean()), 3),
        "high_frac_min": round(float(high_frac.min()), 3),
        "high_frac_max": round(float(high_frac.max()), 3),
    }


def high_layers_table(*, delta: float = HIGH_DELTA) -> pd.DataFrame:
    """High-accuracy-layer table over every registry model with a PA-CCS run on disk."""
    rows = []
    for run in A.MODELS:
        df = A.load_ccs(run)
        if df is not None:
            rows.append(high_layer_row(run, df, delta=delta))
    return pd.DataFrame(rows)


def main() -> None:
    table = high_layers_table()
    out = RUNS / "summary_high_layers.csv"
    table.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table.to_string(index=False))
    print(f"\nwrote {out} ({len(table)} models, HIGH_DELTA={HIGH_DELTA})")


if __name__ == "__main__":
    main()
