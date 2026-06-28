"""Layer-probe similarity and top-layer aggregated PA-CCS scorecard.

This is the importable implementation behind the "layer-probe similarity" section of
``notebooks/analysis_pa_ccs_vs_judges.ipynb`` (so the notebook cell can stay short) and
also a runnable verification script. It is pure pandas/numpy/sklearn/matplotlib — no
torch / model loading — because everything it needs is already on disk:

* per-layer PA-CCS metrics in ``runs/<run>/ccs_summary*.csv`` and
* per-layer probe weight vectors in ``runs/<run>/ccs_full_results.npz``
  (keys ``layer_<i>_weights`` / ``layer_<i>_bias`` plus the per-example
  ``layer_<i>_polar_consistency`` arrays that paired runs also store).

It answers three things per model:

1. **Top-layer selection** — see :func:`analysis.select_top_layers` (size-aware top-K is
   primary; a threshold band is reported as a robustness check).
2. **Do the top layers encode the same thing?**
   (a) *by weights* — pairwise cosine similarity + Pearson correlation between the
       selected layers' probe weight vectors (the primary evidence, available wherever an
       ``ccs_full_results.npz`` exists);
   (b) *by behaviour/info* — for paired runs that store per-example ``polar_consistency``,
       the cross-layer correlation of those per-example confidence vectors; otherwise a
       fallback correlation of the per-layer metric profiles.
3. **Top-layer aggregated scorecard** — mean ± std of the PA-CCS metrics over each model's
   selected top layers (a more robust per-model summary than the single best layer).

Run directly to verify and (re)generate the figures::

    .venv/bin/python runs/layer_similarity_analysis.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplcache"))

import analysis as A  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

RUNS = Path(__file__).resolve().parent
FIGS = RUNS / "figs"

# Runs that ship a per-layer probe-weight ``ccs_full_results.npz`` (only these allow the
# weight-vector similarity). ``has_pc`` flags the paired runs that also store per-example
# polar_consistency arrays (used for the behaviour/info cross-layer comparison).
NPZ_RUNS: list[dict] = [
    {"name": "OLMo-2-1B base · ToxiGen", "dir": "olmo2_1b_base_toxigen", "has_pc": False},
    {"name": "OLMo-2-1B instruct · ToxiGen", "dir": "olmo2_1b_it_toxigen", "has_pc": False},
    {"name": "OLMo-2-1B base · ToxiGen paired", "dir": "olmo2_1b_base_toxigen_paired",
     "has_pc": True},
    {"name": "OLMo-2-1B instruct · ToxiGen paired", "dir": "olmo2_1b_it_toxigen_paired",
     "has_pc": True},
]


# --------------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------------
def load_ccs_csv(run_dir: str) -> pd.DataFrame | None:
    """Per-layer PA-CCS metrics for an arbitrary run dir, with corrected accuracy."""
    import glob

    hits = sorted(glob.glob(str(RUNS / run_dir / "ccs_summary*.csv")))
    if not hits:
        return None
    df = pd.read_csv(hits[0])
    df["acc_corrected"] = np.maximum(df["accuracy"], 1 - df["accuracy"])
    return df


def load_layer_weights(npz_path: Path) -> dict[int, np.ndarray]:
    """Map layer index -> probe weight vector from ``ccs_full_results.npz``."""
    data = np.load(npz_path, allow_pickle=True)
    out: dict[int, np.ndarray] = {}
    for key in data.files:
        m = re.match(r"layer_(\d+)_weights$", key)
        if m:
            out[int(m.group(1))] = np.asarray(data[key], dtype=np.float64).ravel()
    return out


def load_layer_pc(npz_path: Path) -> dict[int, np.ndarray]:
    """Map layer index -> per-example polar_consistency vector (paired runs only)."""
    data = np.load(npz_path, allow_pickle=True)
    out: dict[int, np.ndarray] = {}
    for key in data.files:
        m = re.match(r"layer_(\d+)_polar_consistency$", key)
        if m:
            arr = np.asarray(data[key], dtype=np.float64).ravel()
            if arr.size:
                out[int(m.group(1))] = arr
    return out


# --------------------------------------------------------------------------------------
# Similarity primitives
# --------------------------------------------------------------------------------------
def _cosine_matrix(mat: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity matrix for a (n_items, dim) array."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = mat / norms
    return unit @ unit.T


def _pearson_matrix(mat: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation matrix for a (n_items, dim) array."""
    if mat.shape[1] < 2:
        return np.full((mat.shape[0], mat.shape[0]), np.nan)
    return np.corrcoef(mat)


def layer_weight_similarity(weights: dict[int, np.ndarray], layers: list[int]) -> dict:
    """Cosine + Pearson similarity between the weight vectors of the given layers.

    CCS probe directions are sign-ambiguous, so a pair of layers encoding the same axis
    can show up as ~-1. We therefore also report the absolute-value summaries, which are
    the right read-out for "is it the same direction (up to sign)?".
    """
    layers = [ln for ln in layers if ln in weights]
    if len(layers) < 2:
        return {"layers": layers, "cosine": None, "pearson": None}
    mat = np.vstack([weights[ln] for ln in layers])
    cos = _cosine_matrix(mat)
    pear = _pearson_matrix(mat)
    iu = np.triu_indices(len(layers), k=1)
    return {
        "layers": layers,
        "cosine": cos,
        "pearson": pear,
        "mean_abs_cosine": float(np.mean(np.abs(cos[iu]))),
        "mean_abs_pearson": float(np.nanmean(np.abs(pear[iu]))),
        "mean_cosine": float(np.mean(cos[iu])),
        "mean_pearson": float(np.nanmean(pear[iu])),
    }


def layer_pc_agreement(pc: dict[int, np.ndarray], layers: list[int]) -> dict:
    """Cross-layer correlation of per-example polar_consistency vectors (behaviour/info)."""
    layers = [ln for ln in layers if ln in pc]
    if len(layers) < 2:
        return {"layers": layers, "pearson": None}
    sizes = {pc[ln].size for ln in layers}
    if len(sizes) != 1:
        return {"layers": layers, "pearson": None}
    mat = np.vstack([pc[ln] for ln in layers])
    pear = _pearson_matrix(mat)
    iu = np.triu_indices(len(layers), k=1)
    return {
        "layers": layers,
        "pearson": pear,
        "mean_abs_pearson": float(np.nanmean(np.abs(pear[iu]))),
        "mean_pearson": float(np.nanmean(pear[iu])),
    }


# --------------------------------------------------------------------------------------
# Top-layer aggregated scorecard
# --------------------------------------------------------------------------------------
_SCORE_METRICS = [
    ("acc_corrected", "accuracy", False),
    ("silhouette", "silhouette", False),
    ("polar_consistency_mean", "|polar_consistency|", True),
    ("contradiction_index_mean", "contradiction_index", False),
]


def top_layer_scorecard_row(name: str, df: pd.DataFrame, sel: dict) -> dict:
    """Best-layer vs top-K-mean (± std) of the PA-CCS metrics over the selected layers."""
    layers = sel["topk_layers"]
    sub = df[df["layer"].isin(layers)]
    best = df.loc[df["acc_corrected"].idxmax()]
    row = {
        "model": name,
        "n_layers": sel["n_layers"],
        "k": sel["k"],
        "topk_layers": ",".join(str(x) for x in layers),
        "best_layer": int(best["layer"]),
        "best_acc": round(float(best["acc_corrected"]), 3),
    }
    for col, label, take_abs in _SCORE_METRICS:
        if col not in df.columns or df[col].isna().all():
            row[f"{label}_topk_mean"] = float("nan")
            row[f"{label}_topk_std"] = float("nan")
            continue
        vals = sub[col].abs() if take_abs else sub[col]
        row[f"{label}_topk_mean"] = round(float(vals.mean()), 3)
        row[f"{label}_topk_std"] = round(float(vals.std(ddof=0)), 3)
    return row


def build_scorecard(runs: list[A.ModelRun] | None = None) -> pd.DataFrame:
    """Top-K-mean scorecard over the analysis.MODELS registry (mixed-dataset runs)."""
    runs = list(A.MODELS) if runs is None else runs
    rows = []
    for run in runs:
        df = A.load_ccs(run)
        if df is None:
            continue
        sel = A.select_top_layers(df)
        rows.append(top_layer_scorecard_row(run.name, df, sel))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def plot_weight_similarity(specs: list[dict], out_path: Path) -> Path | None:
    """Cosine-similarity heatmap of the selected top layers' weight vectors per run."""
    panels = []
    for spec in specs:
        df = load_ccs_csv(spec["dir"])
        npz = RUNS / spec["dir"] / "ccs_full_results.npz"
        if df is None or not npz.exists():
            continue
        weights = load_layer_weights(npz)
        sel = A.select_top_layers(df)
        sim = layer_weight_similarity(weights, sel["topk_layers"])
        if sim["cosine"] is not None:
            panels.append((spec["name"], sim))
    if not panels:
        return None
    ncols = min(2, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    im = None
    for ax, (name, sim) in zip(axes.ravel(), panels, strict=False):
        ax.axis("on")
        ax.grid(False)
        cos = sim["cosine"]
        layers = sim["layers"]
        im = ax.imshow(cos, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(layers)))
        ax.set_yticks(range(len(layers)))
        ax.set_xticklabels(layers, fontsize=8)
        ax.set_yticklabels(layers, fontsize=8)
        ax.set_title(f"{name}\nmean|cos|={sim['mean_abs_cosine']:.2f}", fontsize=9)
        for i in range(len(layers)):
            for j in range(len(layers)):
                ax.text(j, i, f"{cos[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
    fig.suptitle("Top-layer probe-weight cosine similarity (selected layers)", y=1.0)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="cosine similarity")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_scorecard(score: pd.DataFrame, out_path: Path) -> Path | None:
    """Best-layer accuracy vs top-K-mean accuracy (± std) per model."""
    if score.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(score) + 1.6))
    y = np.arange(len(score))
    ax.barh(y - 0.2, score["best_acc"], height=0.38, color="#9467bd", label="best layer")
    ax.barh(y + 0.2, score["accuracy_topk_mean"], height=0.38, color="#1f77b4",
            xerr=score["accuracy_topk_std"], capsize=3, label="top-K mean ± std")
    ax.set_yticks(y)
    ax.set_yticklabels(score["model"], fontsize=9)
    ax.axvline(0.5, ls="--", c="grey", lw=1)
    ax.set(xlabel="corrected CCS accuracy", xlim=(0.45, 1.0),
           title="Best-layer vs top-K-mean PA-CCS accuracy")
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
# Verification entry point
# --------------------------------------------------------------------------------------
def main() -> None:
    plt.switch_backend("Agg")  # headless when run as a script; left untouched on import
    FIGS.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    print("=" * 80)
    print("TOP-LAYER SELECTION (size-aware top-K is primary; threshold band as a check)")
    print("=" * 80)
    for run in A.MODELS:
        df = A.load_ccs(run)
        if df is None:
            print(f"  {run.name:24s}  (no ccs_summary on disk)")
            continue
        sel = A.select_top_layers(df)
        print(f"  {run.name:24s}  n={sel['n_layers']:2d} ({sel['size_class']:5s}) "
              f"-> top-{sel['k']} {sel['topk_layers']}  | "
              f"threshold(>= {sel['threshold']:.3f}) {sel['threshold_layers']}")

    print("\n" + "=" * 80)
    print("WEIGHT-VECTOR SIMILARITY across selected top layers (runs with npz weights)")
    print("=" * 80)
    for spec in NPZ_RUNS:
        df = load_ccs_csv(spec["dir"])
        npz = RUNS / spec["dir"] / "ccs_full_results.npz"
        if df is None or not npz.exists():
            print(f"  {spec['name']:36s}  (missing csv/npz)")
            continue
        weights = load_layer_weights(npz)
        sel = A.select_top_layers(df)
        sim = layer_weight_similarity(weights, sel["topk_layers"])
        if sim["cosine"] is None:
            print(f"  {spec['name']:36s}  (fewer than 2 weight vectors)")
            continue
        print(f"  {spec['name']:36s}  layers={sim['layers']}  "
              f"mean|cos|={sim['mean_abs_cosine']:.3f}  mean cos={sim['mean_cosine']:+.3f}  "
              f"mean|pearson|={sim['mean_abs_pearson']:.3f}")
        if spec["has_pc"]:
            pc = load_layer_pc(npz)
            agree = layer_pc_agreement(pc, sel["topk_layers"])
            if agree["pearson"] is not None:
                nlp = len(agree["layers"])
                print(f"      per-example polar_consistency agreement: "
                      f"mean|pearson|={agree['mean_abs_pearson']:.3f}  "
                      f"mean pearson={agree['mean_pearson']:+.3f}  (n_layers={nlp})")

    print("\n" + "=" * 80)
    print("TOP-LAYER AGGREGATED SCORECARD (best-layer vs top-K-mean ± std)")
    print("=" * 80)
    score = build_scorecard()
    if not score.empty:
        print(score.to_string(index=False))

    f1 = plot_weight_similarity(NPZ_RUNS, FIGS / "layer_probe_similarity.png")
    f2 = plot_scorecard(score, FIGS / "top_layer_scorecard.png")
    print("\nfigures:")
    print("  ", f1)
    print("  ", f2)


if __name__ == "__main__":
    main()
