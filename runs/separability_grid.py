"""Grid of PA-CCS latent-separability plots: 4 metrics x 4 dataset families.

Plotting/data logic for ``notebooks/latent_separability_grid.ipynb``. Kept as a module so the
notebook cells stay short and the same code can be exercised by a standalone (``Agg``) script
without a Jupyter kernel.

For EACH metric -- accuracy (sign-corrected), silhouette, |PC| (abs polar consistency) and
CI (contradiction index) -- :func:`make_figure` renders a 2x2 grid, one panel per dataset:

* ``mixed_dataset``   -- the polarity *mixed* set (7 registry models, via ``analysis.load_ccs``)
* ``toxigen_single``  -- ToxiGen ``single`` format (no opposite pairs -> |PC|/CI are NaN by design)
* ``toxigen_paired``  -- ToxiGen ``--with-negations`` paired format
* ``toxigen_groups``  -- per-(target group) k-fold runs (per-group CSV has no CI column)

The first three panels are per-layer curves (x = fractional depth, one line per model); the
fourth is per-target-group markers (x = group, one marker series per model).

Visual conventions (shared across all four figures):

* **colour encodes the model family** (so the base and instruct variants of one family share a
  colour) and the **line style / marker fill encodes the variant**: base = solid line / filled
  marker, instruct = dashed line / hollow marker.
* Every figure carries ONE shared legend placed at the BOTTOM, OUTSIDE the axes, that names the
  models, the base-vs-instruct convention, the chance baseline, and -- explicitly -- the models
  that are MISSING from each panel (expected set = the 7 registry models in ``analysis.MODELS``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path(__file__).resolve().parent
if str(RUNS) not in sys.path:
    sys.path.insert(0, str(RUNS))

import analysis as A  # noqa: E402  (must follow the sys.path tweak above)

# --------------------------------------------------------------------------------------
# Per-model styling: colour = family, line style / marker fill = variant.
# --------------------------------------------------------------------------------------
FAMILY_COLORS = {
    "olmo": "#8c564b",    # brown
    "olmo2": "#1f77b4",   # blue
    "gemma3": "#2ca02c",  # green
    "qwen3": "#9467bd",   # purple
}
FAMILY_MARKERS = {
    "olmo": "X",
    "olmo2": "o",
    "gemma3": "s",
    "qwen3": "^",
}
# base = solid line + filled marker, instruct = dashed line + hollow marker.
VARIANT_LS = {"base": "-", "instruct": "--"}


def model_style(family: str, variant: str) -> dict:
    """Return the matplotlib kwargs bundle (colour / line style / marker / fill) for a model."""
    color = FAMILY_COLORS.get(family, "#444444")
    ls = VARIANT_LS.get(variant, "-")
    marker = FAMILY_MARKERS.get(family, "o")
    filled = variant == "base"
    return {
        "color": color,
        "ls": ls,
        "marker": marker,
        "mfc": color if filled else "white",
        "mec": color,
    }


# Canonical model registry (the "expected" set every panel is measured against).
CANON = {r.name: r for r in A.MODELS}
EXPECTED = [r.name for r in A.MODELS]


# --------------------------------------------------------------------------------------
# Dataset wiring: map each dataset family to (canonical model -> per-layer/per-group CSV).
# --------------------------------------------------------------------------------------
@dataclass
class Series:
    """One model's contribution to one dataset panel."""

    model: str
    family: str
    variant: str
    df: pd.DataFrame | None
    kind: str  # "layer" or "group"


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _series(model: str, df: pd.DataFrame | None, kind: str) -> Series:
    r = CANON[model]
    return Series(model, r.family, r.variant, df, kind)


DATASET_ORDER = ["mixed_dataset", "toxigen_single", "toxigen_paired", "toxigen_groups"]
DATASET_TITLE = {
    "mixed_dataset": "mixed_dataset  ·  polarity mixed set",
    "toxigen_single": "toxigen_single  ·  single format",
    "toxigen_paired": "toxigen_paired  ·  --with-negations",
    "toxigen_groups": "toxigen by groups  ·  per-group k-fold",
}


def build_datasets() -> dict[str, list[Series]]:
    """Load every CSV once and group the resulting :class:`Series` by dataset family."""
    mixed = [_series(r.name, A.load_ccs(r), "layer") for r in A.MODELS]

    single = [
        _series("OLMo-2-1B base",
                _read(RUNS / "olmo2_1b_base_toxigen" / "ccs_summary.csv"), "layer"),
        _series("OLMo-2-1B instruct",
                _read(RUNS / "olmo2_1b_it_toxigen" / "ccs_summary.csv"), "layer"),
        _series("Qwen3-4B instruct",
                _read(RUNS / "qwen_toxigen_single" / "ccs_summary.csv"), "layer"),
    ]

    paired = [
        _series("OLMo-2-1B base",
                _read(RUNS / "olmo2_1b_base_toxigen_paired" / "ccs_summary.csv"), "layer"),
        _series("OLMo-2-1B instruct",
                _read(RUNS / "olmo2_1b_it_toxigen_paired" / "ccs_summary.csv"), "layer"),
        _series("Qwen3-4B instruct",
                _read(RUNS / "qwen3-4b-instruct" / "ccs_summary_toxigen_pair.csv"), "layer"),
    ]

    groups = [
        _series("Qwen3-4B instruct",
                _read(RUNS / "qwen3-4b-instruct" / "toxigen_group_kfold.csv"), "group"),
        _series("Qwen3-8B base",
                _read(RUNS / "qwen3-8b-base" / "toxigen_group_kfold.csv"), "group"),
    ]

    return {
        "mixed_dataset": mixed,
        "toxigen_single": single,
        "toxigen_paired": paired,
        "toxigen_groups": groups,
    }


# --------------------------------------------------------------------------------------
# Metric extraction. |PC| = abs(polar_consistency_mean); CI = contradiction_index_mean.
# --------------------------------------------------------------------------------------
# group CSV columns / std partners (CI is absent from the per-group k-fold CSV -> None).
_GROUP_COL = {
    "accuracy": "accuracy", "silhouette": "silhouette_mean", "pc": "abs_pc_mean", "ci": None,
}
_GROUP_STD = {
    "accuracy": "accuracy_std", "silhouette": "silhouette_std", "pc": "abs_pc_std", "ci": None,
}


def _layer_y(df: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "accuracy":
        return np.maximum(df["accuracy"], 1 - df["accuracy"])
    if metric == "silhouette":
        return df["silhouette"]
    if metric == "pc":
        return df["polar_consistency_mean"].abs()
    return df["contradiction_index_mean"]


def series_values(s: Series, metric: str):
    """Return ``(x, y[, yerr])`` for a series/metric, or ``None`` if it has no usable data.

    "No usable data" covers a missing CSV, an all-NaN metric column (e.g. |PC|/CI under the
    ToxiGen ``single`` format), or a metric not recorded for that dataset (CI for the per-group
    k-fold CSV). Those cases are exactly the panels we annotate as N/A.
    """
    if s.df is None:
        return None
    if s.kind == "group":
        col = _GROUP_COL[metric]
        if col is None or col not in s.df.columns:
            return None
        y = s.df[col]
        if y.isna().all():
            return None
        std_col = _GROUP_STD[metric]
        yerr = s.df[std_col] if std_col and std_col in s.df.columns else None
        return list(s.df["target_group"].astype(str)), y.to_numpy(dtype=float), \
            (yerr.to_numpy(dtype=float) if yerr is not None else None)
    # layer kind
    y = _layer_y(s.df, metric)
    if y.isna().all():
        return None
    x = s.df["layer"] / (len(s.df) - 1)
    return x.to_numpy(dtype=float), y.to_numpy(dtype=float), None


# Per-metric figure metadata; baselines pulled from analysis.py (random/chance nulls).
def _baselines() -> dict:
    pb = A.random_probe_polarity_baseline()
    return {
        "accuracy": (0.5, "chance accuracy = 0.5"),
        "silhouette": (0.0, "random clusters \u2248 0"),
        "pc": (pb["abs_pc_mean"], f"random-probe |PC| \u2248 {pb['abs_pc_mean']:.2f}"),
        "ci": (pb["ci_mean"], f"random-probe CI \u2248 {pb['ci_mean']:.2f}"),
    }


METRIC_TITLE = {
    "accuracy": "Accuracy (sign-corrected CCS)",
    "silhouette": "Silhouette",
    "pc": "|PC|  ·  abs polar consistency",
    "ci": "CI  ·  contradiction index",
}
METRIC_YLABEL = {
    "accuracy": "corrected accuracy",
    "silhouette": "silhouette",
    "pc": "|polar consistency|",
    "ci": "contradiction index",
}
METRIC_FNAME = {
    "accuracy": "sep_grid_accuracy.png",
    "silhouette": "sep_grid_silhouette.png",
    "pc": "sep_grid_pc.png",
    "ci": "sep_grid_ci.png",
}


def _empty_message(dskey: str, metric: str) -> str:
    if dskey == "toxigen_single" and metric in ("pc", "ci"):
        return ("N/A \u2014 single format has no\nopposite-polarity pairs\n"
                "(polar consistency / CI undefined)")
    if dskey == "toxigen_groups" and metric == "ci":
        return "N/A \u2014 CI is not recorded\nin the per-group k-fold CSV"
    return "no data available\nfor this panel"


# --------------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------------
def _draw_panel(ax, dskey, series, metric, baseline, legend_models):
    """Draw one dataset panel; return the list of models that contributed real data."""
    is_group = dskey == "toxigen_groups"
    present: list[str] = []

    base_val, base_lbl = baseline
    if base_val is not None:
        ax.axhline(base_val, ls=":", color="0.4", lw=1.2, zorder=0)

    drawable = [(s, series_values(s, metric)) for s in series]
    drawable = [(s, v) for s, v in drawable if v is not None]

    if is_group and drawable:
        n = len(drawable)
        offsets = (np.arange(n) - (n - 1) / 2) * 0.18
    for i, (s, vals) in enumerate(drawable):
        st = model_style(s.family, s.variant)
        present.append(s.model)
        if is_group:
            labels, y, yerr = vals
            xs = np.arange(len(labels)) + offsets[i]
            ax.errorbar(xs, y, yerr=yerr, ls="none", marker=st["marker"], ms=8,
                        color=st["color"], markerfacecolor=st["mfc"], markeredgecolor=st["mec"],
                        markeredgewidth=1.5, elinewidth=1, capsize=2, alpha=0.9)
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
        else:
            x, y, _ = vals
            ax.plot(x, y, ls=st["ls"], color=st["color"], marker=st["marker"], ms=4, lw=1.8,
                    alpha=0.9, markerfacecolor=st["mfc"], markeredgecolor=st["mec"])
        if s.model not in legend_models:
            import matplotlib.lines as mlines
            legend_models[s.model] = mlines.Line2D(
                [0], [0], color=st["color"], ls=st["ls"], marker=st["marker"],
                markerfacecolor=st["mfc"], markeredgecolor=st["mec"], lw=1.8, ms=7)

    ax.set_title(DATASET_TITLE[dskey], fontsize=11)
    ax.set_ylabel(METRIC_YLABEL[metric], fontsize=9)
    xlabel = "target group" if is_group else "fractional depth (layer / last layer)"
    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(True, alpha=0.25)

    if not drawable:
        ax.text(0.5, 0.5, _empty_message(dskey, metric), ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="#b03030",
                bbox=dict(boxstyle="round", fc="#fdecea", ec="#b03030"))
    return present


# Filename slug per dataset (used for the standalone per-panel PNGs sep_grid_<metric>_<slug>.png).
DATASET_SLUG = {
    "mixed_dataset": "mixed",
    "toxigen_single": "toxigen_single",
    "toxigen_paired": "toxigen_paired",
    "toxigen_groups": "toxigen_groups",
}


def _legend_and_caption(fig, legend_models, baseline, report, panels, *, ncol=None):
    """Attach the shared bottom-outside legend + the explicit per-panel missing-models caption.

    ``panels`` is the list of dataset keys whose missing-model lines to print (all four for the
    combined grid, a single key for a standalone panel).
    """
    import matplotlib.lines as mlines

    model_handles = [legend_models[m] for m in EXPECTED if m in legend_models]
    model_labels = [m for m in EXPECTED if m in legend_models]

    convention = [
        mlines.Line2D([0], [0], color="0.3", ls="-", lw=2),
        mlines.Line2D([0], [0], color="0.3", ls="--", lw=2),
        mlines.Line2D([0], [0], color="0.3", ls="none", marker="o", markerfacecolor="0.3",
                      markeredgecolor="0.3", ms=8),
        mlines.Line2D([0], [0], color="0.3", ls="none", marker="o", markerfacecolor="white",
                      markeredgecolor="0.3", ms=8),
    ]
    convention_labels = [
        "base = solid line", "instruct = dashed line",
        "base = filled marker", "instruct = hollow marker",
    ]
    baseline_handle = [mlines.Line2D([0], [0], color="0.4", ls=":", lw=1.2)]

    handles = model_handles + convention + baseline_handle
    labels = model_labels + convention_labels + [baseline[1]]
    if ncol is None:
        ncol = min(5, max(3, len(handles) // 2 + 1))
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=ncol,
               fontsize=9, frameon=True,
               title="models (colour = family · style = variant)  +  conventions / baseline")

    lines = [f"Missing models per panel (expected = {len(EXPECTED)} registry models):"]
    for dskey in panels:
        miss = report[dskey]["missing"]
        lines.append(f"   \u2022 {DATASET_TITLE[dskey].split('  ·')[0]}: "
                     + (", ".join(miss) if miss else "none \u2014 all expected models present"))
    fig.text(0.5, 0.005, "\n".join(lines), ha="center", va="bottom", fontsize=7.5,
             color="#444", style="italic")


def make_figure(metric: str, datasets: dict | None = None, baselines: dict | None = None):
    """Build the 2x2 figure for one metric. Returns ``(fig, report)``.

    ``report[dskey]`` = ``{"present": [...], "missing": [...]}`` against the 7 registry models.
    """
    import matplotlib.pyplot as plt

    datasets = datasets if datasets is not None else build_datasets()
    baselines = baselines if baselines is not None else _baselines()
    baseline = baselines[metric]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    legend_models: dict = {}
    report: dict = {}

    for ax, dskey in zip(axes, DATASET_ORDER, strict=True):
        present = _draw_panel(ax, dskey, datasets[dskey], metric, baseline, legend_models)
        report[dskey] = {
            "present": present,
            "missing": [m for m in EXPECTED if m not in present],
        }

    fig.suptitle(f"PA-CCS latent separability \u2014 {METRIC_TITLE[metric]}", fontsize=15, y=0.99)
    _legend_and_caption(fig, legend_models, baseline, report, DATASET_ORDER)
    fig.tight_layout(rect=(0, 0.20, 1, 0.97))
    return fig, report


def make_panel_figure(metric: str, dskey: str, datasets: dict | None = None,
                      baselines: dict | None = None):
    """Build a single dataset panel as its own standalone figure (own title + bottom legend).

    Returns ``(fig, present, missing)`` for that one metric x dataset cell.
    """
    import matplotlib.pyplot as plt

    datasets = datasets if datasets is not None else build_datasets()
    baselines = baselines if baselines is not None else _baselines()
    baseline = baselines[metric]

    fig, ax = plt.subplots(figsize=(9, 7))
    legend_models: dict = {}
    present = _draw_panel(ax, dskey, datasets[dskey], metric, baseline, legend_models)
    missing = [m for m in EXPECTED if m not in present]
    report = {dskey: {"present": present, "missing": missing}}

    fig.suptitle(f"PA-CCS {METRIC_TITLE[metric]}\n{DATASET_TITLE[dskey]}", fontsize=12, y=0.995)
    _legend_and_caption(fig, legend_models, baseline, report, [dskey], ncol=3)
    fig.tight_layout(rect=(0, 0.24, 1, 0.95))
    return fig, present, missing


def render_all(outdir: str | Path | None = None, dpi: int = 300) -> dict:
    """Render and save all figures high-res: 4 combined grids + 16 standalone panels.

    Writes ``runs/figs/sep_grid_<metric>.png`` (combined 2x2) and
    ``runs/figs/sep_grid_<metric>_<dataset>.png`` (each panel), all at ``dpi`` with
    ``bbox_inches="tight"``. Returns a nested report incl. every saved path.
    """
    import matplotlib.pyplot as plt

    out = Path(outdir) if outdir is not None else (RUNS / "figs")
    out.mkdir(parents=True, exist_ok=True)
    datasets = build_datasets()
    baselines = _baselines()

    results: dict = {}
    for metric in METRIC_FNAME:
        fig, report = make_figure(metric, datasets=datasets, baselines=baselines)
        combined = out / METRIC_FNAME[metric]
        fig.savefig(combined, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        panels: dict = {}
        for dskey in DATASET_ORDER:
            pfig, present, missing = make_panel_figure(
                metric, dskey, datasets=datasets, baselines=baselines)
            ppath = out / f"sep_grid_{metric}_{DATASET_SLUG[dskey]}.png"
            pfig.savefig(ppath, dpi=dpi, bbox_inches="tight")
            plt.close(pfig)
            panels[dskey] = {"path": str(ppath), "present": present, "missing": missing}

        results[metric] = {"combined": str(combined), "report": report, "panels": panels}
    return results
