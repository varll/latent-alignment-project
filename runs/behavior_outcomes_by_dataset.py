"""Behavior-outcome two-panel plot, reproduced *per dataset that has behavior data*.

Mirrors the two-panel "behavior outcome" cell in
``notebooks/analysis_pa_ccs_vs_judges.ipynb`` (the one that saves ``behavior_outcomes.png``):

* **Left -- outcome mix**: a stacked bar per model of the ``safe`` / ``harmful`` / ``gibberish`` /
  ``tie`` rates, coloured with :data:`analysis.JUDGE_COLORS`.
* **Right -- harmful polarity asymmetry**: grouped bars per model of ``harmful_l0_stmt`` (prompted
  with the hateful/toxic statement, label 0) vs ``harmful_l1_neg`` (prompted with the benign
  negation/rewrite, label 1).

"Behavior" = judged free-generations, which only exist for some datasets:

* ``mixed`` -- the existing 3-judge consensus, reused verbatim from
  :func:`analysis.behavior_summary`
  (``safe_rate`` / ``harmful_rate`` / ``gibberish_rate`` / ``tie_rate`` + ``harmful_l0_stmt`` /
  ``harmful_l1_neg`` over ``judge_3model_label``). Present: the 6 registry models with a
  ``behavior_*/judge_3model_results_*.csv`` (OLMo-1B base, OLMo-2-1B base, OLMo-2-1B instruct,
  Gemma-3-1B base, Gemma-3-1B instruct, Qwen3-4B instruct). Missing behavior: Qwen3-8B base.
* ``toxigen`` -- only Qwen3-4B-instruct has ToxiGen behavior
  (``runs/behavior_qwen3_4b_it/toxigen_paired_judge.csv``). The same quantities are computed from
  that single-judge CSV: the outcome mix from ``judge_label`` (safe/harmful/gibberish; there is no
  ``tie`` column in the single-judge format, so its rate is 0), and the l0/l1 asymmetry where
  **l0 = the toxic statement (``label == 0``)** and **l1 = the benign rewrite (``label == 1``)** --
  the paired set carries both polarities via ``label``, exactly as the mixed set does. Missing
  behavior: every registry model except Qwen3-4B instruct.
* ``toxigen_single`` -- NO behavior data (the single format was latent-only), so this renders an
  explicit "N/A -- no judged generations for this dataset" placeholder instead of a blank.

Missing-model scheme matches the other new notebooks/helpers (e.g. ``runs/separability_grid.py`` /
``runs/knows_vs_does_high.py``): a clear, explicit caption per dataset -- measured against the 7
registry models in :data:`analysis.MODELS` -- naming which models are **present** (have judged
behavior for that dataset) vs **missing behavior**.

Pure pandas/numpy for the table building so it can be exercised by a standalone (``Agg``) script
without a Jupyter kernel; matplotlib is imported lazily inside the drawing helpers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path(__file__).resolve().parent
if str(RUNS) not in sys.path:
    sys.path.insert(0, str(RUNS))

import analysis as A  # noqa: E402  (must follow the sys.path tweak above)

# The 7 registry models every dataset's coverage is measured against (the "expected" set).
EXPECTED = [r.name for r in A.MODELS]

# Outcome-mix segments (stacked, in this order) -- coloured via A.JUDGE_COLORS.
OUTCOME_CATS = ["safe", "harmful", "gibberish", "tie"]

# Polarity-asymmetry bar colours, kept identical to the original notebook cell.
L0_COLOR = "#d62728"  # prompted with the hateful/toxic statement (label 0)
L1_COLOR = "#f4a3a3"  # prompted with the benign negation/rewrite (label 1)

# The numeric columns produced per model (shared shape across datasets).
VALUE_COLS = [
    "safe_rate", "harmful_rate", "gibberish_rate", "tie_rate",
    "coherent_rate", "harmful_l0_stmt", "harmful_l1_neg",
]

# ToxiGen behavior lives under one model's behavior dir; only Qwen3-4B-instruct has it.
TOXIGEN_JUDGE = RUNS / "behavior_qwen3_4b_it" / "toxigen_paired_judge.csv"
TOXIGEN_MODEL = "Qwen3-4B instruct"

DATASETS = ["mixed", "toxigen", "toxigen_single"]

DATASET_TITLE = {
    "mixed": "mixed \u2014 paired hateful statement vs benign negation (3-judge consensus)",
    "toxigen": "toxigen \u2014 paired toxic statement vs benign rewrite (single judge)",
    "toxigen_single": "toxigen_single \u2014 single-format prompts (latent-only)",
}

# Datasets that carry judged free-generations (everything else is an explicit N/A placeholder).
HAS_BEHAVIOR = {"mixed": True, "toxigen": True, "toxigen_single": False}


# --------------------------------------------------------------------------------------
# Per-dataset behavior tables (safe/harmful/gibberish/tie + l0/l1 asymmetry).
# --------------------------------------------------------------------------------------
def _harmful_rate(s: pd.Series) -> float:
    """Fraction of labels equal to ``harmful`` (matches analysis._harmful_rate)."""
    return float((s == "harmful").mean())


def mixed_table() -> pd.DataFrame:
    """Mixed behavior, reused verbatim from analysis.behavior_summary (3-judge consensus).

    Returns one row per registry model that has a ``judge_3model_results_*.csv`` (the 4 present
    models), with the VALUE_COLS columns over the ``judge_3model_label`` consensus.
    """
    beh = A.behavior_summary(label_col="judge_3model_label")
    return beh[["model", "n", *VALUE_COLS]].reset_index(drop=True)


def toxigen_behavior_row(path: Path = TOXIGEN_JUDGE, *, model: str = TOXIGEN_MODEL) -> dict:
    """Compute the same quantities the mixed table holds, from the ToxiGen single-judge CSV.

    The CSV columns are ``statement, generation, label, judge_label, judge_reason``:

    * outcome mix comes from ``judge_label`` (safe/harmful/gibberish). The single-judge format has
      no ``tie`` column, so ``tie_rate`` is 0 by construction.
    * the l0/l1 asymmetry is derived from the paired ``label`` column exactly as the mixed set:
      ``harmful_l0_stmt`` = harmful rate where ``label == 0`` (the toxic statement) and
      ``harmful_l1_neg`` = harmful rate where ``label == 1`` (the benign rewrite).
    """
    df = pd.read_csv(path)
    rate = df["judge_label"].value_counts(normalize=True)
    safe = float(rate.get("safe", 0.0))
    harmful = float(rate.get("harmful", 0.0))
    gibberish = float(rate.get("gibberish", 0.0))
    tie = float(rate.get("tie", 0.0))  # absent in single-judge ToxiGen -> 0
    coherent = safe + harmful
    by_label = df.groupby("label")["judge_label"]
    has0 = 0 in df["label"].values
    has1 = 1 in df["label"].values
    return {
        "model": model,
        "n": int(len(df)),
        "safe_rate": round(safe, 3),
        "harmful_rate": round(harmful, 3),
        "gibberish_rate": round(gibberish, 3),
        "tie_rate": round(tie, 3),
        "coherent_rate": round(coherent, 3),
        "harmful_l0_stmt": round(_harmful_rate(by_label.get_group(0)), 3) if has0 else float("nan"),
        "harmful_l1_neg": round(_harmful_rate(by_label.get_group(1)), 3) if has1 else float("nan"),
    }


def toxigen_table(path: Path = TOXIGEN_JUDGE) -> pd.DataFrame:
    """ToxiGen behavior table (a single row -- Qwen3-4B-instruct is the only model with it)."""
    row = toxigen_behavior_row(path)
    return pd.DataFrame([row])[["model", "n", *VALUE_COLS]]


def behavior_table(dataset: str) -> pd.DataFrame:
    """Per-model behavior table for ``dataset`` (empty for datasets with no judged behavior)."""
    if dataset == "mixed":
        return mixed_table()
    if dataset == "toxigen":
        return toxigen_table()
    if dataset == "toxigen_single":
        return pd.DataFrame(columns=["model", "n", *VALUE_COLS])
    raise ValueError(f"unknown dataset: {dataset!r}")


def coverage(dataset: str) -> tuple[list[str], list[str]]:
    """``(present, missing)`` registry models for ``dataset`` -- present = has judged behavior.

    ``present`` preserves A.MODELS order; ``missing`` is everything else in A.MODELS order.
    """
    if HAS_BEHAVIOR.get(dataset, False):
        present_set = set(behavior_table(dataset)["model"])
    else:
        present_set = set()
    present = [m for m in EXPECTED if m in present_set]
    missing = [m for m in EXPECTED if m not in present_set]
    return present, missing


# --------------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------------
def _coverage_caption(fig, dataset: str, present: list[str], missing: list[str], *, y: float,
                      available: bool = True) -> None:
    """Bottom-outside, explicit per-dataset model-coverage caption (mirrors the other notebooks)."""
    head = (f"Model coverage for '{dataset}' (expected = {len(EXPECTED)} registry models):"
            if available else
            f"Model coverage for '{dataset}': no judged free-generations for ANY model "
            f"(expected = {len(EXPECTED)} registry models).")
    lines = [
        head,
        "   \u2022 present (judged behavior, plotted): "
        + (", ".join(present) if present else "none"),
        "   \u2022 missing behavior (not plotted): "
        + (", ".join(missing) if missing else "none \u2014 all expected models present"),
    ]
    fig.text(0.5, y, "\n".join(lines), ha="center", va="top", fontsize=8,
             color="#444", style="italic")


def make_outcome_figure(dataset: str):
    """Build the two-panel behavior-outcome figure for a dataset that has judged behavior.

    Returns ``(fig, report)``. Raises ``ValueError`` for datasets without behavior data
    (use :func:`make_na_figure` for those).
    """
    if not HAS_BEHAVIOR.get(dataset, False):
        raise ValueError(f"dataset {dataset!r} has no judged behavior; use make_na_figure")
    import matplotlib.pyplot as plt

    beh = behavior_table(dataset)
    present, missing = coverage(dataset)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    xb = np.arange(len(beh))

    # Left: stacked outcome mix (safe / harmful / gibberish / tie), A.JUDGE_COLORS.
    bottom = np.zeros(len(beh))
    for cat in OUTCOME_CATS:
        vals = beh[f"{cat}_rate"].to_numpy()
        ax1.bar(xb, vals, bottom=bottom, color=A.JUDGE_COLORS[cat], label=cat)
        bottom += vals
    ax1.set_xticks(xb)
    ax1.set_xticklabels(beh["model"], rotation=15, ha="right", fontsize=8)
    ax1.set(ylabel="proportion of generations", ylim=(0, 1),
            title="Free-generation outcome mix")
    ax1.legend(fontsize=8, ncol=2)

    # Right: harmful polarity asymmetry (hateful/toxic statement vs benign negation/rewrite).
    w = 0.38
    ax2.bar(xb - w / 2, beh["harmful_l0_stmt"], w, color=L0_COLOR,
            label="prompted with hateful/toxic statement (label 0)")
    ax2.bar(xb + w / 2, beh["harmful_l1_neg"], w, color=L1_COLOR,
            label="prompted with benign negation/rewrite (label 1)")
    ax2.set_xticks(xb)
    ax2.set_xticklabels(beh["model"], rotation=15, ha="right", fontsize=8)
    ax2.set(ylabel="harmful rate", title="Harmful-output polarity asymmetry")
    ax2.legend(fontsize=8)

    fig.suptitle(f"Behavior outcomes \u2014 {DATASET_TITLE[dataset]}", y=1.02, fontsize=12)
    _coverage_caption(fig, dataset, present, missing, y=-0.04)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig, {
        "dataset": dataset,
        "available": True,
        "models": list(beh["model"]),
        "present": present,
        "missing": missing,
        "table": beh.to_dict(orient="records"),
    }


def make_na_figure(dataset: str):
    """Build an explicit "N/A -- no judged generations" placeholder figure for ``dataset``.

    Returns ``(fig, report)``. Used for datasets (e.g. ``toxigen_single``) that have no behavior.
    """
    import matplotlib.pyplot as plt

    present, missing = coverage(dataset)  # present == [] for these datasets
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    ax.text(0.5, 0.5, "N/A \u2014 no judged generations for this dataset",
            ha="center", va="center", fontsize=20, fontweight="bold", color="#888",
            transform=ax.transAxes)
    ax.text(0.5, 0.34,
            f"'{dataset}' was latent-only; there are no free-generation judge labels to score "
            "(outcome mix / l0\u2013l1 asymmetry).",
            ha="center", va="center", fontsize=10, color="#888", transform=ax.transAxes)
    fig.suptitle(f"Behavior outcomes \u2014 {DATASET_TITLE[dataset]}", y=1.02, fontsize=12)
    _coverage_caption(fig, dataset, present, missing, y=0.02, available=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig, {
        "dataset": dataset,
        "available": False,
        "models": [],
        "present": present,
        "missing": missing,
        "table": [],
    }


def make_figure(dataset: str):
    """Dispatch to the outcome figure (if the dataset has behavior) or the N/A placeholder."""
    if HAS_BEHAVIOR.get(dataset, False):
        return make_outcome_figure(dataset)
    return make_na_figure(dataset)


def render(dataset: str, outdir: str | Path | None = None, dpi: int = 300) -> dict:
    """Render ``dataset``; save the figure for datasets with behavior, skip-save for N/A ones.

    Datasets with behavior are saved high-res to ``runs/figs/behavior_outcomes_{dataset}.png``.
    Datasets without behavior return ``path=None`` (the notebook still shows the placeholder).
    """
    import matplotlib.pyplot as plt

    out = Path(outdir) if outdir is not None else (RUNS / "figs")
    out.mkdir(parents=True, exist_ok=True)
    fig, report = make_figure(dataset)
    if report["available"]:
        path = out / f"behavior_outcomes_{dataset}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        report["path"] = str(path)
        report["dpi"] = dpi
    else:
        report["path"] = None
        report["dpi"] = dpi
    plt.close(fig)
    return report


def render_all(outdir: str | Path | None = None, dpi: int = 300) -> dict:
    """Render every dataset; return ``{dataset: report}``."""
    return {ds: render(ds, outdir=outdir, dpi=dpi) for ds in DATASETS}
