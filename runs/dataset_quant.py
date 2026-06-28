"""Quantitative dataset composition + ToxiGen rewrite-noise metric.

Importable implementation behind two notebook sections of
``notebooks/analysis_pa_ccs_vs_judges.ipynb`` (so the cells stay short), and a runnable
verifier. Pure pandas/numpy/matplotlib — no torch / model loading.

**Analysis A — dataset composition.**
    * ToxiGen paired set (``data/toxigen/raw/toxigen_annotated_test_paired.csv``):
      entries per ``target_group`` (n_pairs / n_examples) + overall toxic/benign split.
    * Mixed dataset (``data/polarity_probing/raw/mixed_dataset.csv``): total rows and the
      ``is_harmfull_opposition`` polarity balance; plus per-category counts IF the
      Nemotron-labelled ``runs/mixed_dataset_group_labeled.csv`` (``target_group`` column)
      exists — otherwise the caller is told to run ``runs/label_mixed_groups.py`` first.

**Analysis B — rewrite-noise metric.**
    The 402 benign rewrites are machine-generated, so PA-CCS polarity metrics assume each
    is a clean, on-topic, benign *opposite* of ``toxic_text``. We score each rewrite with
    transparent lexical signals (Jaccard overlap, length ratio, unchanged/artifact flags,
    low-overlap topic-drift flag, clean-minimal-negation flag) and combine them into a
    per-rewrite ``noise_score`` in [0, 1] (see :func:`rewrite_noise_table`). Aggregates are
    cross-checked against the FINDINGS §5d audit (~13% clean minimal negations, ~3%
    artifacts, ~11% low overlap).

Run directly to verify and (re)generate the CSV + figures::

    .venv/bin/python runs/dataset_quant.py
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplcache"))

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
FIGS = RUNS / "figs"

TOXIGEN_PAIRED = ROOT / "data/toxigen/raw/toxigen_annotated_test_paired.csv"
MIXED = ROOT / "data/polarity_probing/raw/mixed_dataset.csv"
MIXED_LABELED = RUNS / "mixed_dataset_group_labeled.csv"

REWRITE_NOISE_CSV = RUNS / "toxigen_rewrite_noise.csv"

# Tokens that signal a (clean) polarity flip when present on exactly one side of a pair.
NEGATION_TOKENS = frozenset({
    "not", "no", "never", "cannot", "none", "nothing", "nobody", "without", "neither",
    "nor", "n't", "dont", "cant", "wont", "isnt", "arent", "doesnt", "didnt",
})

# Reasoning-leak / placeholder phrases that mark a rewrite as an artifact (not a usable
# benign opposite). Kept broad on purpose so the detector stays reusable across rewrite
# batches even though the *current* paired file looks pre-filtered (see read-out).
_ARTIFACT_RE = re.compile(
    r"the user|wants me to|i need to|i should|i cannot|i can't|as an ai|"
    r"language model|here is (a |the |my )?(benign|rewritten|revised|version)|"
    r"rewritten version|revised version|sure,? here|<[^>]*>|\[[^\]]*\]|placeholder",
    re.IGNORECASE,
)
_PLACEHOLDER_RE = re.compile(r"<[^>]+>|\.\.\.\.|…|\[[^\]]+\]")

# Selection thresholds (documented so the notebook and FINDINGS comparison agree).
LOW_OVERLAP_THR = 0.20      # "topic drift / group swap" suspect below this Jaccard
MINNEG_OVERLAP_THR = 0.60   # clean minimal negation needs high overlap ...
MINNEG_SYMDIFF_MAX = 4      # ... and only a few changed tokens ...
#                             ... and a negation flipped on exactly one side.


# --------------------------------------------------------------------------------------
# Text primitives
# --------------------------------------------------------------------------------------
def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(text).lower())


def jaccard_overlap(a: str, b: str) -> float:
    """Token-set Jaccard overlap between two strings (1.0 if both empty)."""
    sa, sb = set(_tokens(a)), set(_tokens(b))
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def _is_blank(text: str) -> bool:
    s = str(text).strip().lower()
    return s in {"", "nan", "none"}


def is_artifact(text: str) -> bool:
    """Reasoning-leak phrase, placeholder, or empty rewrite."""
    if _is_blank(text):
        return True
    return bool(_ARTIFACT_RE.search(str(text)) or _PLACEHOLDER_RE.search(str(text)))


# --------------------------------------------------------------------------------------
# Analysis A — dataset composition
# --------------------------------------------------------------------------------------
def toxigen_category_counts() -> tuple[pd.DataFrame, dict]:
    """Per-``target_group`` counts for the ToxiGen paired set + overall composition.

    Each row of the paired file is one (toxic_text, benign_rewrite) pair, so
    n_examples = 2 * n_pairs (the toxic statement and its benign rewrite).
    """
    df = pd.read_csv(TOXIGEN_PAIRED)
    counts = (
        df["target_group"].value_counts().rename_axis("target_group")
        .reset_index(name="n_pairs").sort_values("n_pairs", ascending=False)
        .reset_index(drop=True)
    )
    counts["n_examples"] = counts["n_pairs"] * 2
    counts["pct_pairs"] = (counts["n_pairs"] / counts["n_pairs"].sum() * 100).round(1)
    overall = {
        "n_pairs": int(len(df)),
        "n_examples": int(len(df) * 2),
        "n_toxic": int(len(df)),
        "n_benign": int(len(df)),
        "n_groups": int(df["target_group"].nunique()),
    }
    return counts, overall


def mixed_composition() -> dict:
    """Total rows and polarity balance of the mixed dataset."""
    df = pd.read_csv(MIXED)
    pol = df["is_harmfull_opposition"].value_counts(dropna=False).to_dict()
    return {
        "n_rows": int(len(df)),
        "n_harmful_polarity_0": int(pol.get(0, 0)),
        "n_benign_polarity_1": int(pol.get(1, 0)),
    }


def mixed_category_counts() -> pd.DataFrame | None:
    """Per-category counts for the mixed dataset if the labelled CSV exists, else None.

    The labelled CSV is produced by ``runs/label_mixed_groups.py`` (Nemotron labelling,
    adds a ``target_group`` column with the 14 ToxiGen groups + ``other``).
    """
    if not MIXED_LABELED.exists():
        return None
    df = pd.read_csv(MIXED_LABELED)
    if "target_group" not in df.columns:
        return None
    return (
        df["target_group"].value_counts(dropna=False).rename_axis("target_group")
        .reset_index(name="n").sort_values("n", ascending=False).reset_index(drop=True)
    )


# --------------------------------------------------------------------------------------
# Analysis B — rewrite-noise metric
# --------------------------------------------------------------------------------------
def _length_anomaly(length_ratio: float) -> float:
    """0 when the rewrite matches the source length, ->1 as it differs by >=4x."""
    if length_ratio <= 0:
        return 1.0
    return min(1.0, abs(math.log2(length_ratio)) / 2.0)


def rewrite_noise_table() -> pd.DataFrame:
    """Per-rewrite noise signals + a combined ``noise_score`` in [0, 1].

    noise_score = clip( 0.5 * (1 - jaccard_overlap)   # lexical drift / topic / group-swap
                      + 0.4 * artifact                 # reasoning-leak / placeholder / empty
                      + 0.4 * unchanged                # rewrite == source -> polarity not flipped
                      + 0.1 * length_anomaly,          # extreme length mismatch
                      0, 1)

    Higher = noisier. Clean minimal negations (high overlap, a single negation flipped, no
    artifact, ~equal length) land near the bottom; artifacts / unchanged / group-swaps near
    the top. ``quality_flag`` is a single categorical read (artifact > unchanged >
    low_overlap > minimal_negation > ok) for quick filtering.
    """
    df = pd.read_csv(TOXIGEN_PAIRED)
    rows = []
    for _, r in df.iterrows():
        toxic, rewrite = str(r["toxic_text"]), str(r["benign_rewrite"])
        ta, ra = _tokens(toxic), _tokens(rewrite)
        sa, sb = set(ta), set(ra)
        union = sa | sb
        overlap = len(sa & sb) / len(union) if union else 1.0
        length_ratio = (len(ra) / len(ta)) if ta else float("nan")
        unchanged = toxic.strip().lower() == rewrite.strip().lower()
        artifact = is_artifact(rewrite)
        low_overlap = overlap < LOW_OVERLAP_THR
        neg_flipped = (NEGATION_TOKENS & sb) != (NEGATION_TOKENS & sa)
        minimal_negation = (
            overlap >= MINNEG_OVERLAP_THR
            and len(sa ^ sb) <= MINNEG_SYMDIFF_MAX
            and neg_flipped
            and not artifact
        )
        la = _length_anomaly(length_ratio) if ta else 1.0
        noise = float(np.clip(
            0.5 * (1 - overlap) + 0.4 * artifact + 0.4 * unchanged + 0.1 * la, 0.0, 1.0
        ))
        if artifact:
            flag = "artifact"
        elif unchanged:
            flag = "unchanged"
        elif low_overlap:
            flag = "low_overlap"
        elif minimal_negation:
            flag = "minimal_negation"
        else:
            flag = "ok"
        rows.append({
            "target_group": r["target_group"],
            "toxic_text": toxic,
            "benign_rewrite": rewrite,
            "jaccard_overlap": round(overlap, 4),
            "length_ratio": round(length_ratio, 3) if ta else float("nan"),
            "unchanged": unchanged,
            "artifact": artifact,
            "low_overlap": low_overlap,
            "minimal_negation": minimal_negation,
            "noise_score": round(noise, 4),
            "quality_flag": flag,
        })
    return pd.DataFrame(rows)


def rewrite_noise_summary(table: pd.DataFrame | None = None) -> dict:
    """Aggregate the per-rewrite table into headline noise rates (+ FINDINGS deltas)."""
    t = rewrite_noise_table() if table is None else table
    n = len(t)
    summary = {
        "n": int(n),
        "mean_overlap": round(float(t["jaccard_overlap"].mean()), 3),
        "median_overlap": round(float(t["jaccard_overlap"].median()), 3),
        "pct_artifact": round(float(t["artifact"].mean() * 100), 1),
        "pct_low_overlap": round(float(t["low_overlap"].mean() * 100), 1),
        "pct_unchanged": round(float(t["unchanged"].mean() * 100), 1),
        "pct_minimal_negation": round(float(t["minimal_negation"].mean() * 100), 1),
        "mean_noise_score": round(float(t["noise_score"].mean()), 3),
        "median_noise_score": round(float(t["noise_score"].median()), 3),
    }
    # FINDINGS §5d audit reference (qualitative pass over an earlier batch).
    summary["findings_ref"] = {
        "minimal_negation": 13.0, "artifact": 3.0, "low_overlap": 11.0,
    }
    summary["delta_vs_findings"] = {
        "minimal_negation": round(summary["pct_minimal_negation"] - 13.0, 1),
        "artifact": round(summary["pct_artifact"] - 3.0, 1),
        "low_overlap": round(summary["pct_low_overlap"] - 11.0, 1),
    }
    return summary


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def plot_dataset_category_counts(out_path: Path) -> Path:
    """Sorted bar chart of ToxiGen entries per category (+ mixed groups if labelled)."""
    counts, overall = toxigen_category_counts()
    mixed_cat = mixed_category_counts()
    ncols = 2 if mixed_cat is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7.5 * ncols, 5.4), squeeze=False)
    ax = axes[0, 0]
    ax.barh(counts["target_group"], counts["n_examples"], color="#1f77b4")
    ax.invert_yaxis()
    ax.set(xlabel="# examples (2 × pairs)",
           title=f"ToxiGen paired — entries per category\n"
                 f"{overall['n_pairs']} pairs / {overall['n_examples']} examples, "
                 f"{overall['n_groups']} groups")
    for y, v in enumerate(counts["n_examples"]):
        ax.text(v + 1, y, str(int(v)), va="center", fontsize=8)
    if mixed_cat is not None:
        ax2 = axes[0, 1]
        ax2.barh(mixed_cat["target_group"].astype(str), mixed_cat["n"], color="#2ca02c")
        ax2.invert_yaxis()
        ax2.set(xlabel="# rows", title="Mixed dataset — entries per category\n"
                                       "(Nemotron-labelled target_group)")
        for y, v in enumerate(mixed_cat["n"]):
            ax2.text(v + 1, y, str(int(v)), va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_rewrite_noise(out_path: Path, table: pd.DataFrame | None = None) -> Path:
    """Overlap histogram + quality-flag breakdown + noise-score histogram."""
    t = rewrite_noise_table() if table is None else table
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(t["jaccard_overlap"], bins=20, color="#1f77b4", edgecolor="white")
    ax[0].axvline(LOW_OVERLAP_THR, ls="--", c="#d62728", lw=1.5, label=f"low<{LOW_OVERLAP_THR}")
    ax[0].set(title="Jaccard overlap (toxic vs rewrite)", xlabel="overlap", ylabel="# rewrites")
    ax[0].legend(fontsize=8)

    order = ["minimal_negation", "ok", "low_overlap", "unchanged", "artifact"]
    vc = t["quality_flag"].value_counts()
    cats = [c for c in order if c in vc.index]
    colors = {"minimal_negation": "#2ca02c", "ok": "#1f77b4", "low_overlap": "#ff7f0e",
              "unchanged": "#9467bd", "artifact": "#d62728"}
    ax[1].bar(cats, [vc[c] for c in cats], color=[colors[c] for c in cats])
    ax[1].set(title="Rewrite quality flags", ylabel="# rewrites")
    ax[1].tick_params(axis="x", rotation=30, labelsize=8)
    for i, c in enumerate(cats):
        ax[1].text(i, vc[c], f"{vc[c]}\n({vc[c]/len(t)*100:.1f}%)", ha="center",
                   va="bottom", fontsize=8)

    ax[2].hist(t["noise_score"], bins=20, color="#8c564b", edgecolor="white")
    ax[2].axvline(t["noise_score"].mean(), ls="--", c="black", lw=1.5,
                  label=f"mean {t['noise_score'].mean():.2f}")
    ax[2].set(title="Per-rewrite noise score", xlabel="noise_score (↑ noisier)",
              ylabel="# rewrites")
    ax[2].legend(fontsize=8)
    fig.suptitle(f"ToxiGen benign-rewrite noise audit (n={len(t)})", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
# Verification entry point
# --------------------------------------------------------------------------------------
def main() -> None:
    plt.switch_backend("Agg")  # headless when run as a script; untouched on import
    FIGS.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    print("=" * 80)
    print("ANALYSIS A — DATASET COMPOSITION")
    print("=" * 80)
    counts, overall = toxigen_category_counts()
    print(f"ToxiGen paired: {overall['n_pairs']} pairs / {overall['n_examples']} examples "
          f"({overall['n_toxic']} toxic + {overall['n_benign']} benign), "
          f"{overall['n_groups']} groups")
    print(counts.to_string(index=False))

    mc = mixed_composition()
    print(f"\nMixed dataset: {mc['n_rows']} rows  "
          f"(polarity 0/harmful={mc['n_harmful_polarity_0']}, "
          f"1/benign={mc['n_benign_polarity_1']})")
    mcat = mixed_category_counts()
    if mcat is None:
        print("  mixed per-category: run runs/label_mixed_groups.py first (labelled CSV absent)")
    else:
        print("  mixed per-category counts (Nemotron-labelled):")
        print(mcat.to_string(index=False))

    print("\n" + "=" * 80)
    print("ANALYSIS B — TOXIGEN REWRITE NOISE")
    print("=" * 80)
    table = rewrite_noise_table()
    summary = rewrite_noise_summary(table)
    for k, v in summary.items():
        print(f"  {k:22s}: {v}")
    table.to_csv(REWRITE_NOISE_CSV, index=False)
    print(f"\nwrote {REWRITE_NOISE_CSV} ({len(table)} rows)")
    print("noisiest examples:")
    cols = ["target_group", "quality_flag", "jaccard_overlap", "noise_score", "benign_rewrite"]
    print(table.sort_values("noise_score", ascending=False)[cols].head(6).to_string(index=False))

    f1 = plot_dataset_category_counts(FIGS / "dataset_category_counts.png")
    f2 = plot_rewrite_noise(FIGS / "toxigen_rewrite_noise.png", table)
    print("\nfigures:")
    print("  ", f1)
    print("  ", f2)


if __name__ == "__main__":
    main()
