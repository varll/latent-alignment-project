"""Analysis layer for the PA-CCS (latent) vs LLM-judge (behavior) experiments.

This module is the single source of truth used by ``notebooks/analysis_pa_ccs_vs_judges.ipynb``.
It loads every run under ``runs/`` and answers the central question:

    "Are the internal PA-CCS metrics consistent with what the model actually generated?"

Two complementary views of the same models:

* **Latent ("the model knows")** — PA-CCS probes trained on paired activations
  (harmful statement vs. its benign negation). Per-layer summaries live in
  ``runs/<run>/ccs_summary*.csv`` with columns ``accuracy``, ``silhouette``,
  ``polar_consistency_mean``, ``contradiction_index_mean``.
* **Behavior ("the model does")** — free generations labelled by three independent
  LLM judges into ``safe`` / ``harmful`` / ``gibberish``. Per-example results live in
  ``runs/behavior_<run>/judge_3model_results_*.csv``.

Everything here is pure pandas/numpy so it can be unit-tested and re-run offline.
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path(__file__).resolve().parent

JUDGE_LABELS = ["safe", "harmful", "gibberish"]
JUDGE_COLORS = {"safe": "#2ca02c", "harmful": "#d62728", "gibberish": "#9e9e9e", "tie": "#ff7f0e"}

# The three judges that vote in the 3-model consensus (column prefixes in the judge CSVs).
JUDGE_COLS = {
    "deepseek": "judge_deepseek_deepseek_v4_flash_label",
    "gpt-oss-120b": "judge_openai_gpt_oss_120b_label",
    "qwen3": "judge_qwen_qwen3_7_plus_label",
}


@dataclass(frozen=True)
class ModelRun:
    """One model evaluated in the study (latent and/or behavior)."""

    name: str            # display name
    family: str          # olmo / gemma / qwen
    variant: str         # base / instruct
    params: str          # rough parameter count
    ccs_dir: str | None  # directory under runs/ holding ccs_summary*.csv
    behavior_dir: str | None  # directory under runs/ holding judge_3model_results_*.csv


# Registry of everything we have on disk. Order = how they appear in tables/plots.
MODELS: list[ModelRun] = [
    ModelRun("OLMo-1B base", "olmo", "base", "1B", "olmo_1b_mixed", "behavior_olmo_1b"),
    ModelRun("OLMo-2-1B base", "olmo2", "base", "1B", "olmo2_1b_base_mixed", "behavior_olmo2_1b_base"),
    ModelRun("OLMo-2-1B instruct", "olmo2", "instruct", "1B", "olmo2_1b_instruct_mixed", "behavior_olmo2_1b_it"),
    ModelRun("Gemma-3-1B base", "gemma3", "base", "1B", "gemma3_1b_base_mixed", None),
    ModelRun("Gemma-3-1B instruct", "gemma3", "instruct", "1B", "gemma3_1b_instruct_mixed", None),
    ModelRun("Gemma-4-E2B base", "gemma4", "base", "E2B", "gemma4_e2b_mixed", None),
    ModelRun("Qwen3-4B instruct", "qwen3", "instruct", "4B", "qwen3-4b-instruct", "behavior_qwen3_4b_it"),
    ModelRun("Qwen3-8B base", "qwen3", "base", "8B", "qwen3_8b_base_mixed", None),
]


# --------------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------------
def _find_ccs_csv(ccs_dir: str) -> Path | None:
    # Some runs were saved as "ccs_summary (1).csv"; match either.
    hits = sorted(glob.glob(str(RUNS / ccs_dir / "ccs_summary*.csv")))
    return Path(hits[0]) if hits else None


def load_ccs(run: ModelRun) -> pd.DataFrame | None:
    """Per-layer PA-CCS metrics with a sign-corrected accuracy column.

    CCS is unsupervised, so the probe can latch onto either polarity; ``acc_corrected``
    folds that ambiguity away with ``max(acc, 1 - acc)``.
    """
    if run.ccs_dir is None:
        return None
    path = _find_ccs_csv(run.ccs_dir)
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    df["acc_corrected"] = np.maximum(df["accuracy"], 1 - df["accuracy"])
    df["model"] = run.name
    return df


def load_behavior(run: ModelRun) -> pd.DataFrame | None:
    """Per-example free generations + 3-judge labels.

    ``label`` (a.k.a. ``is_harmfull_opposition``) is the *prompt* polarity:
    ``0`` = the hateful statement, ``1`` = its benign negation (paired by ``pair_id``).
    """
    if run.behavior_dir is None:
        return None
    hits = sorted(glob.glob(str(RUNS / run.behavior_dir / "judge_3model_results_*.csv")))
    if not hits:
        return None
    df = pd.read_csv(hits[0])
    df["model"] = run.name
    return df


def load_metadata(run: ModelRun) -> dict:
    if run.ccs_dir is None:
        return {}
    path = RUNS / run.ccs_dir / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


# --------------------------------------------------------------------------------------
# Latent (PA-CCS) summary
# --------------------------------------------------------------------------------------
def ccs_row(run: ModelRun, df: pd.DataFrame) -> dict:
    best = df.loc[df["acc_corrected"].idxmax()]
    return {
        "model": run.name,
        "family": run.family,
        "variant": run.variant,
        "params": run.params,
        "n_layers": int(len(df)),
        "best_acc": round(float(best["acc_corrected"]), 3),
        "best_layer": int(best["layer"]),
        "best_layer_frac": round(float(best["layer"]) / (len(df) - 1), 3),
        "mean_acc": round(float(df["acc_corrected"].mean()), 3),
        "max_silhouette": round(float(df["silhouette"].max()), 3),
        "mean_contradiction_idx": round(float(df["contradiction_index_mean"].mean()), 3),
        "polar_consistency_absmean": round(float(df["polar_consistency_mean"].abs().mean()), 3),
    }


def ccs_summary() -> pd.DataFrame:
    rows = []
    for run in MODELS:
        df = load_ccs(run)
        if df is not None:
            rows.append(ccs_row(run, df))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Top-layer selection (used by the layer-probe-similarity analysis)
# --------------------------------------------------------------------------------------
# Selection rule (documented once here so the notebook and the standalone script agree):
#   * size-aware top-K  — top-3 layers for "small" models (< SMALL_LAYER_CUTOFF layers,
#     i.e. the ~17-layer 1B models) and top-5 otherwise. A fixed K would over-count a
#     17-layer model (≈30% of its depth) relative to a 37-layer model (≈14%).
#   * threshold band    — every layer within ``margin`` corrected-accuracy of the best
#     layer (acc_corrected >= max_acc - margin). This adapts to flat vs. peaky accuracy
#     profiles but can return very few (peaky) or very many (flat) layers.
# The size-aware top-K is the PRIMARY rule because it yields a fixed, comparable number
# of layers per model for the cross-layer similarity heatmaps and the top-K-mean
# scorecard; the threshold band is reported alongside as a robustness check.
SMALL_LAYER_CUTOFF = 20
TOPK_SMALL = 3
TOPK_LARGE = 5
ACC_MARGIN = 0.02


def select_top_layers(
    df: pd.DataFrame,
    *,
    small_cutoff: int = SMALL_LAYER_CUTOFF,
    k_small: int = TOPK_SMALL,
    k_large: int = TOPK_LARGE,
    margin: float = ACC_MARGIN,
) -> dict:
    """Pick a model's "high-accuracy" PA-CCS layers two ways from its per-layer summary.

    Returns a dict describing both selections so callers can compare them:
    ``topk_layers`` (primary, size-aware top-K) and ``threshold_layers`` (within
    ``margin`` of the best corrected accuracy). ``acc_corrected`` is recomputed with
    ``max(acc, 1 - acc)`` if the column is absent (CCS is sign-ambiguous).
    """
    d = df.copy()
    if "acc_corrected" not in d.columns:
        d["acc_corrected"] = np.maximum(d["accuracy"], 1 - d["accuracy"])
    n = int(len(d))
    k = k_small if n < small_cutoff else k_large
    k = min(k, n)
    ranked = d.sort_values("acc_corrected", ascending=False)
    topk = sorted(int(x) for x in ranked.head(k)["layer"].tolist())
    max_acc = float(d["acc_corrected"].max())
    thr = max_acc - margin
    threshold = sorted(int(x) for x in d.loc[d["acc_corrected"] >= thr, "layer"].tolist())
    return {
        "n_layers": n,
        "k": k,
        "size_class": "small" if n < small_cutoff else "large",
        "max_acc": round(max_acc, 4),
        "margin": margin,
        "threshold": round(thr, 4),
        "topk_layers": topk,
        "threshold_layers": threshold,
    }


# --------------------------------------------------------------------------------------
# Random / chance baselines (null levels for every PA-CCS metric)
# --------------------------------------------------------------------------------------
# The PA-CCS task is binary (Yes/No, harmful vs. benign), so every performance number has a
# chance/null level. These helpers compute that reference so the notebook can draw "chance" lines
# and "random baseline" rows/columns, letting the reader see PA-CCS performance *above chance*.
# Everything here is pure numpy/sklearn and seeded (``RANDOM_SEED``) for reproducibility; all
# additions are backward-compatible (no existing function changed).
RANDOM_SEED = 0


def _as_labels(labels_or_n: np.ndarray | int) -> np.ndarray:
    """Accept an array of labels, or an int ``n`` (-> a balanced binary 0/1 label vector)."""
    if np.isscalar(labels_or_n):
        n = int(labels_or_n)  # type: ignore[arg-type]
        y = np.zeros(n, dtype=int)
        y[: n // 2] = 1
        return y
    return np.asarray(labels_or_n)


def random_accuracy_baseline(
    labels_or_n: np.ndarray | int,
    *,
    seed: int = RANDOM_SEED,
    n_repeats: int = 200,
    strategies: tuple[str, ...] = ("uniform", "stratified"),
) -> dict:
    """Empirical chance accuracy for a binary Yes/No task via ``sklearn.dummy.DummyClassifier``.

    For each strategy (``"uniform"`` = true random Yes/No, ``"stratified"`` = draw from the class
    prior) we fit a dummy classifier on ``labels`` and score it ``n_repeats`` times, reporting both
    the raw accuracy and the **sign-corrected** ``max(acc, 1 - acc)`` accuracy that CCS uses (CCS is
    sign-ambiguous). Corrected chance is slightly **above** 0.5 in finite samples — report the
    empirical value and draw ``theoretical_chance = 0.5`` as the asymptotic reference line.

    Returns a flat dict with ``n``, ``theoretical_chance`` and, per strategy, ``<s>_acc`` /
    ``<s>_acc_std`` / ``<s>_acc_corrected`` / ``<s>_acc_corrected_std``.
    """
    from sklearn.dummy import DummyClassifier

    y = _as_labels(labels_or_n)
    n = int(len(y))
    rng = np.random.default_rng(seed)
    x_dummy = np.zeros((n, 1))
    out: dict = {"n": n, "theoretical_chance": 0.5}
    for strat in strategies:
        accs, corr = [], []
        for _ in range(n_repeats):
            rs = int(rng.integers(0, 2**31 - 1))
            clf = DummyClassifier(strategy=strat, random_state=rs).fit(x_dummy, y)
            pred = clf.predict(x_dummy)
            acc = float(np.mean(pred == y))
            accs.append(acc)
            corr.append(max(acc, 1.0 - acc))
        out[f"{strat}_acc"] = float(np.mean(accs))
        out[f"{strat}_acc_std"] = float(np.std(accs))
        out[f"{strat}_acc_corrected"] = float(np.mean(corr))
        out[f"{strat}_acc_corrected_std"] = float(np.std(corr))
    return out


def majority_class_accuracy(labels: np.ndarray) -> float:
    """Accuracy of always predicting the most frequent class (the majority-class baseline)."""
    y = np.asarray(labels)
    if len(y) == 0:
        return float("nan")
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / len(y))


def _polar_consistency_np(p_a_neg, p_a_pos, p_not_a_neg, p_not_a_pos):
    """Pure-numpy mirror of ``CCS.polar_consistency`` (per-pair signed PC)."""
    return (
        0.5
        * ((p_a_pos - p_not_a_neg) ** 2 + (p_a_neg - p_not_a_pos) ** 2)
        * np.sign(p_a_pos - p_not_a_pos)
        * np.sign(p_not_a_neg - p_a_neg)
    )


def _contradiction_index_np(p_a_neg, p_a_pos, p_not_a_neg, p_not_a_pos):
    """Pure-numpy mirror of ``CCS.contradiction_index`` (per-pair CI)."""
    return p_a_pos * p_not_a_pos + p_a_neg * p_not_a_neg


def random_probe_polarity_baseline(
    n_pairs: int = 622, *, seed: int = RANDOM_SEED, n_repeats: int = 200
) -> dict:
    """Null level of |PC| / CI under a RANDOM probe (probe outputs ~ Uniform(0, 1)).

    Simulates the four contrast-pair probe outputs ``p(A.No), p(A.Yes), p(¬A.No), p(¬A.Yes)`` as
    independent ``Uniform(0, 1)`` draws over ``n_pairs`` pairs and applies the exact PA-CCS formulas
    (mirrored from ``latent_alignment/ccs.py``). This is what the polarity metrics look like with no
    signal: analytically ``E[CI] = 0.5`` and ``E[|PC|] = 1/6 ≈ 0.167``, so an observed ``CI ≈ 0.5``
    is essentially chance.

    Returns ``pc_mean`` (signed, ≈ 0), ``abs_pc_mean`` (≈ 0.167) and ``ci_mean`` (≈ 0.5) with stds.
    """
    rng = np.random.default_rng(seed)
    pc_means, abs_pc_means, ci_means = [], [], []
    for _ in range(n_repeats):
        p = rng.random((4, int(n_pairs)))
        pc = _polar_consistency_np(p[0], p[1], p[2], p[3])
        ci = _contradiction_index_np(p[0], p[1], p[2], p[3])
        pc_means.append(float(np.mean(pc)))
        abs_pc_means.append(float(np.mean(np.abs(pc))))
        ci_means.append(float(np.mean(ci)))
    return {
        "n_pairs": int(n_pairs),
        "pc_mean": float(np.mean(pc_means)),
        "pc_std": float(np.std(pc_means)),
        "abs_pc_mean": float(np.mean(abs_pc_means)),
        "abs_pc_std": float(np.std(abs_pc_means)),
        "ci_mean": float(np.mean(ci_means)),
        "ci_std": float(np.std(ci_means)),
    }


def shuffled_silhouette_baseline(
    X: np.ndarray | None = None,
    *,
    n_samples: int = 400,
    dim: int = 64,
    n_clusters: int = 2,
    n_repeats: int = 20,
    seed: int = RANDOM_SEED,
    metric: str = "cosine",
) -> dict:
    """Silhouette under RANDOM cluster assignment (the null is ≈ 0).

    Silhouette measures cluster separation; assigning points to clusters at random destroys any
    structure, so the score collapses toward 0 regardless of the data geometry. If ``X`` (a
    ``(n_samples, dim)`` point matrix, e.g. ``positive - negative`` activations) is given we shuffle
    labels over it; otherwise we use a reproducible standard-normal cloud (the null is
    geometry-agnostic). Returns ``silhouette_mean`` / ``silhouette_std`` over ``n_repeats`` shuffles
    plus ``theoretical`` (0.0).
    """
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, dim)) if X is None else np.asarray(X)
    n = len(X)
    scores = []
    for _ in range(n_repeats):
        lab = rng.integers(0, n_clusters, size=n)
        if len(np.unique(lab)) < 2:
            continue
        scores.append(float(silhouette_score(X, lab, metric=metric)))
    return {
        "silhouette_mean": float(np.mean(scores)) if scores else 0.0,
        "silhouette_std": float(np.std(scores)) if scores else 0.0,
        "theoretical": 0.0,
        "n": int(n),
    }


def random_baseline_row(
    acc_base: dict | None = None,
    sil_base: dict | None = None,
    pc_base: dict | None = None,
    *,
    model: str = "RANDOM (chance)",
) -> dict:
    """A ``ccs_summary``-shaped row holding the random/chance null for each PA-CCS metric.

    Pass pre-computed baseline dicts (from the helpers above) or leave them ``None`` to compute
    defaults. Columns with no chance analogue (layer indices, family) are blank/NaN so the row drops
    cleanly into the latent tables as a reference. Uses the ``uniform`` (true random Yes/No)
    corrected accuracy for ``best_acc`` / ``mean_acc``.
    """
    acc_base = acc_base if acc_base is not None else random_accuracy_baseline(1244)
    sil_base = sil_base if sil_base is not None else shuffled_silhouette_baseline()
    pc_base = pc_base if pc_base is not None else random_probe_polarity_baseline()
    return {
        "model": model,
        "family": "—",
        "variant": "—",
        "params": "—",
        "n_layers": np.nan,
        "best_acc": round(acc_base["uniform_acc_corrected"], 3),
        "best_layer": np.nan,
        "best_layer_frac": np.nan,
        "mean_acc": round(acc_base["uniform_acc_corrected"], 3),
        "max_silhouette": round(sil_base["silhouette_mean"], 3),
        "mean_contradiction_idx": round(pc_base["ci_mean"], 3),
        "polar_consistency_absmean": round(pc_base["abs_pc_mean"], 3),
    }


# --------------------------------------------------------------------------------------
# Judge reliability
# --------------------------------------------------------------------------------------
def _harmful_rate(s: pd.Series) -> float:
    return float((s == "harmful").mean())


def judge_reliability() -> pd.DataFrame:
    """How much the three judges (and the single legacy judge) agree per model.

    Surfaces the issue that judges disagree a lot on "harmful", and the
    legacy single ``judge_label`` column is unreliable (e.g. all-``gibberish`` for Qwen).
    """
    rows = []
    for run in MODELS:
        df = load_behavior(run)
        if df is None:
            continue
        n = len(df)
        row = {"model": run.name, "n": n}
        for short, col in JUDGE_COLS.items():
            row[f"harmful_{short}"] = round(_harmful_rate(df[col]), 3) if col in df else float("nan")
        row["all_agree_rate"] = round(float(df["judge_3model_all_agree"].mean()), 3)
        row["tie_rate"] = round(float((df["judge_3model_label"] == "tie").mean()), 3)
        row["fleiss_kappa"] = round(fleiss_kappa(df), 3)
        if "judge_label" in df:
            row["single_judge_harmful"] = round(_harmful_rate(df["judge_label"]), 3)
            row["single_judge_gibberish"] = round(float((df["judge_label"] == "gibberish").mean()), 3)
        row["consensus_harmful"] = round(_harmful_rate(df["judge_3model_label"]), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def fleiss_kappa(df: pd.DataFrame) -> float:
    """Fleiss' kappa over the 3 judges' labels (chance-corrected inter-rater agreement)."""
    cats = JUDGE_LABELS
    cols = [c for c in JUDGE_COLS.values() if c in df]
    if len(cols) < 2:
        return float("nan")
    counts = np.zeros((len(df), len(cats)), dtype=float)
    for ci, cat in enumerate(cats):
        counts[:, ci] = sum((df[c] == cat).to_numpy(dtype=float) for c in cols)
    n_raters = len(cols)
    p_i = (counts * (counts - 1)).sum(axis=1) / (n_raters * (n_raters - 1))
    p_bar = p_i.mean()
    p_j = counts.sum(axis=0) / (len(df) * n_raters)
    p_e = (p_j**2).sum()
    if np.isclose(p_e, 1.0):
        return 1.0
    return float((p_bar - p_e) / (1 - p_e))


def judge_harmful_long() -> pd.DataFrame:
    """Long-format per-judge harmful rate for grouped bar charts (judge x model)."""
    rows = []
    for run in MODELS:
        df = load_behavior(run)
        if df is None:
            continue
        for short, col in JUDGE_COLS.items():
            if col in df:
                rows.append({"model": run.name, "judge": short, "harmful_rate": _harmful_rate(df[col])})
        rows.append({"model": run.name, "judge": "consensus", "harmful_rate": _harmful_rate(df["judge_3model_label"])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Behavior summary (3-model consensus = the reliable signal)
# --------------------------------------------------------------------------------------
def behavior_row(run: ModelRun, df: pd.DataFrame, label_col: str = "judge_3model_label") -> dict:
    n = len(df)
    rate = df[label_col].value_counts(normalize=True)
    safe = float(rate.get("safe", 0.0))
    harmful = float(rate.get("harmful", 0.0))
    gibberish = float(rate.get("gibberish", 0.0))
    tie = float(rate.get("tie", 0.0))
    coherent = safe + harmful
    by_label = df.groupby("label")[label_col]
    return {
        "model": run.name,
        "n": n,
        "safe_rate": round(safe, 3),
        "harmful_rate": round(harmful, 3),
        "gibberish_rate": round(gibberish, 3),
        "tie_rate": round(tie, 3),
        "coherent_rate": round(coherent, 3),
        # Harmful rate restricted to coherent answers — fair to models that mostly emit noise.
        "harmful_rate_coherent": round(harmful / coherent, 3) if coherent else float("nan"),
        # Asymmetry: harmful output when prompted with the hateful statement (0) vs its negation (1).
        "harmful_l0_stmt": round(_harmful_rate(by_label.get_group(0)), 3) if 0 in df["label"].values else float("nan"),
        "harmful_l1_neg": round(_harmful_rate(by_label.get_group(1)), 3) if 1 in df["label"].values else float("nan"),
    }


def behavior_summary(label_col: str = "judge_3model_label") -> pd.DataFrame:
    rows = []
    for run in MODELS:
        df = load_behavior(run)
        if df is not None:
            rows.append(behavior_row(run, df, label_col=label_col))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Consistency: latent "knows" vs behavior "does"
# --------------------------------------------------------------------------------------
def consistency_table() -> pd.DataFrame:
    """Join latent separability with behavioral harmfulness for every model that has both."""
    ccs = ccs_summary()
    beh = behavior_summary()
    merged = ccs.merge(beh, on="model", how="inner")
    merged["latent_minus_behavior"] = (merged["best_acc"] - merged["harmful_rate_coherent"]).round(3)
    return merged


def consistency_correlations(table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cross-model correlations between latent metrics and behavioral harmfulness.

    Few models -> treat as descriptive, not inferential. Coherent-only behavior is the
    fair target because gibberish-dominated runs have no meaningful behavior to align with.
    """
    from scipy.stats import pearsonr, spearmanr

    t = consistency_table() if table is None else table
    pairs = [
        ("best_acc", "harmful_rate_coherent"),
        ("best_acc", "harmful_rate"),
        ("mean_acc", "harmful_rate_coherent"),
        ("max_silhouette", "harmful_rate_coherent"),
        ("best_acc", "coherent_rate"),
    ]
    rows = []
    for x, y in pairs:
        sub = t[[x, y]].dropna()
        if len(sub) >= 3 and sub[x].nunique() > 1 and sub[y].nunique() > 1:
            r, _ = pearsonr(sub[x], sub[y])
            rho, _ = spearmanr(sub[x], sub[y])
            rows.append({"x": x, "y": y, "n": len(sub), "pearson_r": round(r, 3), "spearman_rho": round(rho, 3)})
        else:
            rows.append({"x": x, "y": y, "n": len(sub), "pearson_r": float("nan"), "spearman_rho": float("nan")})
    return pd.DataFrame(rows)


def latent_behavior_scorecard() -> pd.DataFrame:
    """One row per model that has both views: PA-CCS metrics next to behavioral outcomes.

    This is the table that directly answers "are the internal metrics consistent with what the
    model generated?" — read left (latent) against right (behavior).
    """
    ccs = ccs_summary()
    beh = behavior_summary()
    flip = negation_flip_summary()[["model", "neg_flip_rate"]]
    m = ccs.merge(beh, on="model", how="inner").merge(flip, on="model", how="left")
    cols = ["model", "params",
            # latent (PA-CCS)
            "best_acc", "mean_acc", "max_silhouette",
            "polar_consistency_absmean", "mean_contradiction_idx",
            # behavior (judges)
            "coherent_rate", "harmful_rate", "harmful_rate_coherent", "neg_flip_rate"]
    return m[cols]


# Each PA-CCS metric paired with the behavioral signal it *should* predict, with the expected sign.
_METRIC_CHECKS = [
    ("best_acc", "coherent_rate", +1, "separability ⇒ coherent output"),
    ("best_acc", "harmful_rate_coherent", -1, "separability ⇒ safer output"),
    ("mean_acc", "harmful_rate_coherent", -1, "separability ⇒ safer output"),
    ("max_silhouette", "harmful_rate_coherent", -1, "cluster separation ⇒ safer output"),
    ("polar_consistency_absmean", "harmful_rate_coherent", +1, "poor polarity ⇒ more harm"),
    ("polar_consistency_absmean", "neg_flip_rate", +1, "poor latent polarity ⇒ behavioral negation flip"),
    ("mean_contradiction_idx", "harmful_rate_coherent", +1, "more contradiction ⇒ more harm"),
]


def metric_consistency_verdicts() -> pd.DataFrame:
    """Cross-model verdict per (PA-CCS metric -> behavior) pair: does the data agree in sign?

    Descriptive only (n = number of models with both views). A metric whose values barely vary
    across models (e.g. the contradiction index, ~0.5 everywhere) is flagged as uninformative,
    because its correlation is then driven by noise rather than signal.
    """
    from scipy.stats import pearsonr, spearmanr

    m = latent_behavior_scorecard()
    rows = []
    for x, y, exp_sign, desc in _METRIC_CHECKS:
        sub = m[[x, y]].dropna()
        cv = float(sub[x].std() / abs(sub[x].mean())) if sub[x].mean() else float("nan")
        if len(sub) >= 3 and sub[x].nunique() > 1 and sub[y].nunique() > 1:
            r, _ = pearsonr(sub[x], sub[y])
            rho, _ = spearmanr(sub[x], sub[y])
        else:
            r = rho = float("nan")
        if cv < 0.1:
            verdict = "uninformative (≈constant across models)"
        elif np.isnan(r):
            verdict = "n/a"
        elif np.sign(r) == exp_sign and abs(r) >= 0.5:
            verdict = "consistent"
        elif np.sign(r) == exp_sign:
            verdict = "weakly consistent"
        else:
            verdict = "inconsistent"
        rows.append({
            "pa_ccs_metric": x, "behavior": y, "expected_sign": "+" if exp_sign > 0 else "−",
            "pearson_r": round(r, 2), "spearman_rho": round(rho, 2),
            "x_cv": round(cv, 2), "verdict": verdict, "reading": desc,
        })
    return pd.DataFrame(rows)


def coherent_only(run: ModelRun) -> pd.DataFrame | None:
    df = load_behavior(run)
    if df is None:
        return None
    return df[df["judge_3model_label"].isin(["safe", "harmful"])].copy()


# --------------------------------------------------------------------------------------
# Qualitative peek: the negation flip (next-step #2)
# --------------------------------------------------------------------------------------
def flagged_examples(
    run: ModelRun,
    *,
    label: int | None = None,
    consensus: str = "harmful",
) -> pd.DataFrame:
    """Per-example rows where the 3-model consensus == ``consensus`` (optionally for one polarity).

    ``label=1`` selects the *benign negation* prompts — the ones whose harmful generations are the
    "negation flip" we want to read by hand. Adds ``gen_len`` (chars) and a compact ``votes`` string
    so we can tell genuine harmful leakage from strict-judge artefacts on terse refusals.
    """
    df = load_behavior(run)
    if df is None:
        return pd.DataFrame()
    sel = df[df["judge_3model_label"] == consensus].copy()
    if label is not None:
        sel = sel[sel["label"] == label]
    sel["gen_len"] = sel["generation"].astype(str).str.len()
    if len(sel) == 0:
        sel["votes"] = pd.Series(dtype=str)
        return sel
    sel["votes"] = (
        "ds=" + sel[JUDGE_COLS["deepseek"]].str[:4]
        + " oss=" + sel[JUDGE_COLS["gpt-oss-120b"]].str[:4]
        + " qw=" + sel[JUDGE_COLS["qwen3"]].str[:4]
    )
    return sel


def negation_flip_summary() -> pd.DataFrame:
    """Quantify the negation flip per model: harmful on hateful statement vs on its benign negation.

    ``terse_share_neg`` = fraction of the negation-flip harmful generations that are very short
    (<= 40 chars, e.g. bare "No"/"False"); a low value means the harm is in long, coherent text
    (genuine leakage) rather than a judge artefact on terse refusals.
    """
    rows = []
    for run in MODELS:
        df = load_behavior(run)
        if df is None:
            continue
        n0 = int((df["label"] == 0).sum())
        n1 = int((df["label"] == 1).sum())
        h0 = flagged_examples(run, label=0)
        h1 = flagged_examples(run, label=1)
        terse = float((h1["gen_len"] <= 40).mean()) if len(h1) else float("nan")
        rows.append({
            "model": run.name,
            "harmful_on_statement(l0)": len(h0),
            "harmful_on_negation(l1)": len(h1),
            "flip_ratio_l1_over_l0": round(len(h1) / len(h0), 2) if len(h0) else float("inf"),
            "neg_flip_rate": round(len(h1) / n1, 3) if n1 else float("nan"),
            "median_gen_len_neg": int(h1["gen_len"].median()) if len(h1) else 0,
            "terse_share_neg": round(terse, 3),
        })
    return pd.DataFrame(rows)
