"""Knows-vs-does bubble plot with latent metrics aggregated over each model's HIGH-ACCURACY band.

Mirrors the "knows-vs-does coloured by latent polarity metrics" 2-panel scatter in
``notebooks/analysis_pa_ccs_vs_judges.ipynb``, but changes *how the latent half is aggregated*:

* **x** = the model's **mean** PA-CCS corrected accuracy over its **high-accuracy layer band**
  ("knows"), instead of the single best layer. The band is every layer within
  :data:`high_layers.HIGH_DELTA` of the model's best corrected accuracy -- the exact rule from
  ``runs/high_layers.py`` (``mean_acc_across_best`` in ``runs/summary_high_layers.csv``).
* **colour** = **band-mean |PC|** (left panel) and **band-mean CI** (right panel) -- the mean of
  ``abs(polar_consistency_mean)`` and ``contradiction_index_mean`` over those band layers, instead
  of the median over *all* layers. Both are "lower = better" (cmap ``RdYlGn_r``).
* **y** = ``harmful_rate_coherent`` ("does"); **bubble size** proportional to ``coherent_rate``.

Missing-values scheme matches ``runs/separability_grid.py`` /
``notebooks/latent_separability_grid.ipynb``: an explicit bottom-outside caption, measured against
the 7 registry models in :data:`analysis.MODELS`, that names which models are **bubbles** (have
judged behavior: OLMo-1B base, OLMo-2-1B base, OLMo-2-1B instruct, Gemma-3-1B base, Gemma-3-1B
instruct, Qwen3-4B instruct), which are **"behavior pending"** (latent-only, drawn as dashed
verticals: Qwen3-8B base), and which are **absent** (no PA-CCS latent run at all).

Pure pandas/numpy for the aggregation so it can be exercised by a standalone (``Agg``) script
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
import high_layers as H  # noqa: E402

# The 7 registry models every panel is measured against (the "expected" set).
EXPECTED = [r.name for r in A.MODELS]

# Knows-vs-does plotting window, kept identical to the original notebook cell.
XLIM = (0.6, 1.0)
YLIM = (-0.02, 0.2)

# Stable per-model identity: a 1-based number and a fixed colour, both indexed by the A.MODELS
# order so the number <-> model <-> colour mapping is identical across both panels and the legend.
# Colours are the matplotlib ``tab10`` hex values (hard-coded so no pyplot import is needed here).
_TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
MODEL_NUM = {r.name: i + 1 for i, r in enumerate(A.MODELS)}
MODEL_COLOR = {r.name: _TAB10[i % len(_TAB10)] for i, r in enumerate(A.MODELS)}

# A wider palette (tab20, dark shades first so adjacent entities stay distinct) for the ToxiGen
# per-group view, which can have ~15 target groups -- more than tab10 provides.
_TAB20 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
          "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
          "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"]


def assign_identity(labels) -> tuple[dict, dict]:
    """Stable 1-based number + a distinct colour per label, in the given order.

    Used to give each ToxiGen target group the same number/colour treatment the models get, so the
    number <-> entity <-> colour mapping is consistent between the plot and the legend.
    """
    order = list(labels)
    num = {name: i + 1 for i, name in enumerate(order)}
    color = {name: _TAB20[i % len(_TAB20)] for i, name in enumerate(order)}
    return num, color


# --------------------------------------------------------------------------------------
# Band aggregation: latent metrics over each model's high-accuracy layer band.
# --------------------------------------------------------------------------------------
def band_metrics(run: A.ModelRun, *, delta: float = H.HIGH_DELTA) -> dict | None:
    """Aggregate one model's latent metrics over its high-accuracy layer band.

    The band is every layer with ``acc_corrected >= best - delta`` -- the same rule as
    :func:`high_layers.high_layer_row`. Returns ``None`` if the model has no PA-CCS run on disk.
    """
    df = A.load_ccs(run)
    if df is None:
        return None
    d = df.copy()
    if "acc_corrected" not in d.columns:
        d["acc_corrected"] = np.maximum(d["accuracy"], 1 - d["accuracy"])
    best = float(d["acc_corrected"].max())
    band = d[d["acc_corrected"] >= best - delta]
    return {
        "model": run.name,
        "n_high": int(len(band)),
        # mean corrected accuracy over the band == high_layers' mean_acc_across_best.
        "band_mean_acc": float(band["acc_corrected"].mean()),
        # band-mean |PC| and band-mean CI (both "lower = better").
        "band_pc": float(band["polar_consistency_mean"].abs().mean()),
        "band_ci": float(band["contradiction_index_mean"].mean()),
    }


def band_table(*, delta: float = H.HIGH_DELTA) -> pd.DataFrame:
    """Band-aggregated latent metrics for every registry model with a PA-CCS run on disk."""
    rows = [m for m in (band_metrics(r, delta=delta) for r in A.MODELS) if m is not None]
    return pd.DataFrame(rows)


def build_plot_frames(*, delta: float = H.HIGH_DELTA):
    """Split the registry models into bubbles / behavior-pending / absent for the plot.

    Returns ``(bubbles, pending, absent)``:

    * ``bubbles`` -- DataFrame of models with BOTH latent + judged behavior (the scatter points),
      i.e. the band table joined to :func:`analysis.consistency_table`.
    * ``pending`` -- DataFrame of latent-only models (drawn as dashed "behavior pending" verticals).
    * ``absent``  -- list of registry models with no PA-CCS latent run at all.
    """
    band = band_table(delta=delta)
    con = A.consistency_table()  # models with both latent and behavior
    bubbles = con.merge(band, on="model", how="inner")
    pending = band[~band["model"].isin(con["model"])].copy()
    present_latent = set(band["model"])
    absent = [m for m in EXPECTED if m not in present_latent]
    return bubbles, pending, absent


# --------------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------------
# (metric column, panel title) -- both polarity metrics are "lower = better".
METRICS = [
    ("band_pc", "|PC| panel  \u00b7  colour = band-mean |polar consistency|  (\u2193 better)"),
    ("band_ci", "CI panel  \u00b7  colour = band-mean contradiction index  (\u2193 better)"),
]
_CBAR_LABEL = {
    "band_pc": "band-mean |PC|  (green = better)",
    "band_ci": "band-mean CI  (green = better)",
}
# Filename slug per metric for the standalone single-panel PNGs.
_METRIC_SLUG = {"band_pc": "pc", "band_ci": "ci"}


def model_key() -> pd.DataFrame:
    """The stable number -> model -> colour key (1-based, in A.MODELS order)."""
    return pd.DataFrame([
        {"number": MODEL_NUM[r.name], "model": r.name, "color": MODEL_COLOR[r.name]}
        for r in A.MODELS
    ])


def _number_label(ax, x, y, num, *, pe, color="black", dx=7, dy=7, ha="left", va="bottom"):
    """Draw an entity's number *beside* its marker/line (offset, white-outlined for readability)."""
    ax.annotate(str(num), (x, y), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va=va, fontsize=9, fontweight="bold", color=color, zorder=6,
                path_effects=[pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()])


def _entity_legend(fig, order, num, color, *, title, pending=frozenset(), ncol=4, y=0.06) -> None:
    """Combined bottom-outside key: number -> entity, swatch = entity colour.

    Bubbles show as a hollow circle with the entity's edge colour (the fill is metric-coded on the
    plot); behavior-pending entities (``pending``) show as a dashed line in the entity's colour.
    """
    import matplotlib.lines as mlines

    handles, labels = [], []
    for name in order:
        c = color[name]
        if name in pending:
            handles.append(mlines.Line2D([], [], color=c, ls="--", lw=2.4))
            labels.append(f"{num[name]} \u2014 {name}  (behavior pending)")
        else:
            handles.append(mlines.Line2D([], [], ls="none", marker="o", ms=10,
                                         markerfacecolor="white", markeredgecolor=c,
                                         markeredgewidth=2.4))
            labels.append(f"{num[name]} \u2014 {name}")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
               fontsize=8, frameon=True, title=title)


def _model_legend(fig, bubbles, pending) -> None:
    """Bottom-outside number -> model key, marking behavior-pending models (uses _entity_legend)."""
    _entity_legend(
        fig, EXPECTED, MODEL_NUM, MODEL_COLOR, pending=set(pending["model"]),
        title="model key  (number \u00b7 colour = model; used for the number & bubble-edge colour)")


def _caption(fig, bubbles, pending, absent) -> None:
    """Bottom-outside, explicit model-coverage caption (mirrors the separability-grid scheme)."""
    lines = [
        f"Model coverage (expected = {len(EXPECTED)} registry models):",
        "   \u2022 bubbles (latent + 3-judge behavior): "
        + (", ".join(bubbles["model"]) if len(bubbles) else "none"),
        "   \u2022 behavior pending (latent-only, dashed vertical): "
        + (", ".join(pending["model"]) if len(pending) else "none"),
        "   \u2022 absent (no PA-CCS latent run): "
        + (", ".join(absent) if absent else "none \u2014 all expected models have latent data"),
    ]
    fig.text(0.5, -0.16, "\n".join(lines), ha="center", va="top", fontsize=8,
             color="#444", style="italic")


def _draw_mixed_panel(ax, fig, metric, title, bubbles, pending, *, cmap, mpl, pe):
    """Draw one knows-vs-does panel (one polarity metric) into ``ax`` and attach its colorbar.

    Bubble **fill** is metric-coded; the bubble **edge** and the "behavior pending" dashed verticals
    are each model's stable colour. Numbers are placed *beside* the marker / line, not on top.
    """
    vals = pd.concat([bubbles[metric], pending[metric]])
    norm = mpl.colors.Normalize(vmin=float(vals.min()), vmax=float(vals.max()))
    bub_edges = [MODEL_COLOR[m] for m in bubbles["model"]]
    # Fill = metric (cmap+norm); edge = model colour; NUMBER label = model colour (identity).
    sc = ax.scatter(bubbles["band_mean_acc"], bubbles["harmful_rate_coherent"],
                    s=60 + 600 * bubbles["coherent_rate"], c=bubbles[metric],
                    cmap=cmap, norm=norm, edgecolors=bub_edges, linewidth=2.4, zorder=3)
    for _, r in bubbles.iterrows():
        _number_label(ax, r["band_mean_acc"], r["harmful_rate_coherent"],
                      MODEL_NUM[r["model"]], pe=pe, color=MODEL_COLOR[r["model"]], dx=8, dy=8)
    # Latent-only models: dashed vertical at band-mean accuracy, coloured by the METRIC (same
    # cmap+norm as the bubble fill / colorbar); the number carries the model identity.
    for i, (_, r) in enumerate(pending.iterrows()):
        ax.axvline(r["band_mean_acc"], color=cmap(norm(r[metric])), ls="--", lw=2.4, zorder=2)
        _number_label(ax, r["band_mean_acc"], YLIM[1] - 0.012 - 0.016 * i,
                      MODEL_NUM[r["model"]], pe=pe, color=MODEL_COLOR[r["model"]],
                      dx=5, dy=0, va="center")
    ax.axhline(0, ls=":", c="gray", lw=1)
    ax.set(xlabel="mean PA-CCS accuracy over high-acc layers ('knows')",
           ylabel="harmful rate among coherent ('does')", title=title, xlim=XLIM, ylim=YLIM)
    fig.colorbar(sc, ax=ax, label=_CBAR_LABEL[metric])


def make_figure(*, delta: float = H.HIGH_DELTA):
    """Build the combined 2-panel knows-vs-does figure (|PC| left, CI right) over the high-acc band.

    Returns ``(fig, report)`` where
    ``report`` = ``{"bubbles": [...], "pending": [...], "absent": [...], "delta": delta}``.
    """
    import matplotlib as mpl
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    bubbles, pending, absent = build_plot_frames(delta=delta)
    cmap = plt.get_cmap("RdYlGn_r")
    fig, axx = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, (metric, title) in zip(axx, METRICS, strict=True):
        _draw_mixed_panel(ax, fig, metric, title, bubbles, pending, cmap=cmap, mpl=mpl, pe=pe)

    subtitle = ("bubble fill / dashed line = polarity metric \u00b7 edge & number = model colour "
                "\u00b7 size = coherent rate")
    fig.suptitle(
        f"Knows-vs-does over the HIGH-ACCURACY layer band (within {delta:g} of best corrected acc)"
        f"\n{subtitle}", y=1.06)
    _model_legend(fig, bubbles, pending)
    _caption(fig, bubbles, pending, absent)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    return fig, {
        "bubbles": list(bubbles["model"]),
        "pending": list(pending["model"]),
        "absent": absent,
        "delta": delta,
    }


def make_panel_figure(metric: str, *, delta: float = H.HIGH_DELTA):
    """Build ONE polarity-metric panel as its own standalone figure.

    Keeps its own colorbar + bottom legend + missing caption. ``metric`` is ``"band_pc"`` or
    ``"band_ci"``. Returns ``(fig, report)``.
    """
    import matplotlib as mpl
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    title = dict(METRICS)[metric]
    bubbles, pending, absent = build_plot_frames(delta=delta)
    cmap = plt.get_cmap("RdYlGn_r")
    fig, ax = plt.subplots(figsize=(8, 6.4))
    _draw_mixed_panel(ax, fig, metric, title, bubbles, pending, cmap=cmap, mpl=mpl, pe=pe)

    fig.suptitle(
        f"Knows-vs-does over the HIGH-ACCURACY layer band (within {delta:g} of best corrected acc)",
        y=1.02)
    _model_legend(fig, bubbles, pending)
    _caption(fig, bubbles, pending, absent)
    fig.tight_layout(rect=(0, 0.16, 1, 0.98))
    return fig, {
        "metric": metric,
        "bubbles": list(bubbles["model"]),
        "pending": list(pending["model"]),
        "absent": absent,
        "delta": delta,
    }


def render(outdir: str | Path | None = None, dpi: int = 300, *,
           delta: float = H.HIGH_DELTA) -> dict:
    """Render + save the combined 2-panel figure AND the standalone |PC| / CI panels (high-res).

    Writes ``knows_vs_does_high_layers.png`` (combined) plus ``knows_vs_does_high_layers_pc.png``
    and ``knows_vs_does_high_layers_ci.png`` (one polarity metric each). Returns the combined
    report with a ``paths`` dict of every saved figure.
    """
    import matplotlib.pyplot as plt

    out = Path(outdir) if outdir is not None else (RUNS / "figs")
    out.mkdir(parents=True, exist_ok=True)

    fig, report = make_figure(delta=delta)
    combined = out / "knows_vs_does_high_layers.png"
    fig.savefig(combined, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    paths = {"combined": str(combined)}
    for metric in ("band_pc", "band_ci"):
        pfig, _ = make_panel_figure(metric, delta=delta)
        ppath = out / f"knows_vs_does_high_layers_{_METRIC_SLUG[metric]}.png"
        pfig.savefig(ppath, dpi=dpi, bbox_inches="tight")
        plt.close(pfig)
        paths[_METRIC_SLUG[metric]] = str(ppath)

    report["path"] = str(combined)
    report["paths"] = paths
    report["dpi"] = dpi
    return report


# ======================================================================================
# ToxiGen per-group view (Qwen3-4B-instruct only -- the lone model with ToxiGen behavior).
#
# A cross-model ToxiGen knows-vs-does is impossible (one model), so the analog is per TARGET
# GROUP: each bubble = a group. x = per-group latent k-fold accuracy ("knows"); y = per-group
# harmful-response rate ("does"); fill = per-group |PC| (abs_pc_mean, RdYlGn_r); size = n_examples.
# Same number / colour / legend scheme as the mixed view, generalized to groups.
# ======================================================================================
TOXIGEN_KFOLD = RUNS / "qwen3-4b-instruct" / "toxigen_group_kfold.csv"
TOXIGEN_JUDGE = RUNS / "behavior_qwen3_4b_it" / "toxigen_paired_judge.csv"


def toxigen_group_behavior(path: Path = TOXIGEN_JUDGE) -> pd.DataFrame:
    """Per-group harmful-response rate from the judge CSV -- mirrors the main notebook cell exactly.

    Recovers ``target_group`` from the prompt (``toxic for the {group}\\n...``), sets
    ``is_harmful = (judge_label == "harmful")``, then ``groupby(target_group).mean()``.
    """
    bj = pd.read_csv(path)
    bj["target_group"] = (bj["statement"].astype(str)
                          .str.extract(r"toxic for the (.+?)\n")[0].str.rstrip(". ").str.strip())
    bj["is_harmful"] = (bj["judge_label"] == "harmful").astype(int)
    return bj.groupby("target_group").agg(n_out=("is_harmful", "size"),
                                           harmful_rate=("is_harmful", "mean")).reset_index()


def toxigen_group_table(kfold_path: Path = TOXIGEN_KFOLD, judge_path: Path = TOXIGEN_JUDGE):
    """Inner-join per-group latent k-fold (knows) with per-group harmful rate (does).

    Returns ``(merged, latent_only, behavior_only)``: the plotted groups (sorted by harmful rate),
    plus the groups that dropped out of the inner merge from each side.
    """
    kf = pd.read_csv(kfold_path)
    out = toxigen_group_behavior(judge_path)
    merged = (kf.merge(out, on="target_group", how="inner")
              .sort_values("harmful_rate").reset_index(drop=True))
    latent_only = sorted(set(kf["target_group"]) - set(out["target_group"]))
    behavior_only = sorted(set(out["target_group"]) - set(kf["target_group"]))
    return merged, latent_only, behavior_only


def toxigen_group_key(kfold_path: Path = TOXIGEN_KFOLD, judge_path: Path = TOXIGEN_JUDGE):
    """The stable number -> group -> colour key for the ToxiGen per-group view."""
    merged, _, _ = toxigen_group_table(kfold_path, judge_path)
    num, color = assign_identity(merged["target_group"])
    key = pd.DataFrame({
        "number": [num[g] for g in merged["target_group"]],
        "target_group": merged["target_group"],
        "color": [color[g] for g in merged["target_group"]],
    })
    return key


def _toxigen_caption(fig, merged, latent_only, behavior_only) -> None:
    """Bottom-outside per-group coverage caption (which groups dropped from the inner merge)."""
    lines = [
        f"Per-group coverage (inner merge of latent k-fold \u00d7 judged behavior): "
        f"{len(merged)} groups plotted.",
        "   \u2022 dropped (latent k-fold only, no behavior rows): "
        + (", ".join(latent_only) if latent_only else "none"),
        "   \u2022 dropped (behavior only, no latent k-fold): "
        + (", ".join(behavior_only) if behavior_only else "none"),
    ]
    fig.text(0.5, -0.26, "\n".join(lines), ha="center", va="top", fontsize=8,
             color="#444", style="italic")


def make_toxigen_figure(kfold_path: Path = TOXIGEN_KFOLD, judge_path: Path = TOXIGEN_JUDGE):
    """Build the ToxiGen per-group knows-vs-does figure (Qwen3-4B-instruct).

    Returns ``(fig, report)``.
    x = per-group k-fold OOF accuracy (with accuracy_std error bars); y = per-group harmful rate;
    bubble fill = per-group |PC| (``abs_pc_mean``, ``RdYlGn_r``); size = ``n_examples``; edge = the
    group's stable colour; each bubble labelled with its number (see the bottom legend).
    """
    import matplotlib as mpl
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    merged, latent_only, behavior_only = toxigen_group_table(kfold_path, judge_path)
    num, color = assign_identity(merged["target_group"])
    cmap = plt.get_cmap("RdYlGn_r")
    norm = mpl.colors.Normalize(vmin=float(merged["abs_pc_mean"].min()),
                                vmax=float(merged["abs_pc_mean"].max()))

    fig, ax = plt.subplots(figsize=(9.5, 7))
    edges = [color[g] for g in merged["target_group"]]
    # Subtle horizontal error bars from the k-fold accuracy std (the "knows" uncertainty).
    ax.errorbar(merged["accuracy"], merged["harmful_rate"], xerr=merged["accuracy_std"],
                fmt="none", ecolor="0.6", elinewidth=1, capsize=2, zorder=2)
    sc = ax.scatter(merged["accuracy"], merged["harmful_rate"],
                    s=20 + 7 * merged["n_examples"], c=merged["abs_pc_mean"],
                    cmap=cmap, norm=norm, edgecolors=edges, linewidth=2.4, zorder=3)
    for _, r in merged.iterrows():
        _number_label(ax, r["accuracy"], r["harmful_rate"], num[r["target_group"]], pe=pe,
                      color=color[r["target_group"]])
    ax.axvline(0.5, ls=":", c="black", lw=1.2, zorder=1)  # chance / majority (50/50 per group)
    ax.set(xlabel="per-group latent k-fold accuracy ('knows')  \u2014  error bar = accuracy_std",
           ylabel="per-group harmful-response rate ('does')",
           title="Qwen3-4B-instruct \u00b7 ToxiGen \u00b7 knows-vs-does per target group")
    ax.margins(0.14)
    ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax, label="per-group |PC|  (abs_pc_mean; green = better)")

    fig.suptitle("ToxiGen per-group knows-vs-does  \u2014  bubble fill = |PC| \u00b7 "
                 "edge colour = group \u00b7 size = #examples", y=0.99, fontsize=11)
    _entity_legend(fig, list(merged["target_group"]), num, color, ncol=3, y=0.04,
                   title="group key  (number \u00b7 colour = target group; bubble edge colour)")
    _toxigen_caption(fig, merged, latent_only, behavior_only)
    fig.tight_layout(rect=(0, 0.18, 1, 0.97))
    return fig, {
        "groups": list(merged["target_group"]),
        "latent_only": latent_only,
        "behavior_only": behavior_only,
    }


def render_toxigen(outdir: str | Path | None = None, dpi: int = 300) -> dict:
    """Render + save the ToxiGen per-group figure under runs/figs/ (high-res)."""
    import matplotlib.pyplot as plt

    out = Path(outdir) if outdir is not None else (RUNS / "figs")
    out.mkdir(parents=True, exist_ok=True)
    fig, report = make_toxigen_figure()
    path = out / "knows_vs_does_high_layers_toxigen.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    report["path"] = str(path)
    report["dpi"] = dpi
    return report
