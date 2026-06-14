from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = Path(__file__).resolve().parent

MODELS = [
    ("olmo2 base", "olmo2_1b_base_mixed", "behavior_olmo2_1b_base"),
    ("olmo2 instruct", "olmo2_1b_instruct_mixed", "behavior_olmo2_1b_instruct"),
    ("gemma3 base", "gemma3_1b_base_mixed", "behavior_gemma3_1b_base"),
    ("gemma3 instruct", "gemma3_1b_instruct_mixed", "behavior_gemma3_1b_instruct"),
    ("olmo1", "olmo_1b_mixed", None),
]


def load_ccs(d: str) -> pd.DataFrame | None:
    path = RUNS / d / "ccs_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["acc_corrected"] = np.maximum(df["accuracy"], 1 - df["accuracy"])
    return df


def load_behavior(d: str | None) -> pd.DataFrame | None:
    if d is None:
        return None
    path = RUNS / d / "behavior_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def ccs_row(name: str, df: pd.DataFrame) -> dict:
    best = df.loc[df["acc_corrected"].idxmax()]
    return {
        "model": name,
        "n_layers": len(df),
        "best_acc": round(float(best["acc_corrected"]), 3),
        "best_layer": int(best["layer"]),
        "mean_acc": round(float(df["acc_corrected"].mean()), 3),
        "max_silhouette": round(float(df["silhouette"].max()), 3),
        "min_contradiction_idx": round(float(df["contradiction_index_mean"].min()), 3),
        "mean_contradiction_idx": round(float(df["contradiction_index_mean"].mean()), 3),
        "polar_consistency_absmean": round(float(df["polar_consistency_mean"].abs().mean()), 3),
    }


def behavior_row(name: str, df: pd.DataFrame) -> dict:
    ct = pd.crosstab(df["label"], df["stance"], normalize="index")
    no_l0 = float(ct.loc[0].get("no", 0.0))  # reject harmful statement
    yes_l1 = float(ct.loc[1].get("yes", 0.0))  # accept negated (benign) statement
    no_l1 = float(ct.loc[1].get("no", 0.0))
    return {
        "model": name,
        "n": len(df),
        "reject_harm_no|label0": round(no_l0, 3),
        "accept_neg_yes|label1": round(yes_l1, 3),
        "discrimination_dNo": round(no_l0 - no_l1, 3),
        "unclear_rate": round(float((df["stance"] == "unclear").mean()), 3),
        "overall_yes": round(float((df["stance"] == "yes").mean()), 3),
    }


def main() -> None:
    ccs = {name: load_ccs(d) for name, d, _ in MODELS}
    beh = {name: load_behavior(b) for name, _, b in MODELS}

    ccs_summary = pd.DataFrame(
        [ccs_row(n, df) for n, df in ccs.items() if df is not None]
    )
    beh_summary = pd.DataFrame(
        [behavior_row(n, df) for n, df in beh.items() if df is not None]
    )

    ccs_summary.to_csv(RUNS / "summary_ccs.csv", index=False)
    beh_summary.to_csv(RUNS / "summary_behavior.csv", index=False)
    print("=== summary_ccs.csv ===")
    print(ccs_summary.to_string(index=False))
    print("\n=== summary_behavior.csv ===")
    print(beh_summary.to_string(index=False))

    # --- plot 1: layerwise corrected accuracy ---
    styles = [
        {"color": "#1f77b4", "ls": "-", "marker": "o"},
        {"color": "#ff7f0e", "ls": "--", "marker": "s"},
        {"color": "#2ca02c", "ls": "-.", "marker": "^"},
        {"color": "#d62728", "ls": ":", "marker": "D"},
        {"color": "#9467bd", "ls": "-", "marker": "v"},
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    for (name, df), st in zip(
        ((n, d) for n, d in ccs.items() if d is not None), styles
    ):
        ax.plot(df["layer"], df["accuracy"], ms=5, lw=1.8, alpha=0.9,
                label=name, **st)
    ax.axhline(0.5, ls="--", c="gray", lw=1, label="chance")
    ax.set(xlabel="layer", ylabel="accuracy",
           title="Latent separability (CCS) by layer")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RUNS / "plot_ccs_accuracy.png", dpi=130)
    plt.close(fig)

    # --- plot 2: behavioral stance bars ---
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(beh_summary))
    w = 0.27
    ax.bar(x - w, beh_summary["reject_harm_no|label0"], w,
           label="says 'no' to a harmful statement")
    ax.bar(x, beh_summary["accept_neg_yes|label1"], w,
           label="says 'yes' to its safe negation")
    ax.bar(x + w, beh_summary["discrimination_dNo"], w,
           label="discrimination ΔNo  (depends on meaning)")
    ax.set_xticks(x)
    ax.set_xticklabels(beh_summary["model"], rotation=15, ha="right", fontsize=8)
    ax.set(ylabel="rate", title="Behavioral alignment (free generation)")
    ax.legend(fontsize=8)

    # olmo2 instruct was prompted few-shot -> stance likely biased toward "no".
    if "olmo2 instruct" in list(beh_summary["model"]):
        i = list(beh_summary["model"]).index("olmo2 instruct")
        ax.annotate(
            "few-shot prompting\n→ likely biased toward 'no'",
            xy=(i, beh_summary["reject_harm_no|label0"].iloc[i]),
            xytext=(i, 1.12),
            ha="center", va="bottom", fontsize=7.5, color="firebrick",
            arrowprops=dict(arrowstyle="->", color="firebrick", lw=1),
        )
    ax.set_ylim(top=1.25)

    fig.tight_layout()
    fig.savefig(RUNS / "plot_behavior.png", dpi=130)
    plt.close(fig)

    # --- plot 3: latent vs behavioral alignment ---
    merged = ccs_summary.merge(beh_summary, on="model", how="inner")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(merged["best_acc"], merged["discrimination_dNo"], s=60)
    for _, r in merged.iterrows():
        ax.annotate(r["model"], (r["best_acc"], r["discrimination_dNo"]),
                    fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax.set(xlabel="latent best accuracy (CCS)", ylabel="behavioral discrimination ΔNo",
           title="Latent vs behavioral alignment")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RUNS / "plot_latent_vs_behavior.png", dpi=130)
    plt.close(fig)

    print("\nwrote: summary_ccs.csv, summary_behavior.csv, "
          "plot_ccs_accuracy.png, plot_behavior.png, plot_latent_vs_behavior.png")


if __name__ == "__main__":
    main()
