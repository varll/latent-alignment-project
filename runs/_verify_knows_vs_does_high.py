"""Standalone (Agg) verification for runs/knows_vs_does_high.py — no Jupyter kernel needed.

Confirms the band aggregation matches runs/summary_high_layers.csv, prints the per-model
band-mean values (acc / |PC| / CI), the bubbles / behavior-pending / absent split, and that the
figure saves to runs/figs/knows_vs_does_high_layers.png at dpi=300.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

RUNS = Path(__file__).resolve().parent
if str(RUNS) not in sys.path:
    sys.path.insert(0, str(RUNS))

import knows_vs_does_high as K  # noqa: E402
import pandas as pd  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

band = K.band_table()
print("=== band-aggregated latent metrics (per model) ===")
print(band.to_string(index=False))

# Cross-check band_mean_acc against high_layers' mean_acc_across_best.
ref = pd.read_csv(RUNS / "summary_high_layers.csv")[["model", "mean_acc_across_best", "n_high"]]
chk = band.merge(ref, on="model")
chk["acc_matches"] = (chk["band_mean_acc"].round(3) == chk["mean_acc_across_best"]).to_numpy()
chk["n_matches"] = (chk["n_high_x"] == chk["n_high_y"]).to_numpy()
print("\n=== cross-check vs summary_high_layers.csv ===")
print(chk[["model", "band_mean_acc", "mean_acc_across_best", "acc_matches",
           "n_high_x", "n_high_y", "n_matches"]].to_string(index=False))
assert chk["acc_matches"].all(), "band_mean_acc != mean_acc_across_best"
assert chk["n_matches"].all(), "n_high mismatch"

bubbles, pending, absent = K.build_plot_frames()
print("\n=== model split ===")
print("bubbles (latent + behavior):", list(bubbles["model"]))
print("behavior pending (latent-only):", list(pending["model"]))
print("absent (no latent run):", absent)

EXPECT_BUBBLES = {"OLMo-1B base", "OLMo-2-1B base", "OLMo-2-1B instruct",
                  "Gemma-3-1B base", "Gemma-3-1B instruct", "Qwen3-4B instruct"}
EXPECT_PENDING = {"Qwen3-8B base"}
assert set(bubbles["model"]) == EXPECT_BUBBLES, set(bubbles["model"])
assert set(pending["model"]) == EXPECT_PENDING, set(pending["model"])
assert absent == [], absent

print("\n=== per-model band-mean values used in the plot ===")
print("bubbles:")
print(bubbles[["model", "band_mean_acc", "band_pc", "band_ci", "coherent_rate",
               "harmful_rate_coherent"]].to_string(index=False))
print("pending:")
print(pending[["model", "band_mean_acc", "band_pc", "band_ci"]].to_string(index=False))

print("\n=== number -> model -> colour key (stable across both panels) ===")
key = K.model_key()
key["role"] = key["model"].apply(
    lambda m: "behavior pending" if m in set(pending["model"]) else "bubble")
print(key.to_string(index=False))
# Numbers and colours must be unique and cover all expected models (so the legend is unambiguous).
assert list(key["number"]) == list(range(1, len(K.EXPECTED) + 1)), "numbers not 1..N in order"
assert key["color"].nunique() == len(key), "duplicate model colours"
assert list(key["model"]) == K.EXPECTED, "key model order != A.MODELS order"
# Same global maps drive both panels, so the mapping is identical across panels by construction.
assert all(K.MODEL_NUM[m] == n for m, n in zip(key["model"], key["number"], strict=True))
assert all(K.MODEL_COLOR[m] == c for m, c in zip(key["model"], key["color"], strict=True))

report = K.render(dpi=300)
print("\n=== figures (mixed: combined + pc + ci) ===")
for tag, p in report["paths"].items():
    pp = Path(p)
    print(f"  [{tag:8s}] {pp.name}  exists={pp.exists()}  bytes={pp.stat().st_size}")
assert set(report["paths"]) == {"combined", "pc", "ci"}, report["paths"]
assert all(Path(p).exists() for p in report["paths"].values()), "a mixed figure is missing"
print("report:", {k: report[k] for k in ("bubbles", "pending", "absent", "delta")})

# ---------------------------------------------------------------------------------------
# ToxiGen per-group view (Qwen3-4B-instruct).
# ---------------------------------------------------------------------------------------
print("\n" + "#" * 80)
print("# ToxiGen per-group view")
print("#" * 80)

merged, latent_only, behavior_only = K.toxigen_group_table()
print("\n=== per-group table used in the ToxiGen plot ===")
print(merged[["target_group", "accuracy", "accuracy_mean", "accuracy_std", "harmful_rate",
              "abs_pc_mean", "n_examples", "n_out"]].to_string(index=False))
print("\nlatent-only groups dropped (no behavior):", latent_only or "none")
print("behavior-only groups dropped (no latent):", behavior_only or "none")

# Sanity: harmful_rate per group recomputed independently must match the helper.
bj = pd.read_csv(RUNS / "behavior_qwen3_4b_it" / "toxigen_paired_judge.csv")
bj["target_group"] = (bj["statement"].astype(str)
                      .str.extract(r"toxic for the (.+?)\n")[0].str.rstrip(". ").str.strip())
bj["is_harmful"] = (bj["judge_label"] == "harmful").astype(int)
ref = bj.groupby("target_group")["is_harmful"].mean()
for _, r in merged.iterrows():
    assert abs(r["harmful_rate"] - ref[r["target_group"]]) < 1e-9, r["target_group"]
print("\nharmful_rate matches independent groupby recomputation: OK")

print("\n=== number -> group -> colour key (ToxiGen) ===")
gkey = K.toxigen_group_key()
print(gkey.to_string(index=False))
assert list(gkey["number"]) == list(range(1, len(merged) + 1)), "group numbers not 1..N"
assert gkey["color"].nunique() == len(gkey), "duplicate group colours"

treport = K.render_toxigen(dpi=300)
tout = Path(treport["path"])
print("\n=== figure (toxigen) ===")
print("saved:", tout, "exists:", tout.exists(), "bytes:", tout.stat().st_size,
      "dpi:", treport["dpi"])
print("report:", {k: treport[k] for k in ("groups", "latent_only", "behavior_only")})
print("\nOK")
