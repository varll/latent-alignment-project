"""Standalone (Agg) verification for runs/behavior_outcomes_by_dataset.py -- no Jupyter kernel.

For each dataset (mixed / toxigen / toxigen_single) it prints the per-model numbers used in the
plot (outcome mix + l0/l1 asymmetry), shows the present-vs-missing model split, independently
re-derives the ToxiGen l0/l1 values from the judge CSV, and confirms the two behavior datasets save
their figures (dpi=300) while toxigen_single is an explicit N/A placeholder (no file saved).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

RUNS = Path(__file__).resolve().parent
if str(RUNS) not in sys.path:
    sys.path.insert(0, str(RUNS))

import behavior_outcomes_by_dataset as B  # noqa: E402
import pandas as pd  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

SHOW = ["model", "n", "safe_rate", "harmful_rate", "gibberish_rate", "tie_rate",
        "coherent_rate", "harmful_l0_stmt", "harmful_l1_neg"]

# --------------------------------------------------------------------------------------
# Per-dataset tables + coverage split.
# --------------------------------------------------------------------------------------
for ds in B.DATASETS:
    print("#" * 80)
    print(f"# dataset: {ds}   (has_behavior={B.HAS_BEHAVIOR[ds]})")
    print("#" * 80)
    tab = B.behavior_table(ds)
    present, missing = B.coverage(ds)
    if len(tab):
        print(tab[SHOW].to_string(index=False))
    else:
        print("(no judged free-generations -- N/A placeholder)")
    print("\npresent (judged behavior):", present or "none")
    print("missing behavior:         ", missing or "none")
    print()

# --------------------------------------------------------------------------------------
# Expected coverage split per dataset.
# --------------------------------------------------------------------------------------
MIXED_PRESENT = {"OLMo-1B base", "OLMo-2-1B base", "OLMo-2-1B instruct",
                 "Gemma-3-1B base", "Gemma-3-1B instruct", "Qwen3-4B instruct"}
MIXED_MISSING = {"Qwen3-8B base"}

p_mixed, m_mixed = B.coverage("mixed")
assert set(p_mixed) == MIXED_PRESENT, set(p_mixed)
assert set(m_mixed) == MIXED_MISSING, set(m_mixed)

p_tox, m_tox = B.coverage("toxigen")
assert set(p_tox) == {B.TOXIGEN_MODEL}, set(p_tox)
assert set(m_tox) == set(B.EXPECTED) - {B.TOXIGEN_MODEL}, set(m_tox)

p_ts, m_ts = B.coverage("toxigen_single")
assert p_ts == [], p_ts
assert set(m_ts) == set(B.EXPECTED), set(m_ts)
print("coverage split matches the expected present/missing scheme for all datasets: OK\n")

# --------------------------------------------------------------------------------------
# Independent re-derivation of the ToxiGen l0/l1 asymmetry from the judge CSV.
# --------------------------------------------------------------------------------------
print("=== ToxiGen l0/l1 derivation cross-check ===")
tj = pd.read_csv(B.TOXIGEN_JUDGE)
print("judge CSV columns:", list(tj.columns))
print("label value_counts:", tj["label"].value_counts().to_dict())
print("judge_label value_counts:", tj["judge_label"].value_counts().to_dict())

# l0 = harmful rate on the toxic statement (label 0); l1 = on the benign rewrite (label 1).
ref_l0 = float((tj[tj["label"] == 0]["judge_label"] == "harmful").mean())
ref_l1 = float((tj[tj["label"] == 1]["judge_label"] == "harmful").mean())
ref_safe = float((tj["judge_label"] == "safe").mean())
ref_harm = float((tj["judge_label"] == "harmful").mean())
ref_gib = float((tj["judge_label"] == "gibberish").mean())

tox_row = B.toxigen_table().iloc[0]
print(f"\nindependent: l0(label0,toxic stmt)={ref_l0:.3f}  l1(label1,benign rewrite)={ref_l1:.3f}")
print(f"helper:      l0={tox_row['harmful_l0_stmt']:.3f}  l1={tox_row['harmful_l1_neg']:.3f}")
print(f"independent: safe={ref_safe:.3f} harmful={ref_harm:.3f} gibberish={ref_gib:.3f} "
      f"tie=0 (single-judge, no tie column)")
assert abs(tox_row["harmful_l0_stmt"] - round(ref_l0, 3)) < 1e-9
assert abs(tox_row["harmful_l1_neg"] - round(ref_l1, 3)) < 1e-9
assert abs(tox_row["safe_rate"] - round(ref_safe, 3)) < 1e-9
assert abs(tox_row["harmful_rate"] - round(ref_harm, 3)) < 1e-9
assert abs(tox_row["gibberish_rate"] - round(ref_gib, 3)) < 1e-9
assert tox_row["tie_rate"] == 0.0
assert "tie" not in set(tj["judge_label"].unique())
print("ToxiGen outcome mix + l0/l1 match the independent recomputation: OK\n")

# --------------------------------------------------------------------------------------
# Render every dataset; check the two behavior figures save and toxigen_single is N/A.
# --------------------------------------------------------------------------------------
print("=== render ===")
reports = B.render_all(dpi=300)
for ds in B.DATASETS:
    r = reports[ds]
    if r["available"]:
        path = Path(r["path"])
        print(f"{ds}: saved {path}  exists={path.exists()}  bytes={path.stat().st_size}  "
              f"dpi={r['dpi']}")
        assert path.exists() and path.stat().st_size > 0
        assert path.name == f"behavior_outcomes_{ds}.png"
    else:
        print(f"{ds}: N/A placeholder (no file saved, path={r['path']})")
        assert r["path"] is None

assert reports["mixed"]["path"].endswith("behavior_outcomes_mixed.png")
assert reports["toxigen"]["path"].endswith("behavior_outcomes_toxigen.png")
assert reports["toxigen_single"]["path"] is None
print("\nOK")
