"""Standalone verifier for runs/defense_metrics.py (no Jupyter kernel; Agg backend).

Prints every number the notebook / FINDINGS cite for the three defense-review action items and
renders the two new figures. Run:

    .venv/bin/python runs/_verify_defense_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

RUNS = Path(__file__).resolve().parent
if str(RUNS) not in sys.path:
    sys.path.insert(0, str(RUNS))

import defense_metrics as D  # noqa: E402


def main() -> None:
    figs = RUNS / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("TASK 1 — coherent_rate provenance")
    print("=" * 78)
    print(D.coherence_definition())
    print()
    vc = D.verify_coherent_rate()
    print(vc.to_string(index=False))
    assert (vc["consensus_match"] == 1.0).all(), "judge_3model_label is NOT a plain majority vote!"
    print("\nOK: re-derived majority reproduces stored judge_3model_label for every model.")
    print("\n-- thesis correlations (outlier sensitivity) --")
    print(D.coherence_thesis_correlations().to_string(index=False))

    print("\n" + "=" * 78)
    print("TASK 2 — pair types + CI conditioned on accuracy")
    print("=" * 78)
    ptypes = D.classify_pair_types()
    counts = ptypes["pair_type"].value_counts().to_dict()
    print(f"mixed set pair-type counts (n={len(ptypes)}): {counts}")
    print("  mean content-Jaccard  negation={:.2f}  concurrent={:.2f}".format(
        ptypes.loc[ptypes.pair_type == "negation", "content_jaccard"].mean(),
        ptypes.loc[ptypes.pair_type == "concurrent", "content_jaccard"].mean()))
    for run_dir, label in D.MIXED_PAIR_RUNS.items():
        df = D.mixed_pair_metrics(run_dir)
        if df is None:
            print(f"  {run_dir}: no per-pair metrics")
            continue
        tc = df[df.layer == df.layer.min()]["pair_type"].value_counts().to_dict()
        print(f"\n{label} ({run_dir}) — test pairs by type: {tc}")
        summ = D.ci_by_pairtype_summary(df)
        hi = summ[summ.acc_corrected >= 0.9]
        lo = summ[summ.acc_corrected < 0.6]
        print(f"  HIGH-acc layers (>=0.9): mean CI = {hi.ci_mean.mean():.3f}  "
              f"(n_layer-rows={len(hi)})")
        print(f"  LOW-acc  layers (<0.6):  mean CI = {lo.ci_mean.mean():.3f}  "
              f"(range {lo.ci_mean.min():.3f}-{lo.ci_mean.max():.3f})")
        best = summ.loc[summ.acc_corrected.idxmax(), ["pair_type", "pc_abs_mean"]].to_dict()
        print("  |PC| at best layer:", best if not summ.empty else "n/a")
        fig = D.plot_ci_by_pairtype(df, title=f"{label} · mixed")
        out = figs / "ci_by_pairtype.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  wrote {out}")

    print("\n" + "=" * 78)
    print("TASK 3 — signed polar consistency")
    print("=" * 78)
    signed = D.all_signed_pc()
    if signed.empty:
        print("no per-pair runs found")
    else:
        for run, sub in signed.groupby("run"):
            frac_pos = float((sub.pc_signed_mean > 0).mean())
            print(f"{run:<28} layers={len(sub):2d}  "
                  f"frac_pos(per-layer mean signed PC>0)={frac_pos:.2f}  "
                  f"mean signed PC={sub.pc_signed_mean.mean():+.3f}  "
                  f"|PC|={sub.pc_abs_mean.mean():.3f}  sign_flips={bool(sub.sign_flips.iloc[0])}")
        fig = D.plot_pc_sign(signed)
        out = figs / "pc_sign.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")

    print("\nAll verifications passed.")


if __name__ == "__main__":
    main()
