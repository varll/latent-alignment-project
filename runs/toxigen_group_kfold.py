"""K-fold, per-(ToxiGen target group) latent PA-CCS evaluation.

This is the cross-validated successor to ``runs/toxigen_group_latent.py``. Two things change:

1. **K-fold cross-validation (no testing on training data).** Instead of a single 15% test split,
   every metric comes from K-fold *out-of-fold* (OOF) predictions: each example is scored only by
   the fold in which it was held out, so no example is ever scored by a probe trained on it.
   Folding is done on **unique polarity pairs** (an example and its ``opposite_indices`` partner)
   so a pair is never split across train/test (see :func:`latent_alignment.ccs.kfold_indices`).

2. **A probe per target group.** The old script trained ONE global probe and merely *sliced* the
   aggregate metrics per group. Here each target group additionally gets its **own** K-fold probe,
   trained and evaluated only on that group's pairs, so the per-group numbers reflect a
   group-specific decision boundary rather than a global one. For comparison we also keep the
   leak-free global-probe numbers, sliced per group, under the ``global_*`` columns.

Per group we report (out-of-fold, at the chosen layer):

* ``accuracy``           — pooled OOF toxic-vs-benign accuracy of the group's own probe
* ``accuracy_mean/std``  — mean ± std of the per-fold accuracies (the CV estimate)
* ``abs_pc_mean/std``    — mean |polar consistency| over the group's pairs (lower = cleaner)
* ``confidence_mean``    — mean OOF probe confidence over the group's examples
* ``conf_decisiveness``  — mean ``2*|confidence - 0.5|`` (0 = always unsure, 1 = always decisive)
* ``silhouette_mean/std``— cluster separability of held-out examples
* ``global_*``           — the same quantities from the single global probe, sliced to the group

Outputs (defaults under ``runs/qwen3-4b-instruct/``):

* ``--output``        per-group CSV (one row per target group)
* ``--layer-output``  per-layer CSV with fold mean ± std for the global probe (all layers)
* ``--meta-output``   JSON run metadata

Requires the GPU stack (``torch`` + ``transformers``) and the model weights — i.e. ``pip install
-e '.[gpu]'`` (or ``pip install torch transformers``) and access to the HF model. GPU strongly
recommended. Example:

    python runs/toxigen_group_kfold.py \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --paired-csv data/toxigen/raw/toxigen_annotated_test_paired.csv \
        --output runs/qwen3-4b-instruct/toxigen_group_kfold.csv \
        --n-folds 5 --normalizing l2,median --dtype bfloat16 --device cuda

NOTE: ``--model`` must be the SAME model used for the behavior generations, otherwise the
latent ("knows") and output ("does") sides are not comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507",
                    help="HF id of the SAME model whose behavior was judged (default Qwen3-4B).")
    ap.add_argument("--paired-csv",
                    default=str(REPO / "data/toxigen/raw/toxigen_annotated_test_paired.csv"))
    ap.add_argument("--output",
                    default=str(REPO / "runs/qwen3-4b-instruct/toxigen_group_kfold.csv"),
                    help="Per-group metrics CSV.")
    ap.add_argument("--layer-output", default=None,
                    help="Per-layer global-probe CV CSV (default beside --output).")
    ap.add_argument("--meta-output", default=None,
                    help="Run metadata JSON (default: <out dir>/toxigen_kfold_metadata.json).")
    ap.add_argument("--group-col", default="target_group")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu for the LLM (default: auto)")
    ap.add_argument("--probe-device", default=None,
                    help="device for CCS probe training (default: auto)")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["auto", "float32", "float16", "bfloat16"])
    ap.add_argument("--normalizing", default="l2,median",
                    help="Normalization pipeline (match the original run for comparability).")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--layer", type=int, default=None,
                    help="Layer to summarise per group (default: best global mean-accuracy layer).")
    ap.add_argument("--n-folds", type=int, default=5, help="Number of CV folds (>=2).")
    ap.add_argument("--random-state", type=int, default=71, help="Fold-shuffling seed.")
    ap.add_argument("--seed", type=int, default=0, help="Probe-training seed.")
    ap.add_argument("--nepochs", type=int, default=1500)
    ap.add_argument("--ntries", type=int, default=10)
    ap.add_argument("--embeddings-cache", default=None,
                    help="NPZ to cache/reuse extracted activations (default beside --output); "
                         "lets probe configs re-run without re-extracting. 'none' disables.")
    ap.add_argument("--trust-remote-code", action="store_true")
    return ap.parse_args()


def _group_metrics_from_oof(
    oof_pred: np.ndarray,
    oof_conf: np.ndarray,
    labels: np.ndarray,
    ex_mask: np.ndarray,
    oof_abs_pc: np.ndarray,
    pair_mask: np.ndarray,
) -> dict[str, float]:
    """Slice pooled out-of-fold arrays to one group and reduce to scalar metrics."""
    has_ex = bool(ex_mask.any())
    has_pair = bool(pair_mask.any())
    conf = oof_conf[ex_mask]
    nan = float("nan")
    return {
        "accuracy": float((oof_pred[ex_mask] == labels[ex_mask]).mean()) if has_ex else nan,
        "confidence_mean": float(np.nanmean(conf)) if has_ex else nan,
        "conf_decisiveness": float(np.nanmean(np.abs(conf - 0.5)) * 2) if has_ex else nan,
        "abs_pc_mean": float(np.nanmean(oof_abs_pc[pair_mask])) if has_pair else nan,
    }


def main() -> None:
    args = parse_args()

    # Heavy imports kept local so the rest of the analysis stack stays importable without torch.
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from latent_alignment.ccs import (
        ProbeConfig,
        summarize_kfold_results,
        train_ccs_layers_kfold,
    )
    from latent_alignment.data import load_dataset
    from latent_alignment.extract import extract_representation

    paired_csv = Path(args.paired_csv)
    df = pd.read_csv(paired_csv)
    if args.group_col not in df.columns:
        raise SystemExit(
            f"{paired_csv} has no '{args.group_col}' column; columns={list(df.columns)}"
        )

    # _load_paired stacks positive_text (toxic) then negative_text (benign), so the group repeats
    # across both halves: example i and its partner i+n_pairs share the same target group.
    groups = np.asarray(list(df[args.group_col].astype(str)) * 2)

    ds = load_dataset(paired_csv, dataset_format="paired")
    labels = np.asarray(ds.labels, dtype=int)
    opposite = np.asarray(ds.opposite_indices, dtype=int)

    cache_path = None
    if args.embeddings_cache != "none":
        cache_path = Path(args.embeddings_cache) if args.embeddings_cache else \
            Path(args.output).with_name("toxigen_kfold_embeddings.npz")

    if cache_path is not None and cache_path.exists():
        print(f"reusing cached activations -> {cache_path}")
        cached = np.load(cache_path)
        pos, neg = cached["positive"], cached["negative"]
    else:
        dtype_map = {"auto": "auto", "float32": torch.float32,
                     "float16": torch.float16, "bfloat16": torch.bfloat16}
        device = torch.device(
            args.device
            or ("mps" if torch.backends.mps.is_available()
                else ("cuda" if torch.cuda.is_available() else "cpu"))
        )
        print(f"Loading {args.model} (dtype={args.dtype}, device={device}) …")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=args.trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype_map[args.dtype], low_cpu_mem_usage=True,
            trust_remote_code=args.trust_remote_code,
        )
        model.to(device)
        model.eval()

        def extract_all(texts: list[str], tag: str, hf_model, hf_tokenizer) -> np.ndarray:
            # Per-text loop with periodic MPS-cache clearing so the allocator's cache doesn't grow
            # unbounded over ~1.6k forward passes and tip a 24GB Mac into heavy swap / OOM.
            vectors = []
            for i, text in enumerate(texts):
                vectors.append(extract_representation(
                    text, hf_model, hf_tokenizer, get_all_layers=True, strategy="last-token",
                    model_kind="decoder", device=device, max_length=args.max_length))
                # Clear the MPS allocator cache often (every 10) under tight memory; print sparser.
                if str(device) == "mps" and (i + 1) % 10 == 0:
                    torch.mps.empty_cache()
                if (i + 1) % 50 == 0:
                    print(f"  {tag}: {i + 1}/{len(texts)}", flush=True)
            if str(device) == "mps":
                torch.mps.empty_cache()
            gc.collect()
            return np.stack(vectors).astype(np.float32)

        print("extracting hidden states (positive: ' Yes.') …", flush=True)
        pos = extract_all(ds.positive_texts, "positive", model, tokenizer)
        print("extracting hidden states (negative: ' No.') …", flush=True)
        neg = extract_all(ds.negative_texts, "negative", model, tokenizer)

        # Free the LLM before the (CPU) probe training so the model weights don't sit in memory
        # alongside the activations and the k-fold probes (mirrors runs/run_qwen8b_base.py).
        del model
        if str(device) == "mps":
            torch.mps.empty_cache()
        elif str(device) == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, positive=pos, negative=neg)
            print(f"cached activations -> {cache_path}")

    n_layers = pos.shape[1]
    print(f"embeddings: positive {pos.shape}, negative {neg.shape}")

    config = ProbeConfig(nepochs=args.nepochs, ntries=args.ntries,
                         normalizing=args.normalizing, seed=args.seed)

    # --- Global probe, K-fold over all layers (leak-free successor to the old single split). ---
    print(f"global {args.n_folds}-fold PA-CCS over {n_layers} layers "
          f"(normalizing={args.normalizing}) …")
    global_results = train_ccs_layers_kfold(
        pos, neg, labels, n_splits=args.n_folds, config=config,
        opposite_indices=opposite, random_state=args.random_state, device=args.probe_device,
    )

    layer_rows = summarize_kfold_results(global_results)
    layer_output = Path(args.layer_output) if args.layer_output else \
        Path(args.output).with_name("toxigen_kfold_by_layer.csv")
    layer_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(layer_rows).to_csv(layer_output, index=False)
    print(f"wrote per-layer CV summary -> {layer_output}")

    best_layer = args.layer if args.layer is not None else \
        int(max(layer_rows, key=lambda r: r["accuracy_mean"])["layer"])
    print(f"summarising per group at layer {best_layer} "
          f"(global mean acc {global_results[best_layer]['accuracy_mean']:.3f})")

    g = global_results[best_layer]
    g_pred = np.asarray(g["oof_predictions"], dtype=int)
    g_conf = np.asarray(g["oof_confidence"], dtype=np.float32)
    g_abs_pc = np.abs(np.asarray(g["oof_polar_consistency"], dtype=np.float32))
    pair_groups = groups[np.asarray(g["pair_a_idx"], dtype=int)]

    # --- Per-group probes: each group gets its own K-fold probe at the chosen layer. ---
    rows = []
    for grp in sorted(pd.unique(df[args.group_col].astype(str))):
        ex_mask = groups == grp
        ex_idx = np.flatnonzero(ex_mask)
        n_examples = int(ex_mask.sum())
        n_pairs = n_examples // 2

        global_slice = _group_metrics_from_oof(
            g_pred, g_conf, labels, ex_mask, g_abs_pc, pair_groups == grp,
        )

        row = {
            "target_group": grp,
            "n_pairs": n_pairs,
            "n_examples": n_examples,
            "best_layer": best_layer,
            "global_accuracy": global_slice["accuracy"],
            "global_abs_pc_mean": global_slice["abs_pc_mean"],
            "global_confidence_mean": global_slice["confidence_mean"],
            "global_conf_decisiveness": global_slice["conf_decisiveness"],
        }

        group_splits = min(args.n_folds, n_pairs)
        if n_pairs < 2 or group_splits < 2:
            # Too few pairs to cross-validate a group-specific probe.
            nan = float("nan")
            row.update({
                "n_folds": 0,
                "accuracy": nan, "accuracy_mean": nan, "accuracy_std": nan,
                "silhouette_mean": nan, "silhouette_std": nan,
                "abs_pc_mean": nan, "abs_pc_std": nan,
                "confidence_mean": nan, "conf_decisiveness": nan,
            })
            rows.append(row)
            print(f"  {grp}: {n_pairs} pairs -> too few for CV, group probe skipped")
            continue

        # Slice activations to this group's examples and rebuild a local opposite-index map so
        # the group's pairs stay paired inside the subset.
        local_pos = {int(global_idx): local for local, global_idx in enumerate(ex_idx)}
        local_opposite = np.asarray([local_pos[int(opposite[gi])] for gi in ex_idx], dtype=int)

        grp_res = train_ccs_layers_kfold(
            pos[ex_idx], neg[ex_idx], labels[ex_idx], n_splits=group_splits, config=config,
            opposite_indices=local_opposite, layers=[best_layer],
            random_state=args.random_state, device=args.probe_device,
        )[best_layer]

        grp_conf = np.asarray(grp_res["oof_confidence"], dtype=np.float32)
        grp_pred = np.asarray(grp_res["oof_predictions"], dtype=int)
        grp_labels = labels[ex_idx]
        grp_abs_pc = np.abs(np.asarray(grp_res["oof_polar_consistency"], dtype=np.float32))
        row.update({
            "n_folds": group_splits,
            "accuracy": float((grp_pred == grp_labels).mean()),
            "accuracy_mean": float(grp_res["accuracy_mean"]),
            "accuracy_std": float(grp_res["accuracy_std"]),
            "silhouette_mean": float(grp_res["silhouette_mean"]),
            "silhouette_std": float(grp_res["silhouette_std"]),
            "abs_pc_mean": float(np.nanmean(grp_abs_pc)) if grp_abs_pc.size else float("nan"),
            "abs_pc_std": float(grp_res["abs_polar_consistency_std"]),
            "confidence_mean": float(np.nanmean(grp_conf)),
            "conf_decisiveness": float(np.nanmean(np.abs(grp_conf - 0.5)) * 2),
        })
        rows.append(row)
        print(f"  {grp}: {n_pairs} pairs, {group_splits}-fold -> acc {row['accuracy']:.3f} "
              f"(fold {row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f})")

    column_order = [
        "target_group", "n_pairs", "n_examples", "best_layer", "n_folds",
        "accuracy", "accuracy_mean", "accuracy_std",
        "silhouette_mean", "silhouette_std",
        "abs_pc_mean", "abs_pc_std",
        "confidence_mean", "conf_decisiveness",
        "global_accuracy", "global_abs_pc_mean",
        "global_confidence_mean", "global_conf_decisiveness",
    ]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows)[column_order].to_csv(out, index=False)
    print(f"wrote {len(rows)} groups -> {out}")

    meta_output = Path(args.meta_output) if args.meta_output else \
        out.with_name("toxigen_kfold_metadata.json")
    meta_output.write_text(json.dumps({
        "model": args.model,
        "paired_csv": str(paired_csv),
        "n_folds": args.n_folds,
        "random_state": args.random_state,
        "best_layer": best_layer,
        "n_layers": n_layers,
        "n_examples": int(len(labels)),
        "n_pairs": int(len(g["pair_a_idx"])),
        "probe_config": config.__dict__,
    }, indent=2), encoding="utf-8")
    print(f"wrote run metadata -> {meta_output}")


if __name__ == "__main__":
    main()
