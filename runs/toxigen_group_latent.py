"""Per-(ToxiGen target group) latent PA-CCS summary, to pair with per-group behavior.

The per-layer ``ccs_summary*.csv`` files are aggregate only — they carry no per-example or
per-group breakdown. This script runs PA-CCS on the ToxiGen **paired** set with the *same* model
whose free generations we already judged (Qwen3-4B-instruct), then aggregates, **per target
group**, the latent quantities the summaries can't give us:

* ``abs_pc_mean``       — mean |polar consistency| over the group's pairs (lower = cleaner polarity)
* ``confidence_mean``   — mean PA-CCS probe confidence over the group's examples
* ``conf_decisiveness`` — mean ``2*|confidence - 0.5|`` (0 = always unsure, 1 = always decisive)
* ``accuracy``          — toxic-vs-benign separability within the group (orientation folded globally)

The behavior/output side (per-group harmful-response rate) is computed in the notebook from the
judge CSV; this script only produces the *latent* half so the two can be compared per group.

Writes ``runs/qwen3-4b-instruct/toxigen_group_latent.csv``.

Requires the GPU stack (``torch`` + ``transformers``) and the model weights — i.e. ``pip install
-e '.[gpu]'`` (or ``pip install torch transformers``) and access to the HF model. GPU strongly
recommended. Example:

    python runs/toxigen_group_latent.py \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --paired-csv data/toxigen/raw/toxigen_annotated_test_paired.csv \
        --output runs/qwen3-4b-instruct/toxigen_group_latent.csv \
        --normalizing l2,median --dtype bfloat16 --device cuda

NOTE: ``--model`` must be the same model used for the behavior generations, otherwise the latent
("knows") and output ("does") sides are not comparable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507",
                    help="HF id of the SAME model whose behavior was judged (default Qwen3-4B-instruct).")
    ap.add_argument("--paired-csv", default=str(REPO / "data/toxigen/raw/toxigen_annotated_test_paired.csv"))
    ap.add_argument("--output", default=str(REPO / "runs/qwen3-4b-instruct/toxigen_group_latent.csv"))
    ap.add_argument("--group-col", default="target_group")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    ap.add_argument("--dtype", default="bfloat16", choices=["auto", "float32", "float16", "bfloat16"])
    ap.add_argument("--normalizing", default="l2,median",
                    help="Normalization pipeline (match the original run for comparability).")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--layer", type=int, default=None,
                    help="Layer to summarise per group (default: best test-accuracy layer).")
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--random-state", type=int, default=71)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nepochs", type=int, default=1500)
    ap.add_argument("--ntries", type=int, default=10)
    ap.add_argument("--trust-remote-code", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Heavy imports kept local so the rest of the analysis stack stays importable without torch.
    from latent_alignment.ccs import CCS, _normalize_split, set_seed
    from latent_alignment.data import load_dataset
    from latent_alignment.extract import extract_texts, load_hf_model

    paired_csv = Path(args.paired_csv)
    df = pd.read_csv(paired_csv)
    if args.group_col not in df.columns:
        raise SystemExit(f"{paired_csv} has no '{args.group_col}' column; columns={list(df.columns)}")

    # _load_paired stacks positive_text then negative_text, so the group repeats across both halves.
    groups = np.asarray(list(df[args.group_col].astype(str)) * 2)

    ds = load_dataset(paired_csv, dataset_format="paired")
    labels = np.asarray(ds.labels, dtype=int)
    opposite = np.asarray(ds.opposite_indices, dtype=int)
    train_idx, test_idx = ds.train_test_indices(test_size=args.test_size, random_state=args.random_state)
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    print(f"Loading {args.model} (dtype={args.dtype}) …")
    model, tokenizer, device = load_hf_model(
        args.model, model_kind="decoder", device=args.device, dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    pos = extract_texts(ds.positive_texts, model, tokenizer, get_all_layers=True,
                        strategy="last-token", model_kind="decoder", device=device, max_length=args.max_length)
    neg = extract_texts(ds.negative_texts, model, tokenizer, get_all_layers=True,
                        strategy="last-token", model_kind="decoder", device=device, max_length=args.max_length)
    n_layers = pos.shape[1]
    set_seed(args.seed)

    # Unique pairs (a = larger index), matching train_ccs_layers' convention.
    all_pairs = [(i, int(opposite[i])) for i in range(len(opposite)) if i > opposite[i]]
    a_idx = np.asarray([p[0] for p in all_pairs], dtype=int)
    not_a_idx = np.asarray([p[1] for p in all_pairs], dtype=int)

    per_layer_acc: list[float] = []
    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for layer in range(n_layers):
        pos_tr, pos_te, neg_tr, neg_te = _normalize_split(
            pos[train_idx, layer], pos[test_idx, layer],
            neg[train_idx, layer], neg[test_idx, layer], args.normalizing,
        )
        ccs = CCS(neg_tr, pos_tr, labels[train_idx].astype(np.float32),
                  nepochs=args.nepochs, ntries=args.ntries, device=args.device)
        ccs.repeated_train()
        per_layer_acc.append(ccs.accuracy(neg_te, pos_te, labels[test_idx].astype(np.float32)))

        # Per-example confidence on ALL rows (train-fit normalization, like the test split).
        _, pos_all, _, neg_all = _normalize_split(
            pos[train_idx, layer], pos[:, layer], neg[train_idx, layer], neg[:, layer], args.normalizing,
        )
        preds, conf = ccs.predict(neg_all, pos_all)
        # |PC| per pair on ALL pairs (raw embeddings, exactly as train_ccs_layers does).
        pc = ccs.polar_consistency(
            neg[a_idx, layer], pos[a_idx, layer], neg[not_a_idx, layer], pos[not_a_idx, layer],
        )
        cache[layer] = (conf, preds, np.abs(pc))

    best_layer = args.layer if args.layer is not None else int(np.argmax(per_layer_acc))
    print(f"best test-accuracy layer = {best_layer} (acc {per_layer_acc[best_layer]:.3f}); "
          f"summarising per group at this layer")

    conf, preds, abs_pc = cache[best_layer]
    # Fold orientation globally so 1 = the toxic side (mirrors max(acc, 1-acc)).
    if (preds == labels).mean() < 0.5:
        preds = 1 - preds
    pair_groups = groups[a_idx]

    rows = []
    for grp in sorted(pd.unique(df[args.group_col].astype(str))):
        ex_mask = groups == grp
        pair_mask = pair_groups == grp
        rows.append({
            "target_group": grp,
            "n_pairs": int(pair_mask.sum()),
            "n_examples": int(ex_mask.sum()),
            "abs_pc_mean": float(abs_pc[pair_mask].mean()) if pair_mask.any() else float("nan"),
            "confidence_mean": float(conf[ex_mask].mean()) if ex_mask.any() else float("nan"),
            "conf_decisiveness": float(np.abs(conf[ex_mask] - 0.5).mean() * 2) if ex_mask.any() else float("nan"),
            "accuracy": float((preds[ex_mask] == labels[ex_mask]).mean()) if ex_mask.any() else float("nan"),
            "best_layer": best_layer,
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} groups -> {out}")


if __name__ == "__main__":
    main()
