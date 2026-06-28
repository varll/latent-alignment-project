"""Run PA-CCS for a decoder model on the mixed polarity dataset, writing a standard run dir.

A thin, memory-careful wrapper around the repo pipeline so we can add the **base** Qwen model
(`Qwen/Qwen3-8B-Base`) alongside the instruct one. It loads the model with an *explicit* dtype
(transformers 5.x renamed ``torch_dtype``; relying on the old kwarg can silently fall back to
float32 and OOM), extracts hidden states, trains PA-CCS per layer, and writes the same artifacts
as ``latent-align run``: ``ccs_summary.csv``, ``ccs_full_results.npz``, ``metadata.json``.

Defaults mirror the existing Qwen3-4B-instruct run (normalizing ``l2,median``, 1500 epochs, 10
tries, seed 0, 15% test split) so the base/instruct comparison is apples-to-apples.

    python runs/run_qwen8b_base.py --model Qwen/Qwen3-8B-Base \
        --output-dir runs/qwen3_8b_base_mixed --device mps --dtype bfloat16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--dataset", default=str(REPO / "data/polarity_probing/raw/mixed_dataset.csv"))
    ap.add_argument("--dataset-format", default="polarity_raw")
    ap.add_argument("--output-dir", default=str(REPO / "runs/qwen3_8b_base_mixed"))
    ap.add_argument("--device", default="mps", help="mps / cpu / cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--normalizing", default="l2,median")
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--random-state", type=int, default=71)
    ap.add_argument("--nepochs", type=int, default=1500)
    ap.add_argument("--ntries", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.015)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-device", default="cpu", help="device for CCS probe training")
    ap.add_argument("--trust-remote-code", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import gc

    from latent_alignment.ccs import ProbeConfig, summarize_results, train_ccs_layers
    from latent_alignment.data import load_dataset
    from latent_alignment.extract import extract_representation

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, dataset_format=args.dataset_format)
    print(f"dataset: {len(ds.positive_texts)} statements ({args.dataset_format})")

    print(f"loading {args.model} (dtype={args.dtype}, device={args.device}) …")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=args.trust_remote_code,
    )
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    def extract_all(texts: list[str], tag: str) -> np.ndarray:
        # Manual loop with periodic MPS-cache clearing: extract_texts() lets the MPS allocator's
        # cache grow unbounded over ~2.5k forward passes, which tips a 24GB Mac into swap.
        vectors = []
        for i, text in enumerate(texts):
            vectors.append(extract_representation(
                text, model, tokenizer, get_all_layers=True, strategy="last-token",
                model_kind="decoder", device=device, max_length=args.max_length))
            if (i + 1) % 50 == 0:
                if args.device == "mps":
                    torch.mps.empty_cache()
                print(f"  {tag}: {i + 1}/{len(texts)}", flush=True)
        if args.device == "mps":
            torch.mps.empty_cache()
        gc.collect()
        return np.stack(vectors).astype(np.float32)

    print("extracting hidden states (positive: ' Yes.') …", flush=True)
    positive = extract_all(ds.positive_texts, "positive")
    print("extracting hidden states (negative: ' No.') …", flush=True)
    negative = extract_all(ds.negative_texts, "negative")
    print(f"embeddings: positive {positive.shape}, negative {negative.shape}")

    # Free the LLM before the (CPU) probe training to reclaim memory.
    del model
    if args.device == "mps":
        torch.mps.empty_cache()
    elif args.device == "cuda":
        torch.cuda.empty_cache()

    train_idx, test_idx = ds.train_test_indices(test_size=args.test_size, random_state=args.random_state)
    config = ProbeConfig(nepochs=args.nepochs, ntries=args.ntries, lr=args.lr,
                         weight_decay=args.weight_decay, normalizing=args.normalizing, seed=args.seed)
    print(f"training PA-CCS over {positive.shape[1]} layers (normalizing={args.normalizing}) …")
    results = train_ccs_layers(
        positive, negative, ds.labels, np.asarray(train_idx), np.asarray(test_idx),
        config=config, opposite_indices=ds.opposite_indices, device=args.probe_device,
    )

    rows = summarize_results(results)
    pd.DataFrame(rows).to_csv(out_dir / "ccs_summary.csv", index=False)
    full = {f"layer_{layer}_{key}": value for layer, row in results.items() for key, value in row.items()}
    np.savez_compressed(out_dir / "ccs_full_results.npz", **full)
    (out_dir / "metadata.json").write_text(json.dumps({
        "model": args.model,
        "train_idx": np.asarray(train_idx).tolist(),
        "test_idx": np.asarray(test_idx).tolist(),
        "probe_config": config.__dict__,
    }, indent=2), encoding="utf-8")

    best = max(rows, key=lambda r: max(r["accuracy"], 1 - r["accuracy"]))
    print(f"done -> {out_dir}/ccs_summary.csv")
    print(f"best corrected acc {max(best['accuracy'], 1 - best['accuracy']):.3f} @ layer {best['layer']}")


if __name__ == "__main__":
    main()
