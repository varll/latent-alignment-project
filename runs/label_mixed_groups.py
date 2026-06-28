"""Label the mixed-polarity statements with ToxiGen target groups via Nemotron Ultra.

Reads the mixed-polarity probing CSV (default
``data/polarity_probing/raw/mixed_dataset.csv`` with columns: an index column, ``statement``,
``is_harmfull_opposition``), asks NVIDIA Nemotron Ultra on OpenRouter which ToxiGen target group
each statement is about (or ``"other"``), and writes a copy of the CSV with an added
``target_group`` column (plus ``group_reason``). Original columns and row order are preserved.

The OpenRouter / ``openai`` client is imported lazily inside
:mod:`latent_alignment.group_label`, so this script imports without the SDK; you only need it
(and an ``OPENROUTER_API_KEY``) at run time.

Run it (this makes OpenRouter calls — confirm the model slug is current and your key has
access; the default ``nvidia/nemotron-3-ultra-550b-a55b:free`` is the slug confirmed working for
this project's ToxiGen rewrites):

    export OPENROUTER_API_KEY=sk-or-...
    .venv/bin/python runs/label_mixed_groups.py --output runs/mixed_dataset_group_labeled.csv

Small test run first (label only the first 20 rows):

    .venv/bin/python runs/label_mixed_groups.py --limit 20 \
        --output runs/mixed_dataset_group_labeled_sample.csv

Override the model if the slug changes:

    .venv/bin/python runs/label_mixed_groups.py --model nvidia/nemotron-3-ultra-550b-a55b:free \
        --output runs/mixed_dataset_group_labeled.csv
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from latent_alignment.group_label import (
    DEFAULT_CATEGORIES,
    DEFAULT_MODEL,
    EXTENDED_CATEGORIES,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO / "data/polarity_probing/raw/mixed_dataset.csv"
DEFAULT_OUTPUT = REPO / "runs/mixed_dataset_group_labeled.csv"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", default=str(DEFAULT_INPUT), help="Input mixed-dataset CSV.")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output labeled CSV.")
    ap.add_argument(
        "--model", default=DEFAULT_MODEL, help="OpenRouter model slug (default Nemotron Ultra)."
    )
    ap.add_argument("--statement-col", default="statement", help="Column holding the text.")
    ap.add_argument(
        "--extended",
        action="store_true",
        help="Use the EXTENDED taxonomy (ToxiGen-14 + immigrants, socioeconomic, age, "
        "appearance, other-religion, other-nationality, political) instead of ToxiGen-only.",
    )
    ap.add_argument(
        "--categories-file",
        default=None,
        help="Path to a text file of categories, one per line (overrides --extended/default). "
        "An 'other' bucket is appended automatically if absent.",
    )
    ap.add_argument("--max-workers", type=int, default=8, help="Concurrent OpenRouter requests.")
    ap.add_argument(
        "--limit", type=int, default=None, help="Only label the first N rows (for a test run)."
    )
    ap.add_argument(
        "--api-key", default=None, help="OpenRouter key (else read OPENROUTER_API_KEY env)."
    )
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=200)
    return ap.parse_args()


def resolve_categories(args: argparse.Namespace) -> tuple[str, ...]:
    """Pick taxonomy: --categories-file > --extended > ToxiGen default. Always ends in 'other'."""
    from latent_alignment.group_label import OTHER_CATEGORY

    if args.categories_file:
        lines = Path(args.categories_file).read_text(encoding="utf-8").splitlines()
        cats = [c.strip() for c in lines if c.strip()]
        if OTHER_CATEGORY not in cats:
            cats.append(OTHER_CATEGORY)
        return tuple(cats)
    if args.extended:
        return EXTENDED_CATEGORIES
    return DEFAULT_CATEGORIES


def main() -> None:
    args = parse_args()
    categories = resolve_categories(args)

    in_path = Path(args.input)
    df = pd.read_csv(in_path)
    if args.statement_col not in df.columns:
        raise SystemExit(
            f"{in_path} has no '{args.statement_col}' column; columns={list(df.columns)}"
        )
    if args.limit is not None:
        df = df.head(args.limit).copy()

    statements = df[args.statement_col].astype(str).tolist()
    print(
        f"Labeling {len(statements)} statements from {in_path} with {args.model} "
        f"({args.max_workers} workers, {len(categories)} categories incl. 'other') …"
    )

    # Imported here so the heavy OpenRouter path is only touched at run time.
    from latent_alignment.group_label import classify_groups

    results = classify_groups(
        statements,
        categories=categories,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_workers=args.max_workers,
    )

    df["target_group"] = [r["group"] for r in results]
    df["group_reason"] = [r["reason"] for r in results]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} labeled rows -> {out_path}")

    counts = Counter(df["target_group"].tolist())
    print("\nlabel counts per category:")
    for group, count in counts.most_common():
        print(f"  {count:5d}  {group}")


if __name__ == "__main__":
    main()
