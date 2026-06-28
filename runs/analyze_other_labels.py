"""Characterise the ``other`` bucket of the Nemotron group-labelled mixed dataset.

Why this exists: ``runs/mixed_dataset_group_labeled.csv`` put 1068/1244 rows (86%) into
``other``. This standalone, network-free script answers two questions:

1. WHY is ``other`` so large? It splits ``other`` into rows that are genuine model verdicts vs
   rows that merely *degraded* to ``other`` because the OpenRouter call failed (the
   ``group_reason`` carries ``"classify error: ..."`` — almost all are free-tier 429 rate-limit
   errors). Those error rows were never actually classified, so they tell us nothing about the
   taxonomy.
2. CAN ``other`` be labelled? It keyword-buckets every ``other`` statement into (a) clearly maps
   to one of the 14 ToxiGen groups (i.e. recoverable by a working classifier), (b) maps to an
   EXPANDED group outside ToxiGen (immigrants, socioeconomic, age, appearance, nationality,
   other-religion, political/ideological), or (c) genuinely group-less / universal.

Run:

    .venv/bin/python runs/analyze_other_labels.py
    .venv/bin/python runs/analyze_other_labels.py --input runs/mixed_dataset_group_labeled.csv
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO / "runs/mixed_dataset_group_labeled.csv"

# Keyword -> canonical ToxiGen group. Used only to ESTIMATE how many 'other' statements obviously
# target one of the existing 14 groups (i.e. would be recoverable by a working classifier). Word
# boundaries keep "asian" from firing on "Caucasian" etc.
TOXIGEN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "women": ("women", "woman", "female", "girls", "girl", "feminist", "feminists"),
    "lgbtq+ folks": (
        "lgbt", "lgbtq", "gay", "gays", "homosexual", "homosexuals", "lesbian", "lesbians",
        "queer", "transgender", "trans ", "bisexual",
    ),
    "muslim folks": ("muslim", "muslims", "islam", "islamic"),
    "jewish folks": ("jew", "jews", "jewish", "judaism", "semitic"),
    "black folks / african-americans": (
        "black people", "black folks", "blacks", "african", "africans", "african-american",
    ),
    "latino/hispanic folks": ("latino", "latina", "hispanic", "latinx"),
    "mexican folks": ("mexican", "mexicans"),
    "asian folks": ("asian", "asians"),
    "chinese folks": ("chinese", "china"),
    "native american/indigenous folks": ("native american", "indigenous", "aboriginal"),
    "middle eastern folks": ("middle eastern", "middle-eastern", "arab", "arabs"),
    "folks with physical disabilities": (
        "disabled", "disability", "wheelchair", "blind", "deaf", "physical disabilit",
    ),
    "folks with mental disabilities": (
        "mentally retarded", "mental disabilit", "retarded", "autistic", "down syndrome",
    ),
}

# Keyword -> EXPANDED (out-of-ToxiGen) group. These are the recurring themes the ToxiGen taxonomy
# has no slot for. Order matters only for reporting; a row can match several and is counted once
# per group it matches (multi-label) plus once in the headline "any expanded" set.
EXTENDED_KEYWORDS: dict[str, tuple[str, ...]] = {
    "immigrants/migrants/refugees": (
        "immigrant", "immigrants", "migrant", "migrants", "refugee", "refugees", "foreigner",
        "foreigners", "illegal alien", "deport", "asylum",
    ),
    "socioeconomic (poverty/class)": (
        "poor people", "poverty", "the poor", "rich people", "the rich", "welfare", "homeless",
        "public assistance", "low-income", "lower class", "working class", "wealthy",
    ),
    "age (young/old)": (
        "teenager", "teenagers", "teen ", "teens", "elderly", "old people", "older people",
        "boomer", "boomers", "millennial", "aging", "with age", "young people",
    ),
    "appearance/body": (
        "fat people", "overweight", "obese", "ugly", "scars", "accent", "goth", "goths",
        "tattoo", "bald", "appearance", "dress differently", "clothing",
    ),
    "other religion (christian/hindu/atheist/etc.)": (
        "christian", "christians", "catholic", "hindu", "hindus", "buddhist", "atheist",
        "atheists", "non-religious", "non religious", "religious people", "sikh",
    ),
    "nationality/ethnicity (non-ToxiGen)": (
        "eastern european", "eastern europeans", "russian", "russians", "polish", "poles",
        "german", "germans", "gypsy", "gypsies", "roma ", "romani", "irish", "italian",
    ),
    "political/ideological": (
        "obama", "trump", "republican", "democrat", "liberal", "conservative", "communist",
        "socialist", "politician", "politicians",
    ),
}


def _matches(text: str, keywords: tuple[str, ...]) -> bool:
    low = f" {text.lower()} "
    for kw in keywords:
        if " " in kw or "-" in kw:
            if kw in low:
                return True
        elif re.search(rf"\b{re.escape(kw.strip())}\b", low):
            return True
    return False


def first_group(text: str, table: dict[str, tuple[str, ...]]) -> str | None:
    for group, keywords in table.items():
        if _matches(text, keywords):
            return group
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--statement-col", default="statement")
    ap.add_argument(
        "--dump", default=None, help="Optional CSV path to write the per-row 'other' analysis."
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    total = len(df)
    other = df[df["target_group"] == "other"].copy()
    n_other = len(other)
    print(f"rows total: {total}")
    print(f"rows labelled 'other': {n_other}  ({n_other / total:.0%})\n")

    # --- 1. Genuine 'other' verdict vs degraded-to-'other' API failures -----------------------
    reason = other["group_reason"].fillna("").astype(str)
    is_error = reason.str.contains("classify error", case=False)
    n_error = int(is_error.sum())
    n_429 = int(reason.str.contains("429").sum())
    n_none = int(reason.str.contains("NoneType").sum())
    genuine = other[~is_error].copy()
    n_genuine = len(genuine)
    print("=" * 70)
    print("WHY is 'other' so large?")
    print("=" * 70)
    n_other_err = n_error - n_429
    print(f"  API-failure rows (degraded to 'other', never classified): {n_error}")
    print(f"      - rate-limit 429 (free-tier 16 req/min):              {n_429}")
    print(f"      - other API errors (e.g. NoneType / None response):   {n_other_err}  "
          f"(of which NoneType: {n_none})")
    print(f"  GENUINE model 'other' verdicts:                           {n_genuine}")
    print(
        f"\n  => {n_error}/{n_other} ({n_error / n_other:.0%}) of 'other' is an INFRASTRUCTURE\n"
        f"     artefact (rate limiting), not a taxonomy or model-quality problem.\n"
    )

    # --- 2. Keyword-bucket EVERY 'other' statement (errors included: the text is still real) ---
    # Use plain Python lists (not Series.apply) so a None stays None instead of being coerced to
    # NaN, which would break the `is not None` bucketing below.
    other = other.reset_index(drop=True)
    stmts = other[args.statement_col].astype(str).tolist()
    tox_hit = [first_group(s, TOXIGEN_KEYWORDS) for s in stmts]
    ext_hit = [first_group(s, EXTENDED_KEYWORDS) for s in stmts]

    def bucket(i: int) -> str:
        if tox_hit[i] is not None:
            return "(b) mislabelled: maps to a ToxiGen-14 group"
        if ext_hit[i] is not None:
            return "(a) recoverable via EXPANDED taxonomy"
        return "(c) genuinely group-less / universal"

    buckets = [bucket(i) for i in range(len(other))]
    other["analysis_bucket"] = buckets
    other["toxigen_keyword_group"] = tox_hit
    other["extended_keyword_group"] = ext_hit

    print("=" * 70)
    print("CAN 'other' be labelled? (keyword bucketing of all 'other' statements)")
    print("=" * 70)
    bc = Counter(buckets)
    for label, count in bc.most_common():
        print(f"  {count:5d}  ({count / n_other:.0%})  {label}")

    print("\n  (b) ToxiGen-14 groups that 'other' statements actually mention:")
    for group, count in Counter(g for g in tox_hit if g is not None).most_common():
        print(f"        {count:5d}  {group}")

    print("\n  (a) EXPANDED (out-of-ToxiGen) groups present in 'other':")
    for group, count in Counter(g for g in ext_hit if g is not None).most_common():
        print(f"        {count:5d}  {group}")

    # Model-error rate, measured ONLY on rows the model actually classified (non-error) that are
    # NOT 'other': we cross-check those keyword hits against the assigned label elsewhere; here we
    # report the obvious-miss rate inside 'other' restricted to genuine verdicts.
    gen_stmts = genuine[args.statement_col].astype(str).tolist()
    gen_tox = [first_group(s, TOXIGEN_KEYWORDS) for s in gen_stmts]
    n_gen_mis = sum(g is not None for g in gen_tox)
    print("\n" + "=" * 70)
    print("MODEL-QUALITY signal")
    print("=" * 70)
    print(
        f"  Of the {n_genuine} GENUINE 'other' verdicts, {n_gen_mis} contain an obvious ToxiGen\n"
        f"  group keyword => clear model misclassifications "
        f"({(n_gen_mis / n_genuine if n_genuine else 0):.0%} of genuine 'other')."
    )

    if args.dump:
        cols = [
            args.statement_col, "target_group", "group_reason", "analysis_bucket",
            "toxigen_keyword_group", "extended_keyword_group",
        ]
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        other[cols].to_csv(args.dump, index=False)
        print(f"\nwrote per-row analysis -> {args.dump}")


if __name__ == "__main__":
    main()
