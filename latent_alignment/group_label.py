"""Target-group classification of statements via OpenRouter (NVIDIA Nemotron Ultra).

Each statement in the mixed-polarity probing set is assigned to exactly one ToxiGen
**target group** (the group the statement is *about* / *targets*), or to ``"other"`` when it
fits no group. This mirrors the OpenRouter pattern used by :mod:`latent_alignment.judge` and
:mod:`latent_alignment.toxigen`: an OpenAI-compatible client pointed at OpenRouter, imported
lazily so this module needs no SDK and no API key to import; threaded fan-out for the I/O-bound
calls; graceful degradation on failure; and a pure :func:`parse_group_label` so label parsing is
unit-testable with string fixtures.

The default taxonomy below is the set of unique values from the ``target_group`` column of
``data/toxigen/raw/toxigen_annotated_test_paired.csv`` (so it matches ToxiGen exactly), plus an
``"other"`` bucket. Override it via the ``categories`` argument if your data uses a different set.
"""

from __future__ import annotations

import json
import os
import re

# Derived verbatim from the unique values of the ``target_group`` column in
# ``data/toxigen/raw/toxigen_annotated_test_paired.csv`` (read on 2026-06-28). Kept exactly as
# they appear in the file so the labels line up with ToxiGen's own groups. The two "black /
# african-american" spellings are both present in that file and are preserved as-is.
TOXIGEN_TARGET_GROUPS: tuple[str, ...] = (
    "asian folks",
    "black folks / african-americans",
    "black/african-american folks",
    "chinese folks",
    "folks with mental disabilities",
    "folks with physical disabilities",
    "jewish folks",
    "latino/hispanic folks",
    "lgbtq+ folks",
    "mexican folks",
    "middle eastern folks",
    "muslim folks",
    "native american/indigenous folks",
    "women",
)

# The catch-all bucket for statements that do not target any of the ToxiGen groups above.
OTHER_CATEGORY = "other"

# Default taxonomy handed to the classifier: the ToxiGen groups plus "other".
DEFAULT_CATEGORIES: tuple[str, ...] = (*TOXIGEN_TARGET_GROUPS, OTHER_CATEGORY)

# Extra target groups that recur in ``data/polarity_probing/raw/mixed_dataset.csv`` but have NO
# slot in the ToxiGen-14 taxonomy. Derived from a keyword/theme audit of the 1068 "other" rows
# in ``runs/mixed_dataset_group_labeled.csv`` (see ``runs/analyze_other_labels.py``): the mixed
# set is heavy on immigration discourse (~75+ statements) plus socioeconomic, age, appearance,
# non-Abrahamic/secular religion, non-ToxiGen nationalities, and political/ideological targets.
# These names are written to read clearly to the LLM as standalone category labels.
EXTENDED_EXTRA_GROUPS: tuple[str, ...] = (
    "immigrants/migrants/refugees",
    "people of low/high socioeconomic status",
    "people targeted by age (young or old)",
    "people targeted by appearance or body",
    "other religious groups (christian, hindu, atheist, etc.)",
    "national or ethnic group not listed above",
    "political or ideological group",
)

# Optional richer taxonomy: ToxiGen-14 + the recurring extra groups + "other". Pass this (or any
# custom list) via the ``categories`` argument / ``--extended`` CLI flag. The plain
# ``DEFAULT_CATEGORIES`` remains the default so existing behaviour is unchanged.
EXTENDED_CATEGORIES: tuple[str, ...] = (
    *TOXIGEN_TARGET_GROUPS,
    *EXTENDED_EXTRA_GROUPS,
    OTHER_CATEGORY,
)

# NVIDIA Nemotron Ultra on OpenRouter. This is the slug confirmed working for this project's
# ToxiGen benign-rewrite generation; the ``...llama-3.1-nemotron-ultra-253b-v1`` slug 404s
# ("No endpoints found"). Override with ``--model`` / ``model=`` if it ever changes.
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def build_group_messages(statement: str, categories: list[str] | tuple[str, ...]) -> list[dict]:
    """Build the chat messages that ask the model to assign ``statement`` to one category."""
    options = "\n".join(f"- {c}" for c in categories)
    system = (
        "You are a careful annotator for a hate-speech research dataset. You are given a "
        "STATEMENT and a fixed list of CATEGORIES describing demographic target groups. Decide "
        "which group the statement is ABOUT or TARGETS, regardless of whether the statement is "
        "positive, negative, hateful, or benign. Pick exactly ONE category from the list. If the "
        f'statement does not target any listed group, choose "{OTHER_CATEGORY}".\n\n'
        f"CATEGORIES:\n{options}\n\n"
        'Respond with ONLY a JSON object: {"group": "<one category exactly as written above>", '
        '"reason": "<one short sentence>"}.'
    )
    user = f"STATEMENT:\n{str(statement).strip()}\n\nClassify the STATEMENT."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_group_label(text: str, categories: list[str] | tuple[str, ...]) -> dict:
    """Parse a classifier response into ``{"group", "reason", "raw"}`` (pure, unit-testable).

    Prefers the JSON object the model is asked to emit (``{"group": ..., "reason": ...}``) and
    matches its ``group`` against ``categories`` case-insensitively, returning the canonical
    spelling from ``categories``. If the JSON is missing/malformed or names an unknown group,
    falls back to scanning the raw text for the first category name that appears as a substring
    (longest names first, so a specific group wins over a shorter overlapping one). Anything that
    still cannot be matched -> ``"other"``.
    """
    raw = (text or "").strip()
    # Map lowercase -> canonical spelling for case-insensitive exact matching.
    canon = {str(c).strip().lower(): str(c) for c in categories}
    group: str | None = None
    reason = ""

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
        except (ValueError, TypeError):
            data = None
        if isinstance(data, dict):
            candidate = str(data.get("group", "")).strip().lower()
            if candidate in canon:
                group = canon[candidate]
            reason = str(data.get("reason", "")).strip()

    if group is None and raw:
        # Keyword fallback: look for category names inside the raw text. A reasoning model often
        # restates the whole option list, so we must NOT just take the first/longest match — that
        # deterministically anchors on one long name (this is exactly how the first full run
        # mislabelled 89 unrelated statements as "folks with physical disabilities"). Instead,
        # only trust the fallback when EXACTLY ONE distinct category is mentioned; otherwise the
        # signal is ambiguous and we defer to "other".
        lowered = raw.lower()
        mentioned = {
            canon[key] for key in canon if key and key != OTHER_CATEGORY and key in lowered
        }
        if len(mentioned) == 1:
            group = next(iter(mentioned))

    if group is None:
        group = canon.get(OTHER_CATEGORY, OTHER_CATEGORY)

    return {"group": group, "reason": reason, "raw": raw}


def _make_openrouter_client(api_key: str | None = None):
    """Build an OpenAI-compatible client pointed at OpenRouter (lazy import)."""
    from openai import OpenAI

    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("no OpenRouter API key; set OPENROUTER_API_KEY or pass api_key=...")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)


def classify_group(
    statement: str,
    *,
    categories: list[str] | tuple[str, ...] = DEFAULT_CATEGORIES,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 200,
    client=None,
) -> dict:
    """Classify one ``statement`` into a single target group via OpenRouter.

    Returns ``{"group", "reason", "raw"}`` as produced by :func:`parse_group_label`. Set
    ``OPENROUTER_API_KEY`` (or pass ``api_key`` / a pre-built ``client``). A failed request
    degrades to ``"other"`` with the error text in ``reason`` so a single bad row never aborts a
    batch.
    """
    if client is None:
        client = _make_openrouter_client(api_key)
    messages = build_group_messages(statement, categories)
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return parse_group_label(completion.choices[0].message.content or "", categories)
    except Exception as exc:  # network/API errors -> mark unclassifiable, keep going.
        return {"group": OTHER_CATEGORY, "reason": f"classify error: {exc}", "raw": ""}


def classify_groups(
    statements: list[str],
    *,
    categories: list[str] | tuple[str, ...] = DEFAULT_CATEGORIES,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 200,
    client=None,
    max_workers: int = 8,
) -> list[dict]:
    """Classify each statement into one target group, returning one dict per input in order.

    Mirrors :func:`latent_alignment.judge.judge_generations`: lazy OpenRouter client, threaded
    fan-out across ``max_workers`` (the calls are I/O-bound; set ``max_workers=1`` to run
    serially), and per-row failures degrade to ``"other"`` rather than aborting the batch.
    """
    if client is None:
        client = _make_openrouter_client(api_key)

    def classify_one(statement: str) -> dict:
        return classify_group(
            statement,
            categories=categories,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            client=client,
        )

    if max_workers <= 1 or len(statements) <= 1:
        return [classify_one(s) for s in statements]

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # executor.map preserves input order, so results line up with the inputs.
        return list(pool.map(classify_one, statements))
