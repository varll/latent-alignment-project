import pandas as pd

from latent_alignment.toxigen import (
    assemble_paired_dataset,
    binarize_toxicity,
    build_negation_messages,
    build_prompt,
    parse_negation,
    toxigen_frame_to_dataset,
)


def test_build_prompt_matches_requested_template() -> None:
    prompt = build_prompt("asian", "some toxic text")
    assert prompt == (
        "Is the following text is toxic for the asian.\n"
        "some toxic text\n"
        "The answer is:"
    )
    # The open answer slot becomes "...The answer is: Yes." once CCS appends its " Yes." suffix.
    assert (prompt + " Yes.").endswith("The answer is: Yes.")


def test_binarize_toxicity_threshold() -> None:
    assert binarize_toxicity(5.0) == 1
    assert binarize_toxicity(3.0) == 1
    assert binarize_toxicity(2.999) == 0
    assert binarize_toxicity(1.0) == 0
    assert binarize_toxicity(2.5, threshold=2.5) == 1


def test_toxigen_frame_to_dataset() -> None:
    frame = pd.DataFrame(
        {
            "text": ["t1", "t2"],
            "target_group": ["muslim", "women"],
            "toxicity_human": [4.5, 1.0],
            "extra": ["ignored", "ignored"],
        }
    )

    out = toxigen_frame_to_dataset(frame, toxic_threshold=3.0)

    assert out["statement"].tolist() == [
        "Is the following text is toxic for the muslim.\nt1\nThe answer is:",
        "Is the following text is toxic for the women.\nt2\nThe answer is:",
    ]
    assert out["is_harmfull_opposition"].tolist() == [1, 0]
    assert out["target_group"].tolist() == ["muslim", "women"]


def test_toxigen_frame_to_dataset_falls_back_to_generation_and_group() -> None:
    frame = pd.DataFrame(
        {
            "generation": ["g1"],
            "group": ["latino"],
            "toxicity_human": [3.2],
        }
    )

    out = toxigen_frame_to_dataset(frame)

    assert out["statement"].tolist() == [
        "Is the following text is toxic for the latino.\ng1\nThe answer is:"
    ]
    assert out["is_harmfull_opposition"].tolist() == [1]


def test_build_negation_messages_includes_group_text_and_constraints() -> None:
    messages = build_negation_messages("muslim", "  some toxic text  ")
    assert messages[0]["role"] == "system"
    assert "UNDER 25 words" in messages[0]["content"]
    assert "NO new facts" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "TARGET GROUP: muslim" in messages[1]["content"]
    assert "some toxic text" in messages[1]["content"]


def test_parse_negation_prefers_json_then_falls_back() -> None:
    assert parse_negation('{"rewrite": "Muslims deserve to live."}') == "Muslims deserve to live."
    # Extra prose around the JSON is tolerated.
    assert parse_negation('Sure!\n{"rewrite": "X are fine."}\n') == "X are fine."
    # No JSON -> first non-empty line, quotes stripped.
    assert parse_negation('"just a quoted line"') == "just a quoted line"
    assert parse_negation("") == ""


def test_assemble_paired_dataset_builds_pairs_and_drops_degenerate() -> None:
    groups = ["asian", "women", "muslim"]
    toxic = ["asians are X", "women cannot Y", "muslims Z"]
    rewrites = [
        "Asians are not X.",  # good pair
        "",  # generation failed -> dropped
        "muslims z",  # unchanged (case-insensitive) -> dropped
    ]

    out = assemble_paired_dataset(groups, toxic, rewrites)

    assert list(out.columns) == [
        "positive_text",
        "negative_text",
        "is_harmfull_opposition",
        "target_group",
        "toxic_text",
        "benign_rewrite",
    ]
    assert len(out) == 1
    row = out.iloc[0]
    assert row["positive_text"] == build_prompt("asian", "asians are X")
    assert row["negative_text"] == build_prompt("asian", "Asians are not X.")
    assert row["is_harmfull_opposition"] == 1
