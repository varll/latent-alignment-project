from pathlib import Path

import pandas as pd

from latent_alignment.data import load_dataset


def test_load_paired_dataset(tmp_path: Path) -> None:
    path = tmp_path / "pairs.csv"
    pd.DataFrame(
        {
            "positive_text": ["a yes", "b yes"],
            "negative_text": ["a no", "b no"],
            "label": [0, 1],
        }
    ).to_csv(path, index=False)

    dataset = load_dataset(path, label_col="label")

    assert dataset.positive_texts == ["a yes Yes.", "b yes Yes.", "a no Yes.", "b no Yes."]
    assert dataset.negative_texts == ["a yes No.", "b yes No.", "a no No.", "b no No."]
    assert dataset.labels.tolist() == [0, 1, 1, 0]
    assert dataset.opposite_indices.tolist() == [2, 3, 0, 1]


def test_load_polarity_raw_dataset(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "statement": ["harm a", "harm b", "safe a", "safe b"],
            "is_harmfull_opposition": [0, 1, 0, 1],
        }
    ).to_csv(path, index=False)

    dataset = load_dataset(path, dataset_format="polarity_raw")

    assert dataset.positive_texts == ["harm a Yes.", "harm b Yes.", "safe a Yes.", "safe b Yes."]
    assert dataset.negative_texts == ["harm a No.", "harm b No.", "safe a No.", "safe b No."]
    assert dataset.labels.tolist() == [0, 1, 0, 1]
    assert dataset.opposite_indices.tolist() == [2, 3, 0, 1]


def test_load_single_dataset(tmp_path: Path) -> None:
    path = tmp_path / "single.csv"
    pd.DataFrame(
        {
            "statement": ["toxic one", "benign two", "toxic three"],
            "is_harmfull_opposition": [1, 0, 1],
        }
    ).to_csv(path, index=False)

    dataset = load_dataset(path, dataset_format="single")

    assert dataset.positive_texts == ["toxic one Yes.", "benign two Yes.", "toxic three Yes."]
    assert dataset.negative_texts == ["toxic one No.", "benign two No.", "toxic three No."]
    assert dataset.labels.tolist() == [1, 0, 1]
    # No opposite pairing: every row points at itself, so PA metrics stay empty.
    assert dataset.opposite_indices.tolist() == [0, 1, 2]
