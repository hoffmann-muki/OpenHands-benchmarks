from pathlib import Path

import pandas as pd
import pytest

from benchmarks.utils.dataset import prepare_dataset


def test_prepare_dataset_rejects_missing_selected_instances(tmp_path: Path) -> None:
    select_file = tmp_path / "instances.txt"
    select_file.write_text("known\nmissing\n", encoding="utf-8")
    dataset = pd.DataFrame({"instance_id": ["known", "another"]})

    with pytest.raises(
        ValueError,
        match="Selected instance IDs were not found in the dataset: missing",
    ):
        prepare_dataset(dataset, selected_instances_file=str(select_file))


def test_prepare_dataset_keeps_every_selected_instance(tmp_path: Path) -> None:
    select_file = tmp_path / "instances.txt"
    select_file.write_text("second\nfirst\n", encoding="utf-8")
    dataset = pd.DataFrame(
        {
            "instance_id": ["first", "second", "third"],
            "value": [1, 2, 3],
        }
    )

    selected = prepare_dataset(dataset, selected_instances_file=str(select_file))

    assert set(selected["instance_id"]) == {"first", "second"}


def test_prepare_dataset_ordered_mode_preserves_explicit_order(
    tmp_path: Path,
) -> None:
    select_file = tmp_path / "instances.txt"
    select_file.write_text("third\nfirst\nsecond\n", encoding="utf-8")
    dataset = pd.DataFrame(
        {
            "instance_id": ["first", "second", "third", "fourth"],
            "value": [1, 2, 3, 4],
        }
    )

    selected = prepare_dataset(
        dataset,
        n_limit=2,
        selected_instances_file=str(select_file),
        selection_mode="ordered",
    )

    assert selected["instance_id"].tolist() == ["third", "first", "second"]
    assert selected["value"].tolist() == [3, 1, 2]


def test_prepare_dataset_ordered_mode_takes_dataset_prefix() -> None:
    dataset = pd.DataFrame(
        {"instance_id": ["first", "second", "third"], "value": [1, 2, 3]}
    )

    selected = prepare_dataset(dataset, n_limit=2, selection_mode="ordered")

    assert selected["instance_id"].tolist() == ["first", "second"]


def test_prepare_dataset_rejects_duplicate_selected_instances(tmp_path: Path) -> None:
    select_file = tmp_path / "instances.txt"
    select_file.write_text("first\nsecond\nfirst\n", encoding="utf-8")
    dataset = pd.DataFrame({"instance_id": ["first", "second"]})

    with pytest.raises(ValueError, match="Duplicate instance ID.*line 3: first"):
        prepare_dataset(
            dataset,
            selected_instances_file=str(select_file),
            selection_mode="ordered",
        )


def test_prepare_dataset_filters_with_normalized_instance_ids(tmp_path: Path) -> None:
    select_file = tmp_path / "instances.txt"
    select_file.write_text("1\n", encoding="utf-8")
    dataset = pd.DataFrame({"instance_id": [1, 2]})

    selected = prepare_dataset(dataset, selected_instances_file=str(select_file))

    assert selected["instance_id"].tolist() == [1]
