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


def test_prepare_dataset_filters_with_normalized_instance_ids(tmp_path: Path) -> None:
    select_file = tmp_path / "instances.txt"
    select_file.write_text("1\n", encoding="utf-8")
    dataset = pd.DataFrame({"instance_id": [1, 2]})

    selected = prepare_dataset(dataset, selected_instances_file=str(select_file))

    assert selected["instance_id"].tolist() == [1]
