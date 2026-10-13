import pytest

from firewatch.data.dfire import scan_split, ultralytics_data_yaml

NAMES = ["smoke", "fire"]


def test_scan_split_counts(make_dataset):
    root = make_dataset(
        {
            "train": {
                "a": "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n",
                "b": "1 0.5 0.5 0.2 0.2\n",
                "bg": "",
                "nolabel": None,
                "orphan1": "0 0.5 0.5 0.2 0.2\n",
            }
        }
    )
    r = scan_split(root, "train", "train", NAMES)
    assert r.num_images == 4
    assert r.num_label_files == 4
    assert r.num_background_images == 1
    assert r.boxes_per_class == {"smoke": 1, "fire": 2}
    assert r.images_per_class == {"smoke": 1, "fire": 2}
    assert len(r.images_missing_label) == 1
    assert len(r.labels_missing_image) == 1
    assert r.num_errors == 0


def test_scan_split_flags_errors(make_dataset):
    root = make_dataset(
        {"val": {"bad": "0 0.5\n", "unk": "5 0.5 0.5 0.1 0.1\n", "big": "0 1.5 0.5 0.1 0.1\n"}}
    )
    r = scan_split(root, "val", "val", NAMES)
    assert len(r.parse_errors) == 1
    assert r.unknown_class_boxes == 1
    assert r.non_normalised_boxes == 1
    assert r.num_errors == 2
    assert "unknown_5" in r.boxes_per_class


def test_scan_split_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_split(tmp_path, "train", "train", NAMES)


def test_to_dict_truncates_examples(make_dataset):
    root = make_dataset({"train": {f"img{i}": None for i in range(30)}})
    d = scan_split(root, "train", "train", NAMES).to_dict()
    assert d["num_images_missing_label"] == 30
    assert len(d["images_missing_label"]) == 20


def test_ultralytics_data_yaml(tmp_path):
    d = ultralytics_data_yaml(tmp_path, {"train": "train", "val": "valid"}, NAMES)
    assert d["train"] == "train/images"
    assert d["val"] == "valid/images"
    assert d["nc"] == 2 and d["names"] == NAMES


def test_class_id_mapping_matches_dfire(make_dataset):
    # D-Fire: 0 = smoke, 1 = fire.
    root = make_dataset({"train": {"s": "0 0.5 0.5 0.1 0.1\n", "f": "1 0.5 0.5 0.1 0.1\n"}})
    r = scan_split(root, "train", "train", NAMES)
    assert r.boxes_per_class == {"smoke": 1, "fire": 1}


def test_empty_split_is_an_error(make_dataset):
    r = scan_split(make_dataset({"train": {}}), "train", "train", NAMES)
    assert r.num_images == 0
    assert r.num_errors == 1


def test_binary_label_reported_not_raised(make_dataset):
    root = make_dataset({"train": {"a": "placeholder"}})
    (root / "train" / "labels" / "a.txt").write_bytes(b"\xff\xfe\x00")
    r = scan_split(root, "train", "train", NAMES)
    assert len(r.parse_errors) == 1 and r.num_errors == 1


def test_duplicate_image_stems_flagged(make_dataset):
    root = make_dataset({"train": {"a": "0 0.5 0.5 0.1 0.1\n"}})
    (root / "train" / "images" / "a.png").write_bytes(b"")
    r = scan_split(root, "train", "train", NAMES)
    assert r.num_images == 1
    assert r.duplicate_image_stems == ["a"]
    assert r.num_warnings == 1 and r.num_errors == 0
