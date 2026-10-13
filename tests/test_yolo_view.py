import json
import os
import sys

import pytest
import yaml

from conftest import BASELINE, DEFAULT_SPLITS, NAMES, REPO_ROOT, symlinks_supported, tree_digest
from firewatch.data.yolo import YoloBox
from firewatch.data.yolo_view import (
    MANIFEST_NAME,
    SanitizedBox,
    ViewError,
    build_yolo_view,
    load_view_manifest,
    sanitize_box,
    sanitize_label_text,
)

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import prepare_data  # noqa: E402

# --- per-box policy -----------------------------------------------------------


def test_unchanged_box_is_returned_as_is():
    box = YoloBox(0, 0.5, 0.5, 0.2, 0.2)
    assert sanitize_box(box) == SanitizedBox(box, None)


def test_box_wider_than_image_is_clipped():
    result = sanitize_box(YoloBox(0, 0.5, 0.5, 1.05, 0.4))
    assert result.action == "clipped"
    assert result.box.width == pytest.approx(1.0)
    assert result.box.x_center == pytest.approx(0.5)
    assert result.box.height == pytest.approx(0.4)


def test_box_crossing_left_edge_is_clipped():
    result = sanitize_box(YoloBox(1, 0.05, 0.5, 0.2, 0.2))  # x1 = -0.05
    assert result.action == "clipped"
    assert result.box.x_center == pytest.approx(0.075)
    assert result.box.width == pytest.approx(0.15)


@pytest.mark.parametrize(
    "box",
    [
        YoloBox(0, 0.5, 0.5, 0.0, 0.0),
        YoloBox(1, 0.5, 0.5, 0.0, 0.1),
        YoloBox(1, 0.5, 0.5, 0.1, 0.0),
        YoloBox(0, 0.5, 0.5, -0.1, 0.2),
        YoloBox(0, 1.5, 0.5, 0.2, 0.2),  # entirely outside the image
    ],
)
def test_degenerate_boxes_are_dropped(box):
    assert sanitize_box(box) == SanitizedBox(None, "dropped_degenerate")


def test_min_box_size_threshold():
    box = YoloBox(0, 0.5, 0.5, 0.001, 0.2)
    assert sanitize_box(box, min_box_size=0.0).action is None
    assert sanitize_box(box, min_box_size=0.002).action == "dropped_degenerate"


# --- per-file policy ---------------------------------------------------------------


def test_unchanged_lines_are_copied_verbatim():
    line = "0 0.123456789012345 0.5 0.2 0.2"
    lines, changes, n_in = sanitize_label_text(f"{line}\n", 2)
    assert lines == [line] and changes == [] and n_in == 1


def test_changes_are_recorded_with_before_and_after():
    lines, changes, n_in = sanitize_label_text(
        "0 0.5 0.5 1.05 0.4\n1 0.5 0.5 0 0\n", 2, source="a.txt"
    )
    assert n_in == 2 and len(lines) == 1
    assert [c["action"] for c in changes] == ["clipped", "dropped_degenerate"]
    assert changes[0]["file"] == "a.txt" and changes[0]["line"] == 1
    assert changes[0]["after"].startswith("0 0.50000000 0.50000000 1.00000000")
    assert "after" not in changes[1]


@pytest.mark.parametrize("text", ["0 0.5 0.5\n", "2 0.5 0.5 0.1 0.1\n", "0 nan 0.5 0.1 0.1\n"])
def test_malformed_or_unknown_class_aborts(text):
    with pytest.raises(ViewError):
        sanitize_label_text(text, 2)


# --- building a view ----------------------------------------------------------------


def test_view_layout_and_data_yaml(make_view):
    raw, view = make_view()
    data = yaml.safe_load((view / "data.yaml").read_text())
    assert data["path"] == str(view.absolute())
    assert data["train"] == "train/images" and data["val"] == "val/images"
    assert data["test"] == "test/images"
    assert data["names"] == {0: "smoke", 1: "fire"} and data["nc"] == 2
    for split in ("train", "val", "test"):
        assert (view / split / "images").is_dir() and (view / split / "labels").is_dir()


def test_view_applies_policy(make_view):
    _, view = make_view()
    manifest = load_view_manifest(view)
    train = manifest["splits"]["train"]
    assert train["images_in"] == 6
    assert train["images_out"] == 5  # t_alldrop excluded
    assert train["excluded"] == {"excluded_all_boxes_dropped": ["t_alldrop"]}
    assert train["background"] == 1
    assert train["num_clipped"] == 1 and train["num_dropped_degenerate"] == 2
    labels = view / "train" / "labels"
    assert (labels / "t_bg.txt").read_text() == ""
    assert (labels / "t_mixed.txt").read_text() == "0 0.4 0.4 0.2 0.2\n"
    assert (labels / "t_oob.txt").read_text().startswith("0 0.50000000 0.50000000 1.00000000")
    assert not (labels / "t_alldrop.txt").exists()
    assert not (view / "train" / "images" / "t_alldrop.jpg").exists()
    assert manifest["class_names"] == NAMES


def test_source_dataset_is_never_modified(make_dataset, tmp_path):
    raw = make_dataset(DEFAULT_SPLITS)
    before = tree_digest(raw)
    build_yolo_view(raw, tmp_path / "view", {s: s for s in DEFAULT_SPLITS}, NAMES, link_mode="copy")
    assert tree_digest(raw) == before


def test_missing_label_and_duplicate_stem_are_excluded(make_dataset, tmp_path):
    raw = make_dataset({"train": {"a": "0 0.5 0.5 0.1 0.1\n", "nolabel": None}, "val": {"b": ""}})
    (raw / "train" / "images" / "a.png").write_bytes(b"dup")
    manifest = build_yolo_view(
        raw, tmp_path / "view", {"train": "train", "val": "val"}, NAMES, link_mode="copy"
    )
    excluded = manifest["splits"]["train"]["excluded"]
    assert excluded == {"excluded_duplicate_stem": ["a"], "excluded_missing_label": ["nolabel"]}


def test_failure_leaves_no_partial_view(make_dataset, tmp_path):
    raw = make_dataset({"train": {"bad": "0 0.5\n"}, "val": {"b": ""}})
    view = tmp_path / "view"
    with pytest.raises(ViewError, match="bad.txt:1"):
        build_yolo_view(raw, view, {"train": "train", "val": "val"}, NAMES, link_mode="copy")
    assert not view.exists()
    assert not [p for p in tmp_path.iterdir() if ".partial-" in p.name]


def test_existing_view_requires_overwrite(make_view):
    raw, view = make_view()
    splits = {s: s for s in DEFAULT_SPLITS}
    with pytest.raises(ViewError, match="--overwrite"):
        build_yolo_view(raw, view, splits, NAMES, link_mode="copy")
    build_yolo_view(raw, view, splits, NAMES, link_mode="copy", overwrite=True)
    assert (view / MANIFEST_NAME).is_file()


def test_refuses_foreign_non_empty_directory(make_dataset, tmp_path):
    raw = make_dataset(DEFAULT_SPLITS)
    target = tmp_path / "precious"
    target.mkdir()
    (target / "notes.txt").write_text("keep me")
    with pytest.raises(ViewError, match="not a FireWatch dataset view"):
        build_yolo_view(raw, target, {"train": "train", "val": "val"}, NAMES, overwrite=True)
    assert (target / "notes.txt").read_text() == "keep me"


def test_refuses_view_inside_source(make_dataset):
    raw = make_dataset(DEFAULT_SPLITS)
    with pytest.raises(ViewError, match="overlap"):
        build_yolo_view(raw, raw / "view", {"train": "train", "val": "val"}, NAMES)


def test_requires_train_and_val(make_dataset, tmp_path):
    raw = make_dataset(DEFAULT_SPLITS)
    with pytest.raises(ViewError, match="train"):
        build_yolo_view(raw, tmp_path / "v", {"train": "train"}, NAMES)


def test_symlink_mode(make_dataset, tmp_path):
    if not symlinks_supported(tmp_path):
        pytest.skip("symlinks not permitted on this system")
    raw = make_dataset(DEFAULT_SPLITS)
    view = tmp_path / "view"
    build_yolo_view(raw, view, {s: s for s in DEFAULT_SPLITS}, NAMES, link_mode="symlink")
    img = view / "train" / "images" / "t_smoke.jpg"
    assert img.is_symlink()
    assert img.resolve() == (raw / "train" / "images" / "t_smoke.jpg").resolve()
    assert not (view / "train" / "labels" / "t_smoke.txt").is_symlink()


def test_hardlink_mode(make_dataset, tmp_path):
    raw = make_dataset(DEFAULT_SPLITS)
    view = tmp_path / "view"
    build_yolo_view(raw, view, {s: s for s in DEFAULT_SPLITS}, NAMES, link_mode="hardlink")
    img = view / "train" / "images" / "t_smoke.jpg"
    assert os.path.samefile(img, raw / "train" / "images" / "t_smoke.jpg")


def test_load_manifest_errors(tmp_path):
    with pytest.raises(ViewError, match="not a FireWatch dataset view"):
        load_view_manifest(tmp_path)
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"format_version": 99}))
    with pytest.raises(ViewError, match="unsupported"):
        load_view_manifest(tmp_path)


# --- prepare_data.py --output-dir ---------------------------------------------------


def test_cli_builds_view(make_dataset, tmp_path):
    raw = make_dataset(DEFAULT_SPLITS)
    view = tmp_path / "view"
    rc = prepare_data.main(
        [
            "--config",
            str(BASELINE),
            "--data-root",
            str(raw),
            "--output-dir",
            str(view),
            "--link-mode",
            "copy",
        ]
    )
    assert rc == 0
    assert load_view_manifest(view)["splits"]["train"]["images_out"] == 5


def test_cli_does_not_build_view_when_validation_fails(make_dataset, tmp_path):
    raw = make_dataset({"train": {"a": "7 0.5 0.5 0.1 0.1\n"}, "val": {"b": ""}})
    view = tmp_path / "view"
    rc = prepare_data.main(
        ["--config", str(BASELINE), "--data-root", str(raw), "--output-dir", str(view)]
    )
    assert rc == 1
    assert not view.exists()


def test_cli_min_box_size_override(make_dataset, tmp_path):
    raw = make_dataset({"train": {"a": "0 0.5 0.5 0.001 0.2\n"}, "val": {"b": ""}})
    view = tmp_path / "view"
    args = ["--config", str(BASELINE), "--data-root", str(raw), "--link-mode", "copy"]
    assert prepare_data.main([*args, "--output-dir", str(view), "--min-box-size", "0.01"]) == 0
    train = load_view_manifest(view)["splits"]["train"]
    assert train["excluded"] == {"excluded_all_boxes_dropped": ["a"]}
