import json
import sys

import pytest
import yaml

from conftest import REPO_ROOT
from firewatch.utils.config import ConfigError, load_config, resolve_data_root

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import prepare_data  # noqa: E402

BASELINE = REPO_ROOT / "configs" / "baseline.yaml"


def test_baseline_config_is_valid():
    cfg = load_config(BASELINE)
    assert cfg["dataset"]["class_names"] == ["smoke", "fire"]


def test_load_config_rejects_missing_classes(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("dataset:\n  splits: {train: train}\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_resolve_data_root_precedence(monkeypatch, tmp_path):
    cfg = {"dataset": {"root": None}}
    monkeypatch.delenv("FIREWATCH_DATA_ROOT", raising=False)
    with pytest.raises(ConfigError):
        resolve_data_root(cfg)
    monkeypatch.setenv("FIREWATCH_DATA_ROOT", str(tmp_path / "env"))
    assert resolve_data_root(cfg) == tmp_path / "env"
    cfg["dataset"]["root"] = str(tmp_path / "cfg")
    assert resolve_data_root(cfg) == tmp_path / "cfg"
    assert resolve_data_root(cfg, str(tmp_path / "cli")) == tmp_path / "cli"


def test_cli_end_to_end(make_dataset, tmp_path, capsys):
    root = make_dataset(
        {"train": {"a": "0 0.5 0.5 0.2 0.2\n"}, "val": {"b": "1 0.5 0.5 0.2 0.2\n"}}
    )
    report = tmp_path / "out" / "report.json"
    data_yaml = tmp_path / "out" / "data.yaml"
    rc = prepare_data.main(
        [
            "--config",
            str(BASELINE),
            "--data-root",
            str(root),
            "--report",
            str(report),
            "--write-data-yaml",
            str(data_yaml),
        ]
    )
    assert rc == 0
    payload = json.loads(report.read_text())
    assert payload["missing_splits"] == ["test"]
    written = yaml.safe_load(data_yaml.read_text())
    assert set(written) == {"path", "train", "val", "nc", "names"}


def test_cli_fails_on_errors(make_dataset):
    root = make_dataset({"train": {"a": "9 0.5 0.5 0.2 0.2\n"}})
    assert prepare_data.main(["--config", str(BASELINE), "--data-root", str(root)]) == 1


def test_cli_strict_fails_on_warnings(make_dataset):
    root = make_dataset({"train": {"a": "0 0.5 0.5 0.2 0.2\n", "nolabel": None}})
    args = ["--config", str(BASELINE), "--data-root", str(root)]
    assert prepare_data.main(args) == 0
    assert prepare_data.main([*args, "--strict"]) == 1


def test_cli_missing_root(tmp_path):
    rc = prepare_data.main(["--config", str(BASELINE), "--data-root", str(tmp_path / "nope")])
    assert rc == 2


@pytest.mark.parametrize(
    "body, message",
    [
        ("dataset: [unclosed\n", "invalid YAML"),
        ("- a\n- b\n", "top level must be a mapping"),
        ("other: 1\n", "missing 'dataset'"),
        ("dataset:\n  class_names: [smoke]\n", "dataset.splits"),
        ("dataset:\n  class_names: [smoke]\n  splits: {train: null}\n", "splits.train"),
        (
            "dataset:\n  class_names: [smoke]\n  splits: {train: t}\n  image_extensions: .jpg\n",
            "image_extensions",
        ),
    ],
)
def test_load_config_error_messages(tmp_path, body, message):
    p = tmp_path / "c.yaml"
    p.write_text(body)
    with pytest.raises(ConfigError, match=message):
        load_config(p)


def test_cli_bad_config_exit_2(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("dataset: [unclosed\n")
    assert prepare_data.main(["--config", str(p), "--data-root", str(tmp_path)]) == 2


def test_cli_missing_train_split_exit_2(make_dataset):
    root = make_dataset({"val": {"a": "0 0.5 0.5 0.1 0.1\n"}})
    assert prepare_data.main(["--config", str(BASELINE), "--data-root", str(root)]) == 2


def test_cli_empty_train_exit_1(make_dataset):
    root = make_dataset({"train": {}})
    assert prepare_data.main(["--config", str(BASELINE), "--data-root", str(root)]) == 1


def test_cli_report_written_on_errors(make_dataset, tmp_path):
    root = make_dataset({"train": {"a": "7 0.5 0.5 0.1 0.1\n", "nolabel": None}})
    report = tmp_path / "r.json"
    args = ["--config", str(BASELINE), "--data-root", str(root), "--report", str(report)]
    assert prepare_data.main(args) == 1
    payload = json.loads(report.read_text())
    assert payload["exit_code"] == 1
    assert payload["errors"] == 1 and payload["warnings"] == 1
    split = payload["splits"][0]
    assert split["unknown_class_boxes"] == 1
    assert split["boxes_per_class"] == {"unknown_7": 1}
