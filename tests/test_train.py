import json
import os
import sys
from pathlib import Path

import pytest
import yaml

from conftest import BASELINE, NAMES, REPO_ROOT, SMOKE_TEST, symlinks_supported
from firewatch.training import runner
from firewatch.training.options import resolve_train_options
from firewatch.training.preflight import RUN_METADATA_NAME, PreflightError, check_data_yaml
from firewatch.utils.config import load_config

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import train  # noqa: E402


@pytest.fixture
def base_args(make_view, fake_weights, runs_root):
    _, view = make_view()
    return [
        "--config",
        str(BASELINE),
        "--data",
        str(view / "data.yaml"),
        "--weights",
        str(fake_weights),
        "--project",
        str(runs_root),
        "--name",
        "exp1",
    ]


def run(args, capsys):
    rc = train.main(args)
    out, err = capsys.readouterr()
    return rc, out, err


# --- dry run --------------------------------------------------------------------


def test_dry_run_prints_plan_and_writes_nothing(base_args, runs_root, capsys):
    rc, out, err = run([*base_args, "--dry-run"], capsys)
    assert rc == 0, err
    plan = json.loads(out)
    kwargs = plan["ultralytics_train_kwargs"]
    assert kwargs["project"] == str(runs_root) and kwargs["name"] == "exp1"
    assert kwargs["seed"] == 42 and kwargs["imgsz"] == 640 and kwargs["batch"] == 32
    assert kwargs["exist_ok"] is True and kwargs["deterministic"] is True
    assert "split" not in kwargs  # training validates on the data YAML's val split only
    assert not runs_root.exists()


def test_cli_overrides_config(base_args, capsys):
    args = [
        *base_args,
        "--epochs",
        "3",
        "--imgsz",
        "320",
        "--batch",
        "8",
        "--seed",
        "7",
        "--workers",
        "2",
        "--no-amp",
        "--fraction",
        "0.5",
        "--dry-run",
    ]
    rc, out, _ = run(args, capsys)
    assert rc == 0
    kw = json.loads(out)["ultralytics_train_kwargs"]
    assert (kw["epochs"], kw["imgsz"], kw["batch"], kw["seed"]) == (3, 320, 8, 7)
    assert (kw["workers"], kw["amp"], kw["fraction"]) == (2, False, 0.5)


def test_smoke_test_config_extends_baseline(make_view, fake_weights, runs_root, capsys):
    _, view = make_view()
    args = [
        "--config",
        str(SMOKE_TEST),
        "--data",
        str(view / "data.yaml"),
        "--weights",
        str(fake_weights),
        "--project",
        str(runs_root),
        "--dry-run",
    ]
    rc, out, _ = run(args, capsys)
    assert rc == 0
    plan = json.loads(out)
    kw = plan["ultralytics_train_kwargs"]
    assert (kw["epochs"], kw["fraction"], kw["patience"]) == (2, 0.1, 0)
    assert kw["batch"] == 32 and kw["seed"] == 42  # inherited
    assert kw["name"] == "smoke_test"  # default name = config stem


def test_data_yaml_from_environment(make_view, fake_weights, runs_root, monkeypatch, capsys):
    _, view = make_view()
    monkeypatch.setenv("FIREWATCH_DATA_YAML", str(view / "data.yaml"))
    monkeypatch.setenv("FIREWATCH_RUNS_ROOT", str(runs_root))
    rc, out, _ = run(
        ["--config", str(BASELINE), "--weights", str(fake_weights), "--dry-run"], capsys
    )
    assert rc == 0
    assert json.loads(out)["run_dir"] == str(runs_root / "baseline")


# --- argument validation ------------------------------------------------------------


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--imgsz", "650"], "multiple of 32"),
        (["--imgsz", "0"], "imgsz must be >= 32"),
        (["--epochs", "0"], "epochs must be >= 1"),
        (["--batch", "0"], "batch must be >= 1"),
        (["--seed", "-1"], "seed must be >= 0"),
        (["--workers", "-2"], "workers must be >= 0"),
        (["--fraction", "1.5"], "fraction"),
        (["--fraction", "0"], "fraction"),
        (["--device", "0,1"], "multi-GPU"),
        (["--device", "gpu"], "single GPU index"),
        (["--name", "../escape"], "name must match"),
        (["--name", "a b"], "name must match"),
    ],
)
def test_invalid_options_exit_2(base_args, capsys, extra, message):
    rc, _, err = run([*base_args, *extra, "--dry-run"], capsys)
    assert rc == 2
    assert message in err


def test_non_integer_argument_is_rejected_by_argparse(base_args, capsys):
    with pytest.raises(SystemExit) as exc:
        train.main([*base_args, "--epochs", "ten"])
    assert exc.value.code == 2
    assert "invalid int value" in capsys.readouterr().err


def test_help_works(capsys):
    with pytest.raises(SystemExit) as exc:
        train.main(["--help"])
    assert exc.value.code == 0
    assert "--resume" in capsys.readouterr().out


def test_bad_config_value_type(tmp_path, base_args, capsys):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"extends: {BASELINE.as_posix()}\ntrain:\n  epochs: '10'\n")
    args = [*base_args, "--dry-run"]
    args[args.index(str(BASELINE))] = str(cfg)
    rc, _, err = run(args, capsys)
    assert rc == 2 and "epochs must be an integer" in err


# --- missing prerequisites ---------------------------------------------------------------


def test_missing_weights_fails_with_download_hint(base_args, tmp_path, capsys):
    args = [*base_args, "--dry-run"]
    args[args.index("--weights") + 1] = str(tmp_path / "nope" / "yolo26n.pt")
    rc, _, err = run(args, capsys)
    assert rc == 2
    assert "weights file not found" in err
    assert "releases/download/v8.4.0/yolo26n.pt" in err  # weights_url from the config
    assert "never downloads" in err


def test_weights_must_be_pt(base_args, tmp_path, capsys):
    other = tmp_path / "model.onnx"
    other.write_bytes(b"x")
    args = [*base_args, "--dry-run"]
    args[args.index("--weights") + 1] = str(other)
    rc, _, err = run(args, capsys)
    assert rc == 2 and ".pt" in err


def test_no_data_yaml_anywhere(fake_weights, runs_root, capsys):
    args = [
        "--config",
        str(BASELINE),
        "--weights",
        str(fake_weights),
        "--project",
        str(runs_root),
        "--dry-run",
    ]
    rc, _, err = run(args, capsys)
    assert rc == 2 and "no dataset YAML" in err


def test_missing_data_yaml_file(base_args, tmp_path, capsys):
    args = [*base_args, "--dry-run"]
    args[args.index("--data") + 1] = str(tmp_path / "missing.yaml")
    rc, _, err = run(args, capsys)
    assert rc == 2 and "dataset YAML not found" in err


# --- output paths ----------------------------------------------------------------------


def test_project_inside_repository_is_refused(base_args, capsys):
    args = [*base_args, "--dry-run"]
    args[args.index("--project") + 1] = str(REPO_ROOT / "outputs")
    rc, _, err = run(args, capsys)
    assert rc == 2 and "outside the Git repository" in err


@pytest.mark.parametrize("marker", [RUN_METADATA_NAME, "weights", "args.yaml", "results.csv"])
def test_existing_run_is_not_overwritten(base_args, runs_root, capsys, marker):
    run_dir = runs_root / "exp1"
    run_dir.mkdir(parents=True)
    (run_dir / marker).mkdir() if marker == "weights" else (run_dir / marker).write_text("x")
    rc, _, err = run([*base_args, "--dry-run"], capsys)
    assert rc == 2 and "--resume" in err


def test_existing_empty_run_dir_is_allowed(base_args, runs_root, capsys):
    (runs_root / "exp1").mkdir(parents=True)  # e.g. created by a SLURM wrapper
    rc, _, _ = run([*base_args, "--dry-run"], capsys)
    assert rc == 0


# --- resume ----------------------------------------------------------------------------


@pytest.fixture
def previous_run(runs_root):
    run_dir = runs_root / "exp1"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "last.pt").write_bytes(b"ckpt")
    (run_dir / "args.yaml").write_text("imgsz: 640\n")
    return run_dir


def test_resume_dry_run(previous_run, capsys):
    rc, out, err = run(
        [
            "--config",
            str(BASELINE),
            "--resume",
            str(previous_run / "weights" / "last.pt"),
            "--dry-run",
        ],
        capsys,
    )
    assert rc == 0, err
    plan = json.loads(out)
    assert plan["run_dir"] == str(previous_run)
    assert plan["ultralytics_train_kwargs"] == {"resume": True, "device": "0", "workers": 8}


def test_resume_allows_device_and_workers(previous_run, capsys):
    rc, out, _ = run(
        [
            "--config",
            str(BASELINE),
            "--resume",
            str(previous_run / "weights" / "last.pt"),
            "--device",
            "cpu",
            "--workers",
            "2",
            "--dry-run",
        ],
        capsys,
    )
    assert rc == 0
    assert json.loads(out)["ultralytics_train_kwargs"]["workers"] == 2


@pytest.mark.parametrize(
    "flag", [["--epochs", "5"], ["--seed", "1"], ["--name", "x"], ["--no-amp"]]
)
def test_resume_rejects_fresh_run_options(previous_run, capsys, flag):
    rc, _, err = run(
        [
            "--config",
            str(BASELINE),
            "--resume",
            str(previous_run / "weights" / "last.pt"),
            *flag,
            "--dry-run",
        ],
        capsys,
    )
    assert rc == 2 and "cannot be combined with --resume" in err


def test_resume_missing_checkpoint(runs_root, capsys):
    rc, _, err = run(
        [
            "--config",
            str(BASELINE),
            "--resume",
            str(runs_root / "x" / "weights" / "last.pt"),
            "--dry-run",
        ],
        capsys,
    )
    assert rc == 2 and "resume checkpoint not found" in err


def test_resume_requires_run_dir_layout(tmp_path, capsys):
    ckpt = tmp_path / "loose" / "last.pt"
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"x")
    rc, _, err = run(["--config", str(BASELINE), "--resume", str(ckpt), "--dry-run"], capsys)
    assert rc == 2 and "weights/last.pt" in err


# --- dataset YAML validation ----------------------------------------------------------------


def write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def test_view_yaml_passes(make_view):
    _, view = make_view()
    data = check_data_yaml(view / "data.yaml", NAMES)
    assert data["names"] == {0: "smoke", 1: "fire"}


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d, v: d.update(path="relative/dir"), "absolute"),
        (lambda d, v: d.update(path=str(v / "elsewhere")), "does not exist"),
        (lambda d, v: d.update(names={0: "fire", 1: "smoke"}), "class order"),
        (lambda d, v: d.update(names=["smoke", "fire", "person"], nc=3), "class order"),
        (lambda d, v: d.update(nc=3), "nc=3"),
        (lambda d, v: d.update(download="https://example.com/x.zip"), "download"),
        (lambda d, v: d.pop("val"), "missing required split 'val'"),
        (lambda d, v: d.update(val="nope/images"), "not found"),
        (lambda d, v: d.update(test="val/images"), "'test' must not be"),
        (lambda d, v: d.update(val="train/images"), "must be different"),
    ],
)
def test_invalid_data_yaml(make_view, tmp_path, mutate, message):
    _, view = make_view()
    data = yaml.safe_load((view / "data.yaml").read_text())
    mutate(data, view)
    path = write_yaml(tmp_path / "bad.yaml", data)
    with pytest.raises(PreflightError, match=message):
        check_data_yaml(path, NAMES)


def test_empty_split_directory(make_view, tmp_path):
    _, view = make_view()
    (view / "empty" / "images").mkdir(parents=True)
    data = yaml.safe_load((view / "data.yaml").read_text())
    data["val"] = "empty/images"
    with pytest.raises(PreflightError, match="empty"):
        check_data_yaml(write_yaml(tmp_path / "d.yaml", data), NAMES)


def test_raw_dataset_requires_explicit_opt_in(
    make_dataset, fake_weights, runs_root, tmp_path, capsys
):
    raw = make_dataset({"train": {"a": "0 0.5 0.5 0.1 0.1\n"}, "val": {"b": ""}})
    raw_yaml = write_yaml(
        tmp_path / "raw.yaml",
        {"path": str(raw), "train": "train/images", "val": "val/images", "names": NAMES},
    )
    args = [
        "--config",
        str(BASELINE),
        "--data",
        str(raw_yaml),
        "--weights",
        str(fake_weights),
        "--project",
        str(runs_root),
        "--dry-run",
    ]
    rc, _, err = run(args, capsys)
    assert rc == 2 and "not a FireWatch dataset view" in err
    rc, _, _ = run([*args, "--allow-unprepared-data"], capsys)
    assert rc == 0


def test_dangling_image_link_is_detected(make_dataset, tmp_path):
    if not symlinks_supported(tmp_path):
        pytest.skip("symlinks not permitted on this system")
    from firewatch.data.yolo_view import build_yolo_view

    raw = make_dataset({"train": {"a": "0 0.5 0.5 0.1 0.1\n"}, "val": {"b": ""}})
    view = tmp_path / "view"
    build_yolo_view(raw, view, {"train": "train", "val": "val"}, NAMES, link_mode="symlink")
    (raw / "train" / "images" / "a.jpg").unlink()  # source moved/deleted
    with pytest.raises(PreflightError, match="dangling"):
        check_data_yaml(view / "data.yaml", NAMES)


# --- runner contract with a fake Ultralytics ---------------------------------------------------


class FakeTrainer:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.best = save_dir / "weights" / "best.pt"
        self.last = save_dir / "weights" / "last.pt"
        self.metrics = {"metrics/mAP50-95(B)": 0.0, "fitness": float("nan")}


class FakeYOLO:
    calls: list = []
    fail_with: Exception | None = None

    def __init__(self, weights):
        FakeYOLO.calls.append(("init", weights))
        self.weights = weights
        self.trainer = None

    def train(self, **kwargs):
        FakeYOLO.calls.append(("train", kwargs))
        # Ultralytics reads these at import time; they must already be set.
        FakeYOLO.calls.append(
            ("env", os.environ.get("YOLO_OFFLINE"), os.environ.get("YOLO_AUTOINSTALL"))
        )
        if FakeYOLO.fail_with:
            raise FakeYOLO.fail_with
        if kwargs.get("resume"):  # like Ultralytics: continue in the checkpoint's run dir
            save_dir = Path(self.weights).parent.parent
        else:
            save_dir = Path(kwargs["project"]) / kwargs["name"]
        (save_dir / "weights").mkdir(parents=True, exist_ok=True)
        (save_dir / "weights" / "best.pt").write_bytes(b"best")
        (save_dir / "weights" / "last.pt").write_bytes(b"last")
        self.trainer = FakeTrainer(save_dir)


@pytest.fixture
def fake_yolo(monkeypatch):
    FakeYOLO.calls = []
    FakeYOLO.fail_with = None
    monkeypatch.setattr(runner, "load_yolo_class", lambda: FakeYOLO)
    return FakeYOLO


def test_training_run_records_provenance(base_args, runs_root, fake_yolo, fake_weights, capsys):
    rc, out, err = run(base_args, capsys)
    assert rc == 0, err
    assert fake_yolo.calls[0] == ("init", str(fake_weights))
    kwargs = fake_yolo.calls[1][1]
    assert kwargs["project"] == str(runs_root) and kwargs["name"] == "exp1"
    assert fake_yolo.calls[2] == ("env", "1", "false")

    run_dir = runs_root / "exp1"
    record = json.loads((run_dir / RUN_METADATA_NAME).read_text())
    assert record["status"] == "completed" and record["mode"] == "fresh"
    assert record["options"]["seed"] == 42
    assert len(record["start_weights_sha256"]) == 64
    assert record["best_checkpoint"] == str(run_dir / "weights" / "best.pt")
    assert record["final_val_metrics"]["fitness"] is None  # NaN serialised as null
    assert "commit" in record["git"] and "diff" not in record["git"]
    assert record["environment"]["python"]
    assert "use_deterministic_algorithms" in record["reproducibility"]["note"]
    saved_cfg = yaml.safe_load((run_dir / "firewatch_config.yaml").read_text())
    assert saved_cfg["dataset"]["class_names"] == NAMES
    assert "best=" in out


def test_training_failure_is_recorded(base_args, runs_root, fake_yolo, capsys):
    fake_yolo.fail_with = RuntimeError("CUDA out of memory")
    rc, _, err = run(base_args, capsys)
    assert rc == 1 and "training failed" in err
    record = json.loads((runs_root / "exp1" / RUN_METADATA_NAME).read_text())
    assert record["status"] == "failed" and "out of memory" in record["error"]


def test_resume_run_writes_separate_record(previous_run, fake_yolo, capsys):
    ckpt = previous_run / "weights" / "last.pt"
    rc, _, err = run(["--config", str(BASELINE), "--resume", str(ckpt)], capsys)
    assert rc == 0, err
    assert fake_yolo.calls[0] == ("init", str(ckpt))
    assert fake_yolo.calls[1][1] == {"resume": True, "device": "0", "workers": 8}
    records = list(previous_run.glob("firewatch_resume_*.json"))
    assert len(records) == 1 and json.loads(records[0].read_text())["mode"] == "resume"


def test_missing_ultralytics_is_reported(base_args, capsys, monkeypatch):
    def missing():
        raise runner.PrerequisiteError("Ultralytics is not installed")

    monkeypatch.setattr(runner, "load_yolo_class", missing)
    rc, _, err = run(base_args, capsys)
    assert rc == 2 and "not installed" in err


def test_real_loader_reports_missing_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", None)  # makes the import fail
    with pytest.raises(runner.PrerequisiteError, match=r"pip install -e '\.\[train\]'"):
        runner.load_yolo_class()
    monkeypatch.delitem(sys.modules, "ultralytics")


def test_options_resolution_without_cli(make_view, fake_weights, tmp_path):
    _, view = make_view()
    cfg = load_config(BASELINE)
    cfg["train"].update(
        data=str(view / "data.yaml"), weights=str(fake_weights), project=str(tmp_path / "r")
    )

    class Args:
        pass

    opts = resolve_train_options(cfg, Args(), BASELINE)
    assert opts.run_dir == tmp_path / "r" / "baseline"
    assert opts.class_names == ("smoke", "fire")
