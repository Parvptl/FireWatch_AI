import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import BASELINE, REPO_ROOT
from firewatch.evaluation import evaluate as ev

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import evaluate as evaluate_cli  # noqa: E402


@pytest.fixture
def trained_run(tmp_path):
    run_dir = tmp_path / "runs" / "exp1"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best.pt").write_bytes(b"best")
    (run_dir / "args.yaml").write_text("imgsz: 512\n")
    return run_dir


@pytest.fixture
def eval_args(make_view, trained_run):
    _, view = make_view()
    return [
        "--config",
        str(BASELINE),
        "--weights",
        str(trained_run / "weights" / "best.pt"),
        "--data",
        str(view / "data.yaml"),
    ]


def run(args, capsys):
    rc = evaluate_cli.main(args)
    out, err = capsys.readouterr()
    return rc, out, err


def test_dry_run_defaults(eval_args, trained_run, capsys):
    rc, out, err = run([*eval_args, "--dry-run"], capsys)
    assert rc == 0, err
    opts = json.loads(out)
    assert opts["split"] == "val"
    assert opts["imgsz"] == 512  # taken from the run's args.yaml
    assert opts["output_dir"].startswith(str(trained_run / "eval" / "val-best-"))


def test_test_split_requires_final_flag(eval_args, capsys):
    rc, _, err = run([*eval_args, "--split", "test", "--dry-run"], capsys)
    assert rc == 2 and "--final-test" in err
    rc, _, _ = run([*eval_args, "--split", "test", "--final-test", "--dry-run"], capsys)
    assert rc == 0


def test_unknown_split_rejected_by_argparse(eval_args):
    with pytest.raises(SystemExit) as exc:
        evaluate_cli.main([*eval_args, "--split", "train"])
    assert exc.value.code == 2


def test_missing_checkpoint(eval_args, tmp_path, capsys):
    args = [*eval_args, "--dry-run"]
    args[args.index("--weights") + 1] = str(tmp_path / "missing.pt")
    rc, _, err = run(args, capsys)
    assert rc == 2 and "checkpoint not found" in err


def test_output_dir_inside_repo_refused(eval_args, capsys):
    rc, _, err = run(
        [*eval_args, "--output-dir", str(REPO_ROOT / "outputs" / "e"), "--dry-run"], capsys
    )
    assert rc == 2 and "outside the Git repository" in err


def test_loose_checkpoint_needs_output_dir(make_view, tmp_path, capsys):
    _, view = make_view()
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"x")
    args = ["--config", str(BASELINE), "--weights", str(ckpt), "--data", str(view / "data.yaml")]
    rc, _, err = run([*args, "--dry-run"], capsys)
    assert rc == 2 and "--output-dir is required" in err
    rc, out, _ = run([*args, "--output-dir", str(tmp_path / "e"), "--dry-run"], capsys)
    assert rc == 0 and json.loads(out)["imgsz"] == 640


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--imgsz", "100"], "multiple of 32"),
        (["--conf", "1.5"], "--conf"),
        (["--batch", "0"], "batch"),
    ],
)
def test_invalid_eval_options(eval_args, capsys, extra, message):
    rc, _, err = run([*eval_args, *extra, "--dry-run"], capsys)
    assert rc == 2 and message in err


# --- metrics extraction and runner contract (fake Ultralytics) -------------------------------


class FakeBox:
    mp, mr, map50, map75, map = 0.6, 0.5, 0.55, 0.3, 0.32


class FakeMetrics:
    box = FakeBox()
    fitness = 0.32
    results_dict = {"metrics/mAP50-95(B)": 0.32}
    speed = {"inference": 1.5}

    def summary(self):
        return [
            {"Class": "smoke", "Images": 2, "Instances": 3, "mAP50": 0.5, "mAP50-95": 0.3},
            {"Class": "fire", "Images": 1, "Instances": 1, "mAP50": 0.6, "mAP50-95": 0.34},
        ]


def test_metrics_to_dict():
    d = ev.metrics_to_dict(FakeMetrics())
    assert d["overall"]["mAP50-95"] == 0.32 and d["overall"]["precision"] == 0.6
    assert [r["Class"] for r in d["per_class"]] == ["smoke", "fire"]


class FakeYOLO:
    names = {0: "smoke", 1: "fire"}
    val_kwargs: dict = {}

    def __init__(self, weights):
        self.weights = weights

    def val(self, **kwargs):
        FakeYOLO.val_kwargs = kwargs
        return FakeMetrics()


@pytest.fixture
def fake_yolo(monkeypatch):
    monkeypatch.setattr(ev, "load_yolo_class", lambda: FakeYOLO)
    FakeYOLO.names = {0: "smoke", 1: "fire"}
    return FakeYOLO


def test_val_evaluation_writes_metrics(eval_args, trained_run, fake_yolo, capsys):
    rc, out, err = run(eval_args, capsys)
    assert rc == 0, err
    kw = fake_yolo.val_kwargs
    assert kw["split"] == "val" and kw["imgsz"] == 512 and kw["conf"] == 0.001
    out_dir = Path(kw["project"]) / kw["name"]
    result = json.loads((out_dir / "metrics.json").read_text())
    assert result["metrics"]["per_class"][1]["Class"] == "fire"
    assert len(result["checkpoint_sha256"]) == 64
    assert not (trained_run / ev.TEST_LOG_NAME).exists()


def test_final_test_evaluation_is_logged(eval_args, trained_run, fake_yolo, capsys):
    rc, _, err = run([*eval_args, "--split", "test", "--final-test"], capsys)
    assert rc == 0, err
    lines = (trained_run / ev.TEST_LOG_NAME).read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["mAP50-95"] == 0.32


def test_class_mismatch_in_checkpoint(eval_args, fake_yolo, capsys):
    fake_yolo.names = {0: "fire", 1: "smoke"}
    rc, _, err = run(eval_args, capsys)
    assert rc == 2 and "do not match" in err


def test_eval_options_namespace_validation(trained_run, make_view):
    _, view = make_view()
    args = SimpleNamespace(
        weights=str(trained_run / "weights" / "best.pt"),
        data=str(view / "data.yaml"),
        split="val",
        final_test=False,
        imgsz=None,
        batch=4,
        conf=0.001,
        iou=0.7,
        device="0,1",
        workers=0,
        output_dir=None,
        plots=False,
        allow_network=False,
    )
    with pytest.raises(ValueError, match="multi-GPU"):
        ev.resolve_eval_options(args, ["smoke", "fire"])
