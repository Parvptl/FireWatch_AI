import hashlib
import sys
from pathlib import Path

import pytest

from firewatch.data.yolo_view import build_yolo_view

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE = REPO_ROOT / "configs" / "baseline.yaml"
SMOKE_TEST = REPO_ROOT / "configs" / "smoke_test.yaml"
NAMES = ["smoke", "fire"]

# Small but realistic dataset: normal boxes, a background image, an
# out-of-bounds box, a zero-size box next to a valid one, and an image whose
# only box is degenerate.
DEFAULT_SPLITS = {
    "train": {
        "t_smoke": "0 0.5 0.5 0.2 0.2\n",
        "t_fire": "1 0.3 0.3 0.1 0.1\n",
        "t_bg": "",
        "t_oob": "0 0.5 0.5 1.05 0.4\n",
        "t_mixed": "1 0.5 0.5 0.0 0.0\n0 0.4 0.4 0.2 0.2\n",
        "t_alldrop": "1 0.5 0.5 0.0 0.1\n",
    },
    "val": {"v_smoke": "0 0.5 0.5 0.2 0.2\n", "v_bg": ""},
    "test": {"x_fire": "1 0.5 0.5 0.2 0.2\n"},
}


def tree_digest(root: Path) -> dict[str, str]:
    """Relative path -> sha256 for every file under root (to prove it is unchanged)."""
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def symlinks_supported(tmp_path: Path) -> bool:
    target = tmp_path / "_target"
    target.write_text("x")
    try:
        (tmp_path / "_link").symlink_to(target)
    except (OSError, NotImplementedError):
        return False
    return True


@pytest.fixture
def make_dataset(tmp_path):
    """Build a tiny synthetic D-Fire style dataset. Images are small placeholder files;
    only file names matter for validation. Stems starting with 'orphan' get no image."""

    def _make(splits: dict[str, dict[str, str | None]]) -> Path:
        root = tmp_path / "dfire"
        for split, files in splits.items():
            (root / split / "images").mkdir(parents=True)
            (root / split / "labels").mkdir(parents=True)
            for stem, label in files.items():
                if not stem.startswith("orphan"):
                    (root / split / "images" / f"{stem}.jpg").write_bytes(stem.encode())
                if label is not None:
                    (root / split / "labels" / f"{stem}.txt").write_text(label)
        return root

    return _make


@pytest.fixture
def make_view(tmp_path, make_dataset):
    """Build the default dataset and a copy-mode view of it; returns (raw_root, view_root)."""

    def _make(splits=None, **kwargs):
        splits = splits or DEFAULT_SPLITS
        raw = make_dataset(splits)
        view = tmp_path / "view"
        kwargs.setdefault("link_mode", "copy")
        build_yolo_view(raw, view, {s: s for s in splits}, NAMES, **kwargs)
        return raw, view

    return _make


@pytest.fixture
def fake_weights(tmp_path) -> Path:
    path = tmp_path / "weights" / "yolo26n.pt"
    path.parent.mkdir()
    path.write_bytes(b"not really a checkpoint")
    return path


@pytest.fixture
def runs_root(tmp_path) -> Path:
    return tmp_path / "runs"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Keep user/cluster environment variables from leaking into tests."""
    for var in (
        "FIREWATCH_DATA_ROOT",
        "FIREWATCH_DATA_YAML",
        "FIREWATCH_RUNS_ROOT",
        "YOLO_OFFLINE",
        "YOLO_AUTOINSTALL",
    ):
        monkeypatch.delenv(var, raising=False)
    yield
    assert "ultralytics" not in sys.modules, "tests must never import ultralytics"
    for leftover in ("weights", "runs"):
        assert not (REPO_ROOT / leftover).exists(), f"a test wrote {leftover}/ into the repo"
