from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def make_dataset(tmp_path):
    """Build a tiny synthetic D-Fire style dataset. Images are empty placeholder files;
    only file names matter for validation."""

    def _make(splits: dict[str, dict[str, str | None]]) -> Path:
        root = tmp_path / "dfire"
        for split, files in splits.items():
            (root / split / "images").mkdir(parents=True)
            (root / split / "labels").mkdir(parents=True)
            for stem, label in files.items():
                if not stem.startswith("orphan"):
                    (root / split / "images" / f"{stem}.jpg").write_bytes(b"")
                if label is not None:
                    (root / split / "labels" / f"{stem}.txt").write_text(label)
        return root

    return _make
