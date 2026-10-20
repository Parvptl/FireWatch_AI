"""Application configuration loading, validation and environment overrides."""

import pytest

from app_fakes import make_config
from firewatch.app.config import load_app_config
from firewatch.utils.config import ConfigError


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in (
        "FIREWATCH_APP_CONFIG",
        "FIREWATCH_MODEL_PATH",
        "FIREWATCH_APP_HOME",
        "FIREWATCH_API_TOKEN",
    ):
        monkeypatch.delenv(v, raising=False)


def test_shipped_config_loads(tmp_path, monkeypatch):
    monkeypatch.setenv("FIREWATCH_APP_HOME", str(tmp_path))
    from conftest import REPO_ROOT

    cfg = load_app_config(
        REPO_ROOT / "configs" / "app.yaml",
        {
            "video": {"input_dir": str(tmp_path / "in")},
            "storage": {"data_dir": str(tmp_path / "d")},
        },
    )
    assert cfg.model.imgsz == 960 and cfg.model.class_names == ["smoke", "fire"]
    assert cfg.weights_path.name == "best.pt"
    assert cfg.db_path == tmp_path / "d" / "firewatch.sqlite3"


def test_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("FIREWATCH_MODEL_PATH", str(tmp_path / "x.pt"))
    monkeypatch.setenv("FIREWATCH_APP_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FIREWATCH_API_TOKEN", "s3cret")
    cfg = load_app_config(None)
    assert cfg.weights_path == tmp_path / "x.pt"
    assert cfg.input_dir == tmp_path / "home" / "input"
    assert cfg.api.token == "s3cret" and "token" not in cfg.public_dict()["api"]


@pytest.mark.parametrize(
    "over, msg",
    [
        ({"model": {"imgsz": 1000}}, "multiple of 32"),
        ({"events": {"activate_hits": 0}}, "activate_hits"),
        ({"model": {"conf_threshold": 1.5}}, "conf_threshold"),
        ({"model": {"nope": 1}}, "unknown key"),
        ({"bogus": {}}, "unknown section"),
        ({"jobs": {"max_queued": "many"}}, "integer"),
        ({"api": {"host": "0.0.0.0"}}, "token"),
        ({"model": {"device": "gpu"}}, "device"),
    ],
)
def test_invalid_values_rejected(tmp_path, over, msg):
    with pytest.raises(ConfigError, match=msg):
        make_config(tmp_path, **over)


def test_data_dir_inside_repo_rejected(tmp_path):
    from conftest import REPO_ROOT

    with pytest.raises(ConfigError, match="outside the Git repository"):
        make_config(tmp_path, storage={"data_dir": str(REPO_ROOT / "outputs")})


def test_missing_config_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_app_config(tmp_path / "missing.yaml")
