"""Path resolution and config loading for the data layer.

Central source of truth for where raw/processed data lives and how each dataset is
described (``config/datasets.yaml``). Reads ``.env`` (via python-dotenv) for the optional
``GLAUCOMA_DATA_DIR`` override and the ``DATAVERSE_TOKEN`` used for gated downloads.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

try:  # python-dotenv only loads the OPTIONAL .env overrides; envs without it (e.g. .venv-torch) still work
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "datasets.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Parse ``config/datasets.yaml`` (cached)."""
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def data_dir() -> Path:
    """Root data directory. Honors ``GLAUCOMA_DATA_DIR``; else ``<repo>/data``."""
    cfg = load_config()
    raw = os.environ.get("GLAUCOMA_DATA_DIR", cfg.get("data_dir", "data"))
    path = Path(raw).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def raw_dir(dataset: str) -> Path:
    """Raw directory for ``dataset`` (e.g. ``"papila"``, ``"harvard_gf"``)."""
    cfg = load_config()
    return data_dir() / cfg[dataset]["raw_subdir"]


def processed_dir() -> Path:
    """Directory for processed parquet outputs."""
    cfg = load_config()
    return data_dir() / cfg.get("processed_subdir", "processed")


def dataset_cfg(dataset: str) -> dict:
    """Config block for a single dataset."""
    return load_config()[dataset]


def dataverse_token() -> str | None:
    """Harvard Dataverse API token, if configured."""
    return os.environ.get("DATAVERSE_TOKEN")
