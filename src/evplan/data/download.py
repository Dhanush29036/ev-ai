"""Data download — Kaggle datasets via kagglehub with graceful fallback."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from evplan.utils.config import load_config, PROJECT_ROOT
from evplan.utils.logging import get_logger

logger = get_logger(__name__)


def _try_kagglehub(kaggle_ref: str, dest_dir: Path) -> bool:
    """Attempt download via kagglehub. Return True on success."""
    try:
        import kagglehub  # type: ignore

        logger.info("Downloading %s via kagglehub …", kaggle_ref)
        path = kagglehub.dataset_download(kaggle_ref)
        src = Path(path)
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in src.rglob("*"):
            if f.is_file():
                rel = f.relative_to(src)
                target = dest_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
        logger.info("Downloaded to %s", dest_dir)
        return True
    except Exception as exc:
        logger.warning("kagglehub download failed: %s", exc)
        return False


def _print_manual_instructions(dataset_name: str, kaggle_ref: str, dest_dir: Path) -> None:
    """Print clear manual download instructions."""
    border = "=" * 70
    logger.warning(
        "\n%s\nMANUAL DOWNLOAD REQUIRED — %s\n%s\n"
        "1. Go to: https://www.kaggle.com/datasets/%s\n"
        "2. Click 'Download' and unzip.\n"
        "3. Place ALL extracted files in: %s\n"
        "4. Re-run the pipeline.\n%s",
        border, dataset_name, border, kaggle_ref, dest_dir.resolve(), border,
    )


def download_all(force: bool = False) -> dict[str, bool]:
    """Download all configured datasets. Returns {name: success}."""
    cfg = load_config()
    results: dict[str, bool] = {}

    for ds_key, ds_cfg in cfg["datasets"].items():
        if not ds_cfg.get("enabled", True):
            logger.info("Dataset '%s' disabled in config — skipping.", ds_key)
            continue

        name = ds_cfg["name"]
        ref = ds_cfg["kaggle_ref"]
        dest = PROJECT_ROOT / ds_cfg["local_dir"]

        # Check if already downloaded
        if not force and dest.exists() and any(dest.iterdir()):
            logger.info("Dataset '%s' already present at %s", name, dest)
            results[name] = True
            continue

        # Try kagglehub
        ok = _try_kagglehub(ref, dest)
        if not ok:
            _print_manual_instructions(name, ref, dest)
            dest.mkdir(parents=True, exist_ok=True)
        results[name] = ok

    return results


def list_dataset_files(dataset_name: str) -> list[Path]:
    """Return all files in a dataset's raw directory."""
    cfg = load_config()
    for ds_cfg in cfg["datasets"].values():
        if ds_cfg["name"] == dataset_name:
            base = PROJECT_ROOT / ds_cfg["local_dir"]
            if base.exists():
                return sorted(base.rglob("*.*"))
            return []
    return []
