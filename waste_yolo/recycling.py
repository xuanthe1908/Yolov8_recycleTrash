from __future__ import annotations

from pathlib import Path

import yaml

_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent


def default_recycling_path() -> Path:
    return _PROJECT_ROOT / "config" / "recycling.yaml"


def load_recycling_config(path: str | Path | None = None) -> dict[str, bool]:
    p = Path(path) if path else default_recycling_path()
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    classes = data.get("classes") or {}
    return {str(k).strip(): bool(v) for k, v in classes.items()}


def is_recyclable(class_name: str, mapping: dict[str, bool] | None = None) -> bool:
    m = mapping if mapping is not None else load_recycling_config()
    return m.get(class_name.strip(), False)
