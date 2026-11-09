import hashlib
import json
import random
import string
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def resolve_run_name(base_name: str) -> str:
    """Append a timestamp + 4-char random suffix to avoid collisions across repeated runs."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{base_name}__{ts}-{suffix}"


def stable_indices_hash(indices: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for idx in indices:
        digest.update(f"{idx},".encode("utf-8"))
    return digest.hexdigest()

