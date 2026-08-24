from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def ensure_supported_python() -> None:
    if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
        raise RuntimeError(
            f"Python {sys.version.split()[0]} is unsupported. Use Python 3.10-3.12; "
            "Google Colab's Python 3.12 runtime is supported."
        )


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def select_device(use_cuda: bool) -> torch.device:
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. In Colab select Runtime > Change runtime "
            "type > T4 GPU, reconnect, and rerun setup. For a slow CPU smoke test pass --no-cuda."
        )
    return torch.device("cuda" if use_cuda else "cpu")


def validate_demo_metadata(demo_path: str, expected_control_mode: str) -> Path:
    h5_path = Path(demo_path).expanduser()
    json_path = h5_path.with_suffix(".json")
    if not h5_path.is_file():
        raise FileNotFoundError(f"Missing demonstrations: {h5_path}")
    if not json_path.is_file():
        raise FileNotFoundError(f"Missing demonstration metadata: {json_path}")
    with json_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    env_kwargs = metadata.get("env_info", {}).get("env_kwargs", {})
    control_mode = env_kwargs.get("control_mode")
    if control_mode is None and metadata.get("episodes"):
        control_mode = metadata["episodes"][0].get("control_mode")
    if control_mode != expected_control_mode:
        raise ValueError(
            f"Control-mode mismatch: dataset uses {control_mode!r}, config uses "
            f"{expected_control_mode!r}. Replay the demos with the configured controller."
        )
    return h5_path


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
