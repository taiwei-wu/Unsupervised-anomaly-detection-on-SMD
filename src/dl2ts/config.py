from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    data_dir: Path = Path("SMD")
    output_dir: Path = Path("outputs")
    machine: str = "machine-1-1"
    seed: int = 42
    window_size: int = 96
    threshold_quantile: float = 0.995
    ae_epochs: int = 80
    ae_batch_size: int = 256
    ae_learning_rate: float = 1e-3
    ae_weight_decay: float = 1e-5
    ae_patience: int = 12
    ae_hidden_channels: int = 128
    mantis_batch_size: int = 128
    mantis_device: str = "auto"
    figure_dpi: int = 220
    extra: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path | None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    known = {field.name for field in ExperimentConfig.__dataclass_fields__.values()}
    kwargs = {k: v for k, v in raw.items() if k in known}
    extra = {k: v for k, v in raw.items() if k not in known}
    cfg = ExperimentConfig(**kwargs)
    cfg.extra = extra
    cfg.data_dir = Path(cfg.data_dir)
    cfg.output_dir = Path(cfg.output_dir)
    return cfg

