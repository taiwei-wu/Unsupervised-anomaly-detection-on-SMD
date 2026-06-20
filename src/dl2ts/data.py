from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def load_smd_machine(data_dir: str | Path, machine: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    train = np.loadtxt(data_dir / "train" / f"{machine}.txt", delimiter=",").astype(np.float32)
    test = np.loadtxt(data_dir / "test" / f"{machine}.txt", delimiter=",").astype(np.float32)
    labels = np.loadtxt(data_dir / "test_label" / f"{machine}.txt", delimiter=",").astype(np.int64)
    if len(test) != len(labels):
        raise ValueError(f"Test length {len(test)} does not match label length {len(labels)} for {machine}.")
    return train, test, labels


def fit_standardizer(train: np.ndarray, eps: float = 1e-6) -> Standardizer:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return Standardizer(mean=mean.astype(np.float32), std=std.astype(np.float32))


def sliding_windows(x: np.ndarray, window_size: int) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D time-by-channel array, got {x.shape}.")
    if len(x) < window_size:
        raise ValueError(f"Series length {len(x)} is shorter than window_size={window_size}.")
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=window_size, axis=0)
    return np.swapaxes(windows, 1, 2).copy().astype(np.float32)


def align_window_scores(scores: np.ndarray, series_length: int, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Map each window score to the timestamp at the end of that window."""
    expected = series_length - window_size + 1
    if len(scores) != expected:
        raise ValueError(f"Expected {expected} scores, got {len(scores)}.")
    indices = np.arange(window_size - 1, series_length)
    return indices, scores.astype(np.float64)

