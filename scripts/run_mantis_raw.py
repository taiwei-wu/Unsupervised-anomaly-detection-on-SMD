from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl2ts.config import load_config
from dl2ts.data import align_window_scores, fit_standardizer, load_smd_machine, sliding_windows
from dl2ts.methods import score_mantis_raw_embeddings, set_seed
from dl2ts.metrics import evaluate_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Run raw 9728-D MANTIS anomaly scoring.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    output_dir = cfg.output_dir / cfg.machine / "mantis_raw_9728"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_raw, test_raw, labels = load_smd_machine(cfg.data_dir, cfg.machine)
    standardizer = fit_standardizer(train_raw)
    train = standardizer.transform(train_raw)
    test = standardizer.transform(test_raw)
    train_windows = sliding_windows(train, cfg.window_size)
    test_windows = sliding_windows(test, cfg.window_size)
    indices, _ = align_window_scores(np.zeros(len(test_windows), dtype=np.float64), len(test), cfg.window_size)
    aligned_labels = labels[indices]

    mantis = score_mantis_raw_embeddings(
        train_windows=train_windows,
        test_windows=test_windows,
        batch_size=cfg.mantis_batch_size,
        requested_device=cfg.mantis_device,
    )
    threshold = float(np.quantile(mantis.train_scores, cfg.threshold_quantile))
    row = {"method": "MANTIS raw 9728-D"}
    row.update(evaluate_scores(aligned_labels, mantis.test_scores, threshold))
    metrics_df = pd.DataFrame([row])
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    np.savez_compressed(
        output_dir / "scores.npz",
        timestamps=indices,
        labels=aligned_labels,
        train_scores=mantis.train_scores,
        test_scores=mantis.test_scores,
    )
    summary = {
        "machine": cfg.machine,
        "seed": cfg.seed,
        "window_size": cfg.window_size,
        "threshold_quantile": cfg.threshold_quantile,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "n_train_windows": int(len(train_windows)),
        "n_test_windows": int(len(test_windows)),
        "embedding_dim": int(mantis.embedding_dim),
        "projection": None,
        "device": mantis.device,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(metrics_df.to_string(index=False))
    print(f"Saved raw 9728-D outputs to {output_dir}")


if __name__ == "__main__":
    main()
