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
from dl2ts.methods import score_mantis_raw_embeddings, set_seed, train_autoencoder
from dl2ts.metrics import evaluate_scores, evt_pot_threshold, quantile_threshold
from dl2ts.plots import plot_curves, plot_metric_bars, plot_mixed_scale_scores, plot_scores, plot_training_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DL2TS SMD anomaly detection experiments.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    output_dir = cfg.output_dir / cfg.machine
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    train_raw, test_raw, labels = load_smd_machine(cfg.data_dir, cfg.machine)
    standardizer = fit_standardizer(train_raw)
    train = standardizer.transform(train_raw)
    test = standardizer.transform(test_raw)
    train_windows = sliding_windows(train, cfg.window_size)
    test_windows = sliding_windows(test, cfg.window_size)
    indices, aligned_labels = align_window_scores(
        np.zeros(len(test_windows), dtype=np.float64), len(test), cfg.window_size
    )
    aligned_labels = labels[indices]

    print(f"Loaded {cfg.machine}: train={train.shape}, test={test.shape}, windows={train_windows.shape}")

    ae = train_autoencoder(
        train_windows=train_windows,
        test_windows=test_windows,
        output_dir=output_dir,
        seed=cfg.seed,
        epochs=cfg.ae_epochs,
        batch_size=cfg.ae_batch_size,
        learning_rate=cfg.ae_learning_rate,
        weight_decay=cfg.ae_weight_decay,
        patience=cfg.ae_patience,
        hidden_channels=cfg.ae_hidden_channels,
    )
    mantis = score_mantis_raw_embeddings(
        train_windows=train_windows,
        test_windows=test_windows,
        batch_size=cfg.mantis_batch_size,
        requested_device=cfg.mantis_device,
    )

    methods = {
        "Autoencoder": (ae.train_scores, ae.test_scores),
        "MANTIS-9728": (mantis.train_scores, mantis.test_scores),
    }
    metrics_rows = []
    thresholds = {}
    test_scores = {}
    for name, (train_scores, scores) in methods.items():
        test_scores[name] = scores
        threshold_specs = {
            "train_q995": quantile_threshold(train_scores, cfg.threshold_quantile),
            "evt_pot": evt_pot_threshold(train_scores),
        }
        thresholds[name] = threshold_specs["train_q995"]
        for threshold_method, threshold in threshold_specs.items():
            row = {"method": name, "threshold_method": threshold_method}
            row.update(evaluate_scores(aligned_labels, scores, threshold))
            metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    pd.DataFrame(ae.history).to_csv(output_dir / "autoencoder_history.csv", index=False)

    np.savez_compressed(
        output_dir / "scores.npz",
        timestamps=indices,
        labels=aligned_labels,
        ae_train_scores=ae.train_scores,
        ae_test_scores=ae.test_scores,
        mantis_train_scores=mantis.train_scores,
        mantis_test_scores=mantis.test_scores,
    )

    summary = {
        "machine": cfg.machine,
        "seed": cfg.seed,
        "window_size": cfg.window_size,
        "threshold_quantile": cfg.threshold_quantile,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "n_windows": int(len(train_windows)),
        "mantis_embedding_dim": int(mantis.embedding_dim),
        "mantis_projection_dim": mantis.projection_dim,
        "mantis_device": mantis.device,
        "threshold_methods": ["train_q995", "evt_pot"],
        "autoencoder_checkpoint": str(ae.checkpoint_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_scores(
        figure_dir / "score_timeline.png",
        indices,
        aligned_labels,
        test_scores,
        thresholds,
        dpi=cfg.figure_dpi,
    )
    plot_mixed_scale_scores(
        figure_dir / "score_timeline_mixed_scale.png",
        indices,
        aligned_labels,
        ae.test_scores,
        mantis.test_scores,
        thresholds["Autoencoder"],
        thresholds["MANTIS-9728"],
        dpi=max(cfg.figure_dpi, 420),
    )
    plot_curves(figure_dir / "roc_pr_curves.png", aligned_labels, test_scores, dpi=cfg.figure_dpi)
    plot_metric_bars(figure_dir / "metric_bars.png", metrics_path, dpi=cfg.figure_dpi)
    plot_training_history(figure_dir / "autoencoder_training.png", ae.history, dpi=cfg.figure_dpi)

    print(metrics_df.to_string(index=False))
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()

