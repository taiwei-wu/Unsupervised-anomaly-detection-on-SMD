from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .metrics import roc_pr_curves


def set_plot_style() -> None:
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.bbox": "tight",
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
    })


def plot_scores(
    output_path: Path,
    timestamps: np.ndarray,
    labels: np.ndarray,
    method_scores: dict[str, np.ndarray],
    thresholds: dict[str, float],
    dpi: int,
) -> None:
    set_plot_style()
    fig, axes = plt.subplots(len(method_scores), 1, figsize=(7.0, 3.6), sharex=True)
    if len(method_scores) == 1:
        axes = [axes]
    anomaly = labels.astype(bool)
    for ax, (name, scores) in zip(axes, method_scores.items()):
        ax.plot(timestamps, scores, linewidth=0.8, label=f"{name} score")
        ax.axhline(thresholds[name], color="tab:red", linestyle="--", linewidth=0.9, label="train 99.5% threshold")
        ymin, ymax = ax.get_ylim()
        ax.fill_between(timestamps, ymin, ymax, where=anomaly, color="tab:orange", alpha=0.18, label="anomaly label")
        ax.set_ylabel("score")
        ax.set_title(name)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("test timestamp")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_mixed_scale_scores(
    output_path: Path,
    timestamps: np.ndarray,
    labels: np.ndarray,
    ae_scores: np.ndarray,
    mantis_scores: np.ndarray,
    ae_threshold: float,
    mantis_threshold: float,
    dpi: int,
) -> None:
    set_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 3.7), sharex=True)
    anomaly = labels.astype(bool)

    axes[0].plot(timestamps, np.maximum(ae_scores, 1e-8), linewidth=0.75, label="Autoencoder score")
    axes[0].axhline(ae_threshold, color="tab:red", linestyle="--", linewidth=1.0, label="train 99.5% threshold")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("log score")
    axes[0].set_title("Autoencoder")

    axes[1].plot(timestamps, mantis_scores, linewidth=0.75, label="MANTIS score")
    axes[1].axhline(mantis_threshold, color="tab:red", linestyle="--", linewidth=1.0, label="train 99.5% threshold")
    axes[1].set_ylabel("score")
    axes[1].set_title("MANTIS")
    axes[1].set_xlabel("test timestamp")

    for ax in axes:
        ymin, ymax = ax.get_ylim()
        ax.fill_between(timestamps, ymin, ymax, where=anomaly, color="tab:orange", alpha=0.16, label="anomaly label")
        ax.set_ylim(ymin, ymax)
        ax.legend(loc="upper right", ncol=3, frameon=True)

    fig.tight_layout(h_pad=0.8)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_curves(output_path: Path, labels: np.ndarray, method_scores: dict[str, np.ndarray], dpi: int) -> None:
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for name, scores in method_scores.items():
        curves = roc_pr_curves(labels, scores)
        axes[0].plot(curves["fpr"], curves["tpr"], label=name)
        axes[1].plot(curves["recall"], curves["precision"], label=name)
    axes[0].plot([0, 1], [0, 1], color="0.5", linestyle=":", linewidth=0.8)
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("ROC curve")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("Precision-recall curve")
    for ax in axes:
        ax.legend()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_metric_bars(output_path: Path, metrics_csv: Path, dpi: int) -> None:
    set_plot_style()
    metrics = pd.read_csv(metrics_csv)
    keep = ["auroc", "auprc", "f1", "pa_f1"]
    plot_df = metrics.melt(id_vars="method", value_vars=keep, var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    sns.barplot(data=plot_df, x="metric", y="value", hue="method", ax=ax)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("value")
    ax.set_title("Detection metrics on SMD machine-1-1")
    ax.legend(title="")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_training_history(output_path: Path, history: list[dict[str, float]], dpi: int) -> None:
    set_plot_style()
    hist = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    ax.plot(hist["epoch"], hist["train_loss"], label="train")
    ax.plot(hist["epoch"], hist["val_loss"], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Autoencoder training curve")
    ax.legend()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

