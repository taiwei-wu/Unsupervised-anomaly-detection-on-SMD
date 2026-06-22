# Unsupervised anomaly detection on SMD

This repository contains the code and report for the DL4TS project on unsupervised anomaly detection in multivariate time series. The experiment compares two methods on the Server Machine Dataset (SMD):

1. A convolutional autoencoder trained only on normal SMD windows.
2. A zero-shot MANTIS time-series foundation model used as embedding extractor.

The default configuration uses `machine-1-1`, 96-step windows, train-only standardization, and the same evaluation code for both methods. 
Metrics are reported with the original train 99.5% quantile threshold and with an EVT peak-over-threshold threshold fitted on normal training scores.

## Repository structure

```text
configs/default.yaml        Experiment configuration
scripts/run_experiment.py   Main training, scoring, evaluation, and plotting script
src/dl2ts/                  Project package
outputs/machine-1-1/        Reproduced metrics and figures from the submitted run
```

Raw SMD files are not included.

## Data

Download or copy SMD into the repository root. The code expects this layout:

```text
SMD/
  train/machine-1-1.txt
  test/machine-1-1.txt
  test_label/machine-1-1.txt
```

The original dataset is distributed with OmniAnomaly:
<https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset>

## Environment

Use Python 3.11.

Create an environment and install the requirements:

```bash
python -m pip install -r requirements.txt
```

## Run the experiment

From the repository root:

```bash
python scripts/run_experiment.py --config configs/default.yaml
```

The script trains the autoencoder, extracts frozen raw MANTIS embeddings, computes metrics, and writes:

```text
outputs/machine-1-1/metrics.csv
outputs/machine-1-1/scores.npz
outputs/machine-1-1/summary.json
outputs/machine-1-1/autoencoder_history.csv
outputs/machine-1-1/figures/*.png
```

The raw MANTIS-only script recomputes the frozen 9728-dimensional embedding score without training the autoencoder:

```bash
python scripts/run_mantis_raw.py --config configs/default.yaml
```

It writes `outputs/machine-1-1/mantis_raw_9728/metrics.csv` and `summary.json`.

The submitted run obtained:

| Method | Threshold | AUROC | AUPRC | F1 | Point-adjusted F1 |
|---|---|---:|---:|---:|---:|
| Autoencoder | train q99.5 | 0.947 | 0.564 | 0.241 | 0.241 |
| Autoencoder | EVT-POT | 0.947 | 0.564 | 0.243 | 0.243 |
| MANTIS 9728-D | train q99.5 | 0.832 | 0.327 | 0.431 | 0.573 |
| MANTIS 9728-D | EVT-POT | 0.832 | 0.327 | 0.430 | 0.571 |

