from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.random_projection import GaussianRandomProjection
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from .models import ConvWindowAutoencoder


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@dataclass
class AutoencoderResult:
    train_scores: np.ndarray
    test_scores: np.ndarray
    history: list[dict[str, float]]
    checkpoint_path: Path


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_autoencoder(
    train_windows: np.ndarray,
    test_windows: np.ndarray,
    output_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    hidden_channels: int,
) -> AutoencoderResult:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device()

    tensor = torch.from_numpy(train_windows)
    dataset = TensorDataset(tensor)
    val_size = max(int(0.15 * len(dataset)), 1)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = ConvWindowAutoencoder(train_windows.shape[-1], hidden_channels=hidden_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(batch)
            n_train += len(batch)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                loss = criterion(model(batch), batch)
                val_loss += float(loss.item()) * len(batch)
                n_val += len(batch)

        row = {"epoch": epoch, "train_loss": train_loss / n_train, "val_loss": val_loss / n_val}
        history.append(row)
        if row["val_loss"] < best_val - 1e-6:
            best_val = row["val_loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint_path = output_dir / "autoencoder.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "n_channels": train_windows.shape[-1],
            "hidden_channels": hidden_channels,
            "history": history,
        },
        checkpoint_path,
    )

    train_scores = score_autoencoder(model, train_windows, batch_size=batch_size, device=device)
    test_scores = score_autoencoder(model, test_windows, batch_size=batch_size, device=device)
    return AutoencoderResult(train_scores=train_scores, test_scores=test_scores, history=history, checkpoint_path=checkpoint_path)


def score_autoencoder(
    model: ConvWindowAutoencoder,
    windows: np.ndarray,
    batch_size: int,
    device: torch.device | None = None,
) -> np.ndarray:
    device = device or _device()
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(windows)), batch_size=batch_size, shuffle=False)
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            err = torch.mean((recon - batch) ** 2, dim=(1, 2))
            scores.append(err.detach().cpu().numpy())
    return np.concatenate(scores).astype(np.float64)


@dataclass
class MantisResult:
    train_scores: np.ndarray
    test_scores: np.ndarray
    embedding_dim: int
    projection_dim: int | None
    device: str


def score_mantis_embeddings(
    train_windows: np.ndarray,
    test_windows: np.ndarray,
    batch_size: int,
    projection_dim: int,
    seed: int,
    requested_device: str = "auto",
) -> MantisResult:
    from mantis.architecture import Mantis8M
    from mantis.trainer import MantisTrainer

    device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
    if device == "auto":
        device = "cpu"

    try:
        network = Mantis8M(device=device).from_pretrained("paris-noah/Mantis-8M")
        model = MantisTrainer(network=network, device=device)
        first = _mantis_transform(model, train_windows[: min(batch_size, len(train_windows))])
    except Exception:
        device = "cpu"
        network = Mantis8M(device=device).from_pretrained("paris-noah/Mantis-8M")
        model = MantisTrainer(network=network, device=device)
        first = _mantis_transform(model, train_windows[: min(batch_size, len(train_windows))])

    embedding_dim = first.shape[1]
    projector = GaussianRandomProjection(n_components=projection_dim, random_state=seed)
    projector.fit(first)

    train_proj = []
    for emb in _iter_mantis_embeddings(model, train_windows, batch_size, first_batch=first):
        train_proj.append(projector.transform(emb).astype(np.float32))
    train_proj_arr = np.concatenate(train_proj, axis=0)
    mean = train_proj_arr.mean(axis=0, keepdims=True)
    var = train_proj_arr.var(axis=0, keepdims=True) + 1e-6
    train_scores = np.mean((train_proj_arr - mean) ** 2 / var, axis=1)

    test_scores_chunks = []
    for emb in _iter_mantis_embeddings(model, test_windows, batch_size):
        z = projector.transform(emb).astype(np.float32)
        test_scores_chunks.append(np.mean((z - mean) ** 2 / var, axis=1))
    test_scores = np.concatenate(test_scores_chunks, axis=0)
    return MantisResult(
        train_scores=train_scores.astype(np.float64),
        test_scores=test_scores.astype(np.float64),
        embedding_dim=embedding_dim,
        projection_dim=projection_dim,
        device=device,
    )


def score_mantis_raw_embeddings(
    train_windows: np.ndarray,
    test_windows: np.ndarray,
    batch_size: int,
    requested_device: str = "auto",
) -> MantisResult:
    """Score frozen MANTIS embeddings in their original 9728-dimensional space."""
    from mantis.architecture import Mantis8M
    from mantis.trainer import MantisTrainer

    device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
    if device == "auto":
        device = "cpu"

    try:
        network = Mantis8M(device=device).from_pretrained("paris-noah/Mantis-8M")
        model = MantisTrainer(network=network, device=device)
        first = _mantis_transform(model, train_windows[: min(batch_size, len(train_windows))])
    except Exception:
        device = "cpu"
        network = Mantis8M(device=device).from_pretrained("paris-noah/Mantis-8M")
        model = MantisTrainer(network=network, device=device)
        first = _mantis_transform(model, train_windows[: min(batch_size, len(train_windows))])

    embedding_dim = first.shape[1]
    n = 0
    sum_x = np.zeros(embedding_dim, dtype=np.float64)
    sum_x2 = np.zeros(embedding_dim, dtype=np.float64)
    for emb in _iter_mantis_embeddings(model, train_windows, batch_size, first_batch=first):
        e = emb.astype(np.float64, copy=False)
        n += e.shape[0]
        sum_x += e.sum(axis=0)
        sum_x2 += (e * e).sum(axis=0)

    mean = sum_x / n
    var = np.maximum(sum_x2 / n - mean * mean, 1e-6)

    train_scores_chunks = []
    for emb in _iter_mantis_embeddings(model, train_windows, batch_size):
        e = emb.astype(np.float64, copy=False)
        train_scores_chunks.append(np.mean((e - mean) ** 2 / var, axis=1))
    train_scores = np.concatenate(train_scores_chunks, axis=0)

    test_scores_chunks = []
    for emb in _iter_mantis_embeddings(model, test_windows, batch_size):
        e = emb.astype(np.float64, copy=False)
        test_scores_chunks.append(np.mean((e - mean) ** 2 / var, axis=1))
    test_scores = np.concatenate(test_scores_chunks, axis=0)

    return MantisResult(
        train_scores=train_scores.astype(np.float64),
        test_scores=test_scores.astype(np.float64),
        embedding_dim=embedding_dim,
        projection_dim=None,
        device=device,
    )


def _iter_mantis_embeddings(
    model: object,
    windows: np.ndarray,
    batch_size: int,
    first_batch: np.ndarray | None = None,
) -> Iterable[np.ndarray]:
    start = 0
    if first_batch is not None:
        yield first_batch
        start = len(first_batch)
    for i in tqdm(range(start, len(windows), batch_size), desc="MANTIS embeddings", leave=False):
        yield _mantis_transform(model, windows[i : i + batch_size])


def _mantis_transform(model: object, windows: np.ndarray) -> np.ndarray:
    x = np.swapaxes(windows, 1, 2).astype(np.float32)
    return model.transform(x).astype(np.float32)

