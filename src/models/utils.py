"""
Shared model utilities: normalization statistics, checkpoint I/O, parameter counting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    config: dict,
    scheduler=None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    # Atomic write: save to temp file first, then rename.  If torch.save()
    # crashes mid-write (disk full, power loss), the original checkpoint is
    # not corrupted — the temp file is simply discarded.
    tmp_path = path.with_suffix(".pt.tmp")
    torch.save(payload, str(tmp_path))
    tmp_path.replace(path)
    log.info("Checkpoint saved to %s (epoch %d, val_loss=%.4f)", path, epoch, val_loss)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: str = "cpu",
) -> dict:
    path = Path(path)
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        log.info("Scheduler state restored from checkpoint.")
    log.info("Loaded checkpoint from %s (epoch %d)", path, ckpt.get("epoch", -1))
    return ckpt


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# Learning rate scheduling
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
    n_epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build LR scheduler with optional linear warmup.

    If config['n_epochs_warmup'] > 0, chains a LinearLR warmup with the
    main scheduler via SequentialLR.  The warmup ramps from lr/10 to lr
    over n_epochs_warmup epochs, then the main schedule takes over.
    """
    sched_name = config.get("lr_scheduler", "cosine_annealing_with_restarts")
    n_warmup = config.get("n_epochs_warmup", 0)

    if sched_name == "cosine_annealing_with_restarts":
        main_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.get("T_0", 50),
            T_mult=config.get("T_mult", 2),
        )
    elif sched_name == "cosine":
        main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    else:
        main_sched = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    if n_warmup > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=n_warmup,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, main_sched],
            milestones=[n_warmup],
        )
    return main_sched


# ---------------------------------------------------------------------------
# Normaliser persistence
# ---------------------------------------------------------------------------

def save_normalizer(path: str | Path, mean: np.ndarray, std: np.ndarray) -> None:
    np.savez(str(path), mean=mean, std=std)


def load_normalizer(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(str(path))
    return d["mean"], d["std"]


# ---------------------------------------------------------------------------
# Embedding PCA visualisation helper
# ---------------------------------------------------------------------------

def pca_embeddings(embeddings: np.ndarray, labels: np.ndarray, n_components: int = 2) -> dict:
    """Run PCA on embedding vectors and return dict for plotting."""
    from sklearn.decomposition import PCA  # noqa: PLC0415
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(embeddings)
    return {
        "coords": coords,
        "labels": labels,
        "explained_variance": pca.explained_variance_ratio_,
    }


# ---------------------------------------------------------------------------
# Posterior serialization (safer than raw pickle)
# ---------------------------------------------------------------------------

def save_posterior(
    path: str | Path,
    posterior,
    dm_model: str,
    density_estimator_state_dict: dict,
    prior_bounds: dict,
    sbi_version: str | None = None,
) -> None:
    """Save a trained SBI posterior in a reproducible format.

    Uses torch.save for the density estimator state dict (portable, version-
    independent) and stores metadata needed to reconstruct the posterior
    without relying on pickle's implicit class serialization.

    The full posterior object is still pickled as a fallback (sbi's posterior
    class is complex to reconstruct from scratch), but the state dict allows
    recovery if the pickle breaks across sbi versions.

    Args:
        path: Output .pt file path.
        posterior: sbi posterior object.
        dm_model: DM model name (for metadata).
        density_estimator_state_dict: State dict from the density estimator.
        prior_bounds: Dict with 'low' and 'high' tensors for BoxUniform prior.
        sbi_version: Optional sbi version string for compatibility tracking.
    """
    import pickle  # noqa: PLC0415

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if sbi_version is None:
        try:
            import sbi
            sbi_version = sbi.__version__
        except (ImportError, AttributeError):
            sbi_version = "unknown"

    payload = {
        "dm_model": dm_model,
        "density_estimator_state_dict": density_estimator_state_dict,
        "prior_bounds": prior_bounds,
        "sbi_version": sbi_version,
        "posterior_pickle": pickle.dumps(posterior),
    }
    tmp_path = path.with_suffix(".pt.tmp")
    torch.save(payload, str(tmp_path))
    tmp_path.replace(path)
    log.info("Posterior saved to %s (dm_model=%s, sbi=%s)", path, dm_model, sbi_version)


def load_posterior(
    path: str | Path,
    device: str = "cpu",
) -> tuple:
    """Load a posterior saved by save_posterior.

    Attempts to unpickle the full posterior object first. If that fails
    (e.g. sbi version mismatch), returns the density estimator state dict
    and prior bounds so the caller can reconstruct.

    Args:
        path: .pt file saved by save_posterior.
        device: Device to map tensors to.

    Returns:
        (posterior_or_None, metadata_dict) where metadata_dict contains
        'dm_model', 'density_estimator_state_dict', 'prior_bounds', 'sbi_version'.
        posterior is None if unpickling failed.
    """
    import pickle  # noqa: PLC0415

    payload = torch.load(str(path), map_location=device, weights_only=False)

    posterior = None
    if "posterior_pickle" in payload:
        try:
            posterior = pickle.loads(payload["posterior_pickle"])
            log.info("Posterior loaded from pickle (sbi=%s)", payload.get("sbi_version"))
        except Exception as e:
            log.warning(
                "Failed to unpickle posterior (sbi version mismatch?): %s. "
                "Falling back to state dict for manual reconstruction.", e,
            )

    metadata = {
        "dm_model": payload.get("dm_model"),
        "density_estimator_state_dict": payload.get("density_estimator_state_dict"),
        "prior_bounds": payload.get("prior_bounds"),
        "sbi_version": payload.get("sbi_version"),
    }
    return posterior, metadata
