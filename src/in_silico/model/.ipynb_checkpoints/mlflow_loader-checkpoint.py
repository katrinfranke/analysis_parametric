# src/in_silico/model/mlflow_loader.py
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from hydra.utils import instantiate
from mlflow.artifacts import download_artifacts

from experanto.dataloaders import get_multisession_concat_dataloader
from monkey_baselines.model_factory import load_model


@dataclass(frozen=True)
class ModelPaths:
    checkpoint_uri: str
    config_uri: str


@dataclass(frozen=True)
class DataPaths:
    session_dirs: Sequence[str]


def _configure_for_train(cfg):
    """
    Minimal overrides mirrored from your demo notebook for building the model.
    """
    cfg.dataset.modality_config.screen.valid_condition = {"tier": "train"}
    cfg.dataset.modality_config.screen.include_blanks = True
    cfg.dataset.modality_config.screen.sample_stride = 1
    return cfg


def _configure_for_validation(cfg):
    """
    Settings used for validation correlations / neuron selection.
    """
    cfg.dataset.modality_config.screen.valid_condition = {"tier": "validation"}
    cfg.dataset.modality_config.screen.include_blanks = False
    cfg.dataset.modality_config.screen.sample_stride = 1
    return cfg


def load_free_viewing_model_from_mlflow(
    model_paths: ModelPaths,
    data_paths: DataPaths,
    *,
    device: Optional[str] = None,
    strict: bool = True,
    model_type: str = "free_viewing",
    cuda_visible_devices: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_username: Optional[str] = None,
    mlflow_password: Optional[str] = None,
) -> Tuple[torch.nn.Module, int, object, object]:
    """
    One-stop loader:
      - sets optional env vars (CUDA + MLflow)
      - downloads cfg + checkpoint from MLflow artifacts
      - instantiates cfg
      - builds a minimal TRAIN dataloader needed for load_model(...)
      - loads the model
      - builds a VALIDATION dataloader for computing correlations / neuron selection
      - returns (model, skip_samples, cfg, val_dl)

    Notes:
      - We deep-copy cfg to avoid train/val overrides clobbering each other.
      - 'skip_samples' is taken from cfg.trainer.skip_n_samples.
    """

    # Optional env setup
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    if mlflow_tracking_uri is not None:
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
    if mlflow_username is not None:
        os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
    if mlflow_password is not None:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password

    # Download artifacts
    ckpt_path = download_artifacts(model_paths.checkpoint_uri)
    cfg_path = download_artifacts(model_paths.config_uri)

    # Load + instantiate cfg
    cfg = torch.load(cfg_path, weights_only=False)
    cfg = instantiate(cfg)

    # Build TRAIN dataloader (required by load_model)
    cfg_train = copy.deepcopy(cfg)
    cfg_train = _configure_for_train(cfg_train)
    train_dl = get_multisession_concat_dataloader(list(data_paths.session_dirs), cfg_train)

    # Skip samples (super important for alignment)
    skip_samples = int(cfg.trainer.skip_n_samples)

    # Load the model
    model = load_model(cfg_path, ckpt_path, train_dl, strict=strict, model_type=model_type)

    # Device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # Build VALIDATION dataloader (for correlations / neuron selection)
    cfg_val = copy.deepcopy(cfg)
    cfg_val = _configure_for_validation(cfg_val)
    val_dl = get_multisession_concat_dataloader(list(data_paths.session_dirs), cfg_val)

    return model, skip_samples, cfg, val_dl
