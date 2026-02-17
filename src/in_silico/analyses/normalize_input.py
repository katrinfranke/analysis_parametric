# normalize_input.py
from __future__ import annotations
import numpy as np

# RGB statistics in 0–255 space
RGB_MEAN_255 = np.array([101.53574522, 93.05977233, 81.9272337], dtype=np.float32)
RGB_STD_255  = np.array([65.57974361, 63.30442289, 65.71941174], dtype=np.float32)


def normalize_input(
    x: np.ndarray,
    *,
    assume_rgb: bool = True,
) -> np.ndarray:
    """
    Normalize RGB input using fixed 0–255 mean/std.

    Accepts:
        (T, C, H, W)
        (C, H, W)

    Supports:
        uint8 in [0,255]
        float in [0,255]
        float in [0,1]  (auto-detected and scaled)

    Returns:
        float32 normalized array (same shape as input)
    """

    if x.ndim not in (3, 4):
        raise ValueError(f"Expected 3D or 4D tensor, got shape {x.shape}")

    x = x.astype(np.float32)

    # Detect if float in [0,1]
    if x.max() <= 1.0:
        x = x * 255.0

    if assume_rgb:
        if x.shape[-3] != 3:
            raise ValueError(
                f"Expected 3 channels in RGB mode, got shape {x.shape}"
            )

        # reshape mean/std for broadcasting
        if x.ndim == 4:
            mean = RGB_MEAN_255[None, :, None, None]
            std  = RGB_STD_255[None, :, None, None]
        else:  # (C,H,W)
            mean = RGB_MEAN_255[:, None, None]
            std  = RGB_STD_255[:, None, None]

        x = (x - mean) / std
    else:
        raise NotImplementedError("Non-RGB normalization not implemented.")

    return x
