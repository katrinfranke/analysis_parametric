"""Block-based spatiotemporal white / pink noise stimulus for stRF mapping."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WhiteNoiseSpec:
    """Specification for a block-based spatiotemporal noise stimulus.

    Each frame in the batch is an independent random noise pattern.  The image
    is divided into a regular grid of ``block_size_px × block_size_px`` blocks;
    every pixel within a block shares the same noise value, so ``block_size_px``
    controls the effective spatial resolution of the noise.

    Parameters
    ----------
    num_frames:
        Number of frames per stimulus batch (e.g. 12 for a standard model batch).
    fps:
        Frames per second (used only for generating timestamps if needed).
    size_hw:
        Full spatial dimensions ``(height, width)`` in pixels.
    block_size_px:
        Side length (in pixels) of each noise unit block.  Must be >= 1.
        E.g. ``block_size_px=5`` means each 5×5 pixel region shares one value.
    noise_type:
        ``"white"`` – independent binary (±1) noise per block and frame.
        ``"pink"``  – binary white noise filtered by a 1/f spatiotemporal kernel,
        producing noise with 1/f correlations in both space and time.
    contrast:
        Peak-to-peak contrast in [0, 1].  With binary noise the pixel range is
        ``[mean_lum - contrast/2, mean_lum + contrast/2]``.
    mean_lum:
        Mean luminance in [0, 1].
    seed:
        Base random seed.
    rgb:
        If True, generate three *independent* noise channels (R, G, B).
        If False, generate a single luminance channel and replicate it to 3.
    temporal_weight:
        Controls the relative contribution of temporal frequencies in the 3D
        pink filter (``noise_type="pink"`` only).  The filter is
        ``1 / sqrt(f_x² + f_y² + (temporal_weight * f_t)²)``.
        - Values < 1 → slower temporal variation (longer correlation times).
        - Values > 1 → faster temporal decorrelation.
        A good starting point is ``1.0``; tune relative to your fps and the
        integration window of the neurons you're mapping (DS RGCs ~ 50–100 ms).
    """

    num_frames: int = 12
    fps: float = 30.0
    size_hw: tuple[int, int] = (236, 420)
    block_size_px: int = 5
    noise_type: str = "white"  # "white" | "pink"
    contrast: float = 1.0
    mean_lum: float = 0.5
    seed: int = 0
    rgb: bool = True
    temporal_weight: float = 1.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _binary_block_noise(
    T: int,
    C: int,
    bh: int,
    bw: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return independent binary (±1) block noise, shape (T, C, bh, bw) float32."""
    bits = rng.integers(0, 2, size=(T, C, bh, bw), dtype=np.uint8)
    return (bits.astype(np.float32) * 2.0 - 1.0)


def _apply_pink_filter_3d(
    noise: np.ndarray,
    temporal_weight: float = 1.0,
) -> np.ndarray:
    """Apply a 1/f amplitude filter in 3-D (time × height × width) Fourier space.

    Unlike a purely spatial 2-D filter, this introduces temporal correlations
    across frames, which is necessary for mapping motion-selective neurons via
    reverse correlation.  The filter kernel is:

        H(f_t, f_y, f_x) = 1 / sqrt(f_x² + f_y² + (temporal_weight * f_t)²)

    with the DC component zeroed.

    Parameters
    ----------
    noise:
        Array with shape (T, C, bh, bw).  The filter is applied independently
        to each channel C.
    temporal_weight:
        Scales the temporal frequency axis relative to spatial frequencies.

    Returns
    -------
    np.ndarray, shape (T, C, bh, bw), float32
        Filtered noise, roughly zero-mean.  Values are *not* clipped.
    """
    T, C, bh, bw = noise.shape

    # Build 3-D frequency grid
    ft = np.fft.fftfreq(T).astype(np.float32)[:, None, None]   # (T, 1, 1)
    fy = np.fft.fftfreq(bh).astype(np.float32)[None, :, None]  # (1, bh, 1)
    fx = np.fft.fftfreq(bw).astype(np.float32)[None, None, :]  # (1, 1, bw)

    f = np.sqrt(fx**2 + fy**2 + (temporal_weight * ft)**2)
    f[0, 0, 0] = 1.0        # avoid division by zero at DC
    h_filt = 1.0 / f  # 1/f^0.5 amplitude → 1/f power spectrum
    h_filt[0, 0, 0] = 0.0  # zero DC

    result = np.empty_like(noise, dtype=np.float32)
    for c in range(C):
        spectrum = np.fft.fftn(noise[:, c, :, :])          # 3-D FFT over (T, bh, bw)
        result[:, c, :, :] = np.real(
            np.fft.ifftn(spectrum * h_filt)
        ).astype(np.float32)

    return result


def _pink_block_noise(
    T: int,
    C: int,
    bh: int,
    bw: int,
    rng: np.random.Generator,
    temporal_weight: float = 1.0,
) -> np.ndarray:
    """Return 3-D pink-filtered block noise, shape (T, C, bh, bw) float32 in [-1, 1]."""
    white = _binary_block_noise(T, C, bh, bw, rng).astype(np.float32)
    pink = _apply_pink_filter_3d(white, temporal_weight=temporal_weight)

    # Normalise to [-1, 1]: scale so ≈4 σ fits the range
    std = pink.std()
    if std > 0.0:
        pink = pink / (4.0 * std)
    return np.clip(pink, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_white_noise(spec: WhiteNoiseSpec) -> np.ndarray:
    """Generate a block-based spatiotemporal noise stimulus.

    Each of the ``num_frames`` frames contains an independent random noise
    pattern (white) or a temporally correlated pattern (pink).  The noise is
    generated at block-grid resolution and then up-sampled by simple pixel
    repetition so that every ``block_size_px × block_size_px`` region shares
    a single value.

    Parameters
    ----------
    spec:
        Stimulus specification.

    Returns
    -------
    np.ndarray, shape (T, 3, H, W), float32 in [0, 1]
        Stimulus frames ready for model input (after normalisation).
        Always returns 3 channels: independent for ``rgb=True``,
        replicated luminance for ``rgb=False``.
    """
    rng = np.random.default_rng(spec.seed)

    H, W = spec.size_hw
    T = spec.num_frames
    C = 3 if spec.rgb else 1

    bsz = max(1, int(spec.block_size_px))
    bh = max(1, H // bsz)
    bw = max(1, W // bsz)

    if spec.noise_type == "white":
        noise = _binary_block_noise(T, C, bh, bw, rng)   # (T, C, bh, bw) in [-1, 1]
    elif spec.noise_type == "pink":
        noise = _pink_block_noise(
            T, C, bh, bw, rng,
            temporal_weight=spec.temporal_weight,
        )                                                  # (T, C, bh, bw) in [-1, 1]
    else:
        raise ValueError(
            f"Unknown noise_type={spec.noise_type!r}. "
            "Use 'white' or 'pink'."
        )

    # Scale from [-1, 1] → [mean_lum - contrast/2, mean_lum + contrast/2]
    frames_block = np.clip(
        spec.mean_lum + 0.5 * spec.contrast * noise, 0.0, 1.0
    )  # (T, C, bh, bw)

    # Up-sample: repeat each block cell bsz times along H and W
    frames_up = np.repeat(np.repeat(frames_block, bsz, axis=-2), bsz, axis=-1)

    # Crop to exact (H, W) in case H or W is not divisible by bsz
    frames_up = frames_up[:, :, :H, :W]   # (T, C, H, W)

    # Replicate luminance channel to 3 channels for the model
    if not spec.rgb:
        frames_up = np.repeat(frames_up, 3, axis=1)   # (T, 3, H, W)

    return frames_up
