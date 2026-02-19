"""Drifting sinusoidal grating stimulus for SF/TF/color/direction tuning.

Gratings are parametrised by:
  - direction of motion (degrees, 0 = rightward, counter-clockwise positive)
  - spatial frequency (cycles per degree of visual angle)
  - temporal frequency (Hz)
  - color axis (achromatic, L-M isoluminant, or S-cone isoluminant)

The conversion between pixels and visual angle is given by ``px_per_deg``
(default 6.7 px/deg, matching the experimental setup).

Color axes are defined in RGB space relative to a mid-gray background
(mean_lum = 0.5):

  - ``"achromatic"`` – equal R/G/B modulation (luminance grating)
  - ``"lm"``         – L-M isoluminant (red-green, no luminance change):
                       ΔR = 1, ΔG = −(L_coeff / M_coeff) ≈ −0.51, ΔB = 0
  - ``"s"``          – S-cone isoluminant (blue-yellow, no luminance change):
                       ΔR = ΔG = −S_coeff / (L_coeff + M_coeff) ≈ −0.129, ΔB = 1

All raw color vectors are normalised to unit length so that ``contrast``
has a consistent meaning across conditions.  After adding the modulation to
``mean_lum``, pixel values are clipped to [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# -------------------------------------------------------------------------
# CIE luminance coefficients for the sRGB primaries
# -------------------------------------------------------------------------
_L_COEFF = 0.299
_M_COEFF = 0.587
_S_COEFF = 0.114  # blue luminance coefficient

# Raw (un-normalised) color direction vectors in (R, G, B)
_COLOR_VECTORS_RAW: dict[str, np.ndarray] = {
    # Equal R/G/B – pure luminance
    "achromatic": np.array([1.0, 1.0, 1.0], dtype=np.float64),
    # Isoluminant L-M: dR = 1, dG = -(L/M), dB = 0  →  0.299*1 + 0.587*(−L/M) = 0 ✓
    "lm": np.array([1.0, -(_L_COEFF / _M_COEFF), 0.0], dtype=np.float64),
    # Isoluminant S-cone: dB = 1, dR = dG = −S/(L+M)
    "s": np.array(
        [-_S_COEFF / (_L_COEFF + _M_COEFF),
         -_S_COEFF / (_L_COEFF + _M_COEFF),
         1.0],
        dtype=np.float64,
    ),
}

# Unit-length color vectors (float64 for precision, cast to float32 during use)
COLOR_VECTORS: dict[str, np.ndarray] = {
    name: (v / np.linalg.norm(v)).astype(np.float32)
    for name, v in _COLOR_VECTORS_RAW.items()
}

#: Ordered tuple of supported color condition names
COLORS: tuple[str, ...] = ("achromatic", "lm", "s")

#: Default pixels-per-degree conversion factor
PX_PER_DEG: float = 6.7


# -------------------------------------------------------------------------
# Spec
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class DriftingGratingSpec:
    """Specification for a drifting sinusoidal grating stimulus.

    Parameters
    ----------
    size_hw:
        Full spatial dimensions ``(height, width)`` in pixels.
    fps:
        Frames per second.
    px_per_deg:
        Pixels per degree of visual angle.
    num_frames:
        Total number of frames to generate.  For single-window model inputs
        (e.g. 12-frame batches) set ``num_frames=12``.  For longer sequences
        intended to be fed to the model via a sliding window, set this to the
        total number of frames needed (e.g. 66 for a ~2 s sequence at 30 fps
        with a 12-frame window and 9-frame stride).
    direction_deg:
        Direction of grating motion in degrees (0 = rightward, increasing
        counter-clockwise).
    sf_cpd:
        Spatial frequency in cycles per degree of visual angle.
    tf_hz:
        Temporal frequency in Hz (cycles per second).
    color:
        Color axis: ``"achromatic"``, ``"lm"``, or ``"s"``.
    phase_deg:
        Initial spatial phase offset in degrees.  A phase of 0 places a
        sinewave zero-crossing at the image centre.
    contrast:
        Modulation amplitude (half the peak-to-peak range) in [0, 1].
        Applied after normalising the color vector to unit length.
        Values above 0.5 may cause clipping to [0, 1] depending on
        ``mean_lum`` and the color axis.
    mean_lum:
        Background / mean luminance in [0, 1].
    """

    size_hw: tuple[int, int] = (236, 420)
    fps: float = 30.0
    px_per_deg: float = PX_PER_DEG
    num_frames: int = 66

    direction_deg: float = 0.0
    sf_cpd: float = 1.0
    tf_hz: float = 2.0
    color: str = "achromatic"
    phase_deg: float = 0.0

    contrast: float = 0.5
    mean_lum: float = 0.5


# -------------------------------------------------------------------------
# Stimulus generation
# -------------------------------------------------------------------------

def make_drifting_grating(spec: DriftingGratingSpec) -> np.ndarray:
    """Generate drifting sinusoidal grating frames.

    Parameters
    ----------
    spec:
        Grating specification.

    Returns
    -------
    np.ndarray, shape ``(T, 3, H, W)``, float32 in [0, 1]
        Stimulus frames suitable for model input (after normalisation via
        :func:`~in_silico.analyses.dotmapping.normalize_input`).
    """
    if spec.color not in COLOR_VECTORS:
        raise ValueError(
            f"Unknown color {spec.color!r}. Choose from {list(COLOR_VECTORS)!r}."
        )

    H, W = spec.size_hw
    T = spec.num_frames

    # Spatial frequency in cycles per pixel
    sf_cpp = float(spec.sf_cpd) / float(spec.px_per_deg)

    # Pixel coordinate grid, centred at image midpoint
    ys = (np.arange(H, dtype=np.float32) - H / 2.0)
    xs = (np.arange(W, dtype=np.float32) - W / 2.0)
    XX, YY = np.meshgrid(xs, ys)  # (H, W)

    # Project pixel coordinates onto the grating propagation direction
    dir_rad = float(np.deg2rad(spec.direction_deg))
    # spatial_phase_map[h, w] = phase (radians) of pixel (h, w)
    spatial_phase_map = (XX * np.cos(dir_rad) + YY * np.sin(dir_rad)) * (
        2.0 * np.pi * sf_cpp
    )  # (H, W)

    # Temporal phase per frame: 2π * tf * t / fps
    ts = np.arange(T, dtype=np.float32)
    temporal_phase = 2.0 * np.pi * float(spec.tf_hz) * ts / float(spec.fps)  # (T,)

    # Initial phase offset
    phase_offset = float(np.deg2rad(spec.phase_deg))

    # Sine wave: sin(spatial − temporal + phase_0)
    # Broadcasting: (1,H,W) − (T,1,1) → (T,H,W)
    sinewave = np.sin(
        spatial_phase_map[np.newaxis, :, :] - temporal_phase[:, np.newaxis, np.newaxis] + phase_offset
    ).astype(np.float32)  # (T, H, W)

    # Apply color direction vector and scale by contrast
    cvec = COLOR_VECTORS[spec.color]  # (3,) float32

    # frames[t, c, h, w] = mean_lum + contrast * cvec[c] * sinewave[t, h, w]
    frames = (
        float(spec.mean_lum)
        + float(spec.contrast) * cvec[np.newaxis, :, np.newaxis, np.newaxis] * sinewave[:, np.newaxis, :, :]
    ).astype(np.float32)  # (T, 3, H, W)

    return np.clip(frames, 0.0, 1.0, out=frames)
