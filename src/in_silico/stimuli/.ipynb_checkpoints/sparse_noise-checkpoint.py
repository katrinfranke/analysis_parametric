from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SparseNoiseSpec:
    num_samples: int = 12
    fps: float = 30.0
    size_hw: tuple[int, int] = (236, 420)

    # dot size control:
    # - int: fixed size
    # - (min_px, max_px): sample uniformly per dot (inclusive)
    # - (v1, v2, v3, ...): sample uniformly from that discrete set
    square_size_px: int | tuple[int, int] | tuple[int, ...] = 10

    dots_per_frame: int = 10  # means "dots per PATTERN"
    contrast: float = 1.0
    mean_lum: float = 0.5
    seed: int = 0
    rgb: bool = True

    # timing (in frames / samples)
    dot_offset_samples: int = 0
    dot_duration_samples: int | None = None


def _sample_square_size_px(spec: SparseNoiseSpec, rng: np.random.Generator) -> int:
    """
    Returns an integer pixel size for a single dot, based on spec.square_size_px.
    """
    s = spec.square_size_px

    # Fixed size
    if isinstance(s, int):
        return max(1, int(s))

    # Range (min,max) - inclusive
    if isinstance(s, tuple) and len(s) == 2 and all(isinstance(v, int) for v in s):
        lo, hi = int(s[0]), int(s[1])
        if lo <= 0 or hi <= 0:
            raise ValueError(f"square_size_px range must be positive, got {s}")
        if hi < lo:
            lo, hi = hi, lo
        return int(rng.integers(lo, hi + 1))

    # Discrete set
    if isinstance(s, tuple) and len(s) >= 1 and all(isinstance(v, int) for v in s):
        vals = np.asarray(s, dtype=np.int32)
        if np.any(vals <= 0):
            raise ValueError(f"square_size_px choices must be positive, got {s}")
        return int(rng.choice(vals))

    raise TypeError(
        "square_size_px must be int, (min,max) int tuple, or tuple of ints. "
        f"Got {type(s)}: {s}"
    )


def make_sparse_noise(spec: SparseNoiseSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      frames:
        - if spec.rgb: (T, 3, H, W) float32 in [0,1]
        - else:        (T, 1, H, W) float32 in [0,1]
      timestamps: (T,) float32 seconds
      events:
        - if spec.rgb: (N, 8) int32 [start_frame_idx, y, x, polarity, size_px, r255, g255, b255]
        - else:        (N, 5) int32 [start_frame_idx, y, x, polarity, size_px]
    Notes:
      - same *pattern* persists for dot_duration_samples frames within the window
      - each dot has its own random RGB color (direction), scaled by signed contrast
      - per-dot varying size is supported via spec.square_size_px
    """
    rng = np.random.default_rng(spec.seed)

    H, W = spec.size_hw
    T = int(spec.num_samples)
    timestamps = (np.arange(T) / spec.fps).astype(np.float32)

    # window [start, end)
    start = int(spec.dot_offset_samples)
    if start < 0:
        raise ValueError(f"dot_offset_samples must be >= 0, got {start}")

    if spec.dot_duration_samples is None:
        end = T
    else:
        dur = int(spec.dot_duration_samples)
        if dur < 0:
            raise ValueError(f"dot_duration_samples must be >= 0, got {dur}")
        end = start + dur

    start = min(start, T)
    end = min(max(end, start), T)

    # ----- RGB version
    if spec.rgb:
        # full movie background
        frames = np.full((T, 3, H, W), spec.mean_lum, dtype=np.float32)

        # if no active window, no events
        if end == start:
            return frames, timestamps, np.empty((0, 8), dtype=np.int32)

        # ONE RGB pattern, reused across the whole window
        pattern = np.full((3, H, W), spec.mean_lum, dtype=np.float32)

        # events: [start, y, x, pol, size_px, r255, g255, b255]
        events = np.empty((spec.dots_per_frame, 8), dtype=np.int32)

        for k in range(spec.dots_per_frame):
            y = int(rng.integers(0, H))
            x = int(rng.integers(0, W))
            pol = int(rng.choice([-1, 1]))

            size_px = _sample_square_size_px(spec, rng)
            half = size_px // 2

            # random RGB direction (unit vector)
            c = rng.random(3).astype(np.float32)
            c /= (np.linalg.norm(c) + 1e-8)

            # signed amplitude around mean
            amp = pol * (spec.contrast * 0.5)

            y0, y1 = max(0, y - half), min(H, y + half + 1)
            x0, x1 = max(0, x - half), min(W, x + half + 1)

            # apply color dot: mean + amp * color_direction
            rgb_val = np.clip(spec.mean_lum + amp * c[:, None, None], 0.0, 1.0)
            pattern[:, y0:y1, x0:x1] = rgb_val

            # store dot color direction as 0..255 ints (for reproducibility/inspection)
            r255, g255, b255 = (np.clip(c, 0.0, 1.0) * 255.0).astype(np.int32)
            events[k] = (start, y, x, pol, size_px, r255, g255, b255)

        # stamp same pattern into every frame in the window
        frames[start:end] = pattern[None]
        return frames, timestamps, events

    # ----- Luminance version
    frames_lum = np.full((T, H, W), spec.mean_lum, dtype=np.float32)

    if end == start:
        frames = frames_lum[:, None, :, :]
        return frames.astype(np.float32), timestamps, np.empty((0, 5), dtype=np.int32)

    pattern = np.full((H, W), spec.mean_lum, dtype=np.float32)

    # events: [start, y, x, pol, size_px]
    events = np.empty((spec.dots_per_frame, 5), dtype=np.int32)

    for k in range(spec.dots_per_frame):
        y = int(rng.integers(0, H))
        x = int(rng.integers(0, W))
        pol = int(rng.choice([-1, 1]))

        size_px = _sample_square_size_px(spec, rng)
        half = size_px // 2

        val = float(np.clip(spec.mean_lum + pol * (spec.contrast * 0.5), 0.0, 1.0))

        y0, y1 = max(0, y - half), min(H, y + half + 1)
        x0, x1 = max(0, x - half), min(W, x + half + 1)
        pattern[y0:y1, x0:x1] = val

        events[k] = (start, y, x, pol, size_px)

    frames_lum[start:end] = pattern[None]
    frames = frames_lum[:, None, :, :]
    return frames.astype(np.float32), timestamps, events
