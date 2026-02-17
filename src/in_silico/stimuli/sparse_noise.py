from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SparseNoiseSpec:
    num_samples: int = 12
    fps: float = 30.0
    size_hw: tuple[int, int] = (236, 420)
    square_size_px: int = 10
    dots_per_frame: int = 10          # now means "dots per PATTERN"
    contrast: float = 1.0
    mean_lum: float = 0.5
    seed: int = 0
    rgb: bool = True

    # timing (in frames / samples)
    dot_offset_samples: int = 0
    dot_duration_samples: int | None = None


def make_sparse_noise(spec: SparseNoiseSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      frames: (T, C, H, W) float32 in [0,1]
      timestamps: (T,) float32 seconds
      events: (N, 4) int32 [start_frame_idx, y, x, polarity]
              polarity in {-1, +1}; same pattern persists for dot_duration_samples frames
    """
    rng = np.random.default_rng(spec.seed)

    H, W = spec.size_hw
    T = int(spec.num_samples)
    timestamps = (np.arange(T) / spec.fps).astype(np.float32)

    # full movie is background first
    frames_lum = np.full((T, H, W), spec.mean_lum, dtype=np.float32)

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

    # if no active window, no events
    if end == start:
        if spec.rgb:
            frames = np.repeat(frames_lum[:, None, :, :], 3, axis=1)
        else:
            frames = frames_lum[:, None, :, :]
        return frames.astype(np.float32), timestamps, np.empty((0, 4), dtype=np.int32)

    # ---- build ONE pattern frame (same dots reused across the whole window)
    pattern = np.full((H, W), spec.mean_lum, dtype=np.float32)
    half = spec.square_size_px // 2

    events = np.empty((spec.dots_per_frame, 4), dtype=np.int32)
    k = 0

    for _ in range(spec.dots_per_frame):
        y = int(rng.integers(0, H))
        x = int(rng.integers(0, W))
        pol = int(rng.choice([-1, 1]))

        val = spec.mean_lum + pol * (spec.contrast * 0.5)
        val = float(np.clip(val, 0.0, 1.0))

        y0, y1 = max(0, y - half), min(H, y + half + 1)
        x0, x1 = max(0, x - half), min(W, x + half + 1)
        pattern[y0:y1, x0:x1] = val

        events[k] = (start, y, x, pol)  # start frame marks when this pattern begins
        k += 1

    events = events[:k]

    # ---- stamp that same pattern into every frame in the window
    frames_lum[start:end, :, :] = pattern[None, :, :]

    if spec.rgb:
        frames = np.repeat(frames_lum[:, None, :, :], 3, axis=1)
    else:
        frames = frames_lum[:, None, :, :]

    return frames.astype(np.float32), timestamps, events

