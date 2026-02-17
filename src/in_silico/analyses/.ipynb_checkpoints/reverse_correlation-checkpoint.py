# src/psa/analyses/rf_reverse_corr.py
from __future__ import annotations
import numpy as np


def zscore_time(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """Z-score along time axis (default last axis)."""
    m = x.mean(axis=axis, keepdims=True)
    s = x.std(axis=axis, keepdims=True)
    return (x - m) / (s + eps)


def reverse_corr_rf(
    frames_tchw: np.ndarray,
    pred_units_t: np.ndarray,
    lags: list[int] | tuple[int, ...] = (0, 1, 2, 3),
    channel: int = 0,
    skip_frames: int = 0,
    center_stim: bool = True,
    center_resp: bool = True,
) -> np.ndarray:
    """
    Reverse-correlation RF estimate.

    Args:
      frames_tchw: (T, C, H, W) float32 in [0,1]
      pred_units_t: (U, T_pred) float32 (predicted responses)
      lags: list of non-negative integer lags in frames
            RF at lag L correlates response[t] with stimulus[t-L]
      channel: which channel to use (0 for luminance if rgb repeated)
      skip_frames: if your model drops first N frames (e.g. skip_samples),
                   set skip_frames=N so pred aligns to frames[skip_frames:].
      center_stim/center_resp: subtract mean across time before correlation.

    Returns:
      rf: (U, len(lags), H, W) float32
    """
    assert frames_tchw.ndim == 4, "frames must be (T,C,H,W)"
    assert pred_units_t.ndim == 2, "pred must be (U,T_pred)"

    stim = frames_tchw[:, channel]  # (T,H,W)
    T_stim = stim.shape[0]
    U, T_pred = pred_units_t.shape

    # Align: predictions correspond to stim indices [skip_frames : skip_frames+T_pred)
    stim_aligned = stim[skip_frames: skip_frames + T_pred]
    T = min(stim_aligned.shape[0], T_pred)
    stim_aligned = stim_aligned[:T]
    resp = pred_units_t[:, :T]

    if center_stim:
        stim_aligned = stim_aligned - stim_aligned.mean(axis=0, keepdims=True)
    if center_resp:
        resp = resp - resp.mean(axis=1, keepdims=True)

    H, W = stim_aligned.shape[1:]
    rf = np.zeros((U, len(lags), H, W), dtype=np.float32)

    for j, lag in enumerate(lags):
        if lag < 0:
            raise ValueError("lags must be non-negative ints")
        if lag >= T:
            continue

        # response at time t corresponds to stimulus at time t-lag
        stim_l = stim_aligned[: T - lag]          # (T-lag,H,W)
        resp_l = resp[:, lag: lag + (T - lag)]    # (U,T-lag)

        # Correlation-like estimate: sum_t resp[u,t] * stim[t,h,w]
        rf[:, j] = np.einsum("ut,thw->uhw", resp_l, stim_l) / float(T - lag)

    return rf