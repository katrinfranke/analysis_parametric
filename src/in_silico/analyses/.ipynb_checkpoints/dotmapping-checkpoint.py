"""Dotmapping analysis: sparse-noise reverse correlation to estimate spatial STAs."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from in_silico.stimuli.sparse_noise import SparseNoiseSpec, make_sparse_noise

# RGB statistics in 0–255 space (used for model input normalisation)
_RGB_MEAN_255 = np.array([101.53574522, 93.05977233, 81.9272337], dtype=np.float32)
_RGB_STD_255 = np.array([65.57974361, 63.30442289, 65.71941174], dtype=np.float32)


def normalize_input(x: np.ndarray) -> np.ndarray:
    """Normalize RGB input using fixed 0–255 mean/std.

    Parameters
    ----------
    x:
        Array with shape (T, C, H, W) or (C, H, W).
        Accepts uint8 in [0, 255], float in [0, 255], or float in [0, 1]
        (auto-detected and scaled).

    Returns
    -------
    np.ndarray
        float32 normalized array, same shape as input.
    """
    if x.ndim not in (3, 4):
        raise ValueError(f"Expected 3D or 4D input, got shape {x.shape}")

    x = x.astype(np.float32)

    if x.max() <= 1.0:
        x = x * 255.0

    if x.shape[-3] != 3:
        raise ValueError(f"Expected 3 channels, got shape {x.shape}")

    if x.ndim == 4:
        mean = _RGB_MEAN_255[None, :, None, None]
        std = _RGB_STD_255[None, :, None, None]
    else:
        mean = _RGB_MEAN_255[:, None, None]
        std = _RGB_STD_255[:, None, None]

    return (x - mean) / std


def reverse_corr_rf_batch_sum(
    frames_btchw: np.ndarray,
    pred_but: np.ndarray,
    lags: list[int] | tuple[int, ...] = (0, 1, 2, 3),
    channel: int = 0,
    skip_frames: int = 0,
    center_stim: bool = True,
    center_resp: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate reverse-correlation numerator and denominator across a batch.

    Parameters
    ----------
    frames_btchw:
        Stimulus frames, shape (B, T, C, H, W).
    pred_but:
        Neural predictions, shape (B, U, T_pred).
    lags:
        Non-negative integer lags (in frames) at which to compute the RF.
    channel:
        Stimulus channel to use.
    skip_frames:
        Number of leading stimulus frames that have no corresponding prediction.
    center_stim:
        Subtract per-stimulus temporal mean at each pixel before correlating.
    center_resp:
        Subtract per-stimulus per-unit temporal mean before correlating.

    Returns
    -------
    rf_sum : np.ndarray, shape (U, L, H, W)
        Accumulated sum of response-weighted stimulus across (batch, time).
    n_sum : np.ndarray, shape (L,)
        Total number of timepoints contributing per lag (divide rf_sum by this
        at the end to get the mean RF).
    """
    assert frames_btchw.ndim == 5, "frames must be (B,T,C,H,W)"
    assert pred_but.ndim == 3, "pred must be (B,U,T_pred)"

    B, T_stim, C, H, W = frames_btchw.shape
    B2, U, T_pred = pred_but.shape
    assert B == B2

    stim = frames_btchw[:, :, channel, :, :]  # (B,T,H,W)

    stim_aligned = stim[:, skip_frames: skip_frames + T_pred, :, :]
    T_use = min(stim_aligned.shape[1], T_pred)
    stim_aligned = stim_aligned[:, :T_use, :, :]
    resp = pred_but[:, :, :T_use]

    if center_stim:
        stim_aligned = stim_aligned - stim_aligned.mean(axis=1, keepdims=True)
    if center_resp:
        resp = resp - resp.mean(axis=2, keepdims=True)

    L = len(lags)
    rf_sum = np.zeros((U, L, H, W), dtype=np.float32)
    n_sum = np.zeros((L,), dtype=np.int64)

    for j, lag in enumerate(lags):
        if lag < 0:
            raise ValueError("lags must be non-negative ints")
        if lag >= T_use:
            continue
        stim_l = stim_aligned[:, :T_use - lag, :, :]
        resp_l = resp[:, :, lag:lag + (T_use - lag)]
        rf_sum[:, j] += np.einsum("but,bthw->uhw", resp_l, stim_l).astype(np.float32)
        n_sum[j] += B * (T_use - lag)

    return rf_sum, n_sum


def downsample_avg_pool_nchw(x: np.ndarray, factor: int) -> np.ndarray:
    """Average-pool downsample an array with shape (..., C, H, W).

    Parameters
    ----------
    x:
        Array with shape (T, C, H, W) or (N, C, H, W).
    factor:
        Downsampling factor. H and W must be divisible by factor.

    Returns
    -------
    np.ndarray
        Downsampled array with shape (..., C, H // factor, W // factor).
    """
    if factor == 1:
        return x

    *lead, C, H, W = x.shape
    if H % factor != 0 or W % factor != 0:
        raise ValueError(
            f"H and W must be divisible by factor. Got H={H}, W={W}, factor={factor}"
        )

    x = x.reshape(*lead, C, H // factor, factor, W // factor, factor)
    return x.mean(axis=(-1, -3))


def predict_responses(
    wrapper,
    *,
    key: str = "37_3843837605846_0_V3A_V4",
    num_samples: int = 12,
    dot_offset_samples: int = 3,
    dot_duration_samples: int = 6,
    fps: float = 30.0,
    square_size_px: int = 25,
    dots_per_frame: int = 100,
    base_seed: int = 61,
    N: int = 5000,
    batch_size: int = 10,
    win_start: int = 2,
    win_dur: int = 6,
    ds_factor: int = 4,
    dtype=np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Generate sparse-noise stimuli, predict neural responses, and return stored arrays.

    Parameters
    ----------
    wrapper:
        Wrapped model for generating predictions.
    key:
        Session key passed to the wrapper.
    num_samples:
        Number of time samples per stimulus.
    dot_offset_samples:
        Dot onset offset in samples.
    dot_duration_samples:
        Dot duration in samples.
    fps:
        Frames per second.
    square_size_px:
        Dot size in pixels.
    dots_per_frame:
        Number of dots per frame.
    base_seed:
        Base random seed (each stimulus uses base_seed + stimulus_index).
    N:
        Total number of stimuli to generate.
    batch_size:
        Number of stimuli to process per batch.
    win_start:
        Start index of the response window.
    win_dur:
        Duration of the response window in samples.
    ds_factor:
        Downsampling factor for stored stimuli (1 = no downsampling).
    dtype:
        Output dtype for stored arrays.

    Returns
    -------
    frames_all : np.ndarray, shape (N, T, C, H_ds, W_ds)
        Downsampled stimulus frames.
    pred_all : np.ndarray, shape (N, U, T_pred)
        Baseline-shifted neural predictions.
    avg_resp_all : np.ndarray, shape (N, U)
        Mean response over the response window.
    seeds : list[int]
        Random seeds used for each stimulus.
    """
    wrapper.key = key

    base_spec = SparseNoiseSpec(
        num_samples=num_samples,
        dot_offset_samples=dot_offset_samples,
        dot_duration_samples=dot_duration_samples,
        fps=fps,
        square_size_px=square_size_px,
        dots_per_frame=dots_per_frame,
        seed=base_seed,
        rgb=True,
    )

    win_end = win_start + win_dur
    frames_batches: list[np.ndarray] = []
    pred_batches: list[np.ndarray] = []
    avg_batches: list[np.ndarray] = []
    seeds: list[int] = []

    with tqdm(total=N, desc="Predicting responses", unit="stim") as pbar:
        for start in range(0, N, batch_size):
            b = min(batch_size, N - start)
            frames_list, preds_list, seeds_list = [], [], []

            for i in range(b):
                seed_i = base_seed + start + i
                spec_i = replace(base_spec, seed=seed_i)

                frames_raw, _, _ = make_sparse_noise(spec_i)  # (T,C,H,W) in [0,1]
                frames_store = downsample_avg_pool_nchw(
                    frames_raw.astype(np.float32, copy=False), ds_factor
                )
                frames_norm = normalize_input(frames_raw)
                pred_ut, _ = wrapper.predict(frames_norm)  # (U, T_pred)

                frames_list.append(frames_store.astype(dtype, copy=False))
                preds_list.append(pred_ut.astype(dtype, copy=False))
                seeds_list.append(seed_i)
                pbar.update(1)

            frames_btchw = np.stack(frames_list, axis=0)   # (b, T, C, Hds, Wds)
            pred_but = np.stack(preds_list, axis=0)         # (b, U, T_pred)

            # baseline-shift so t=0 is zero
            pred_but = pred_but - pred_but[:, :, [0]]

            T_pred = pred_but.shape[2]
            w0, w1 = max(0, win_start), min(T_pred, win_end)
            avg_bu = (
                pred_but[:, :, w0:w1].mean(axis=2).astype(dtype)
                if w1 > w0
                else np.full((b, pred_but.shape[1]), np.nan, dtype=dtype)
            )

            frames_batches.append(frames_btchw)
            pred_batches.append(pred_but.astype(dtype, copy=False))
            avg_batches.append(avg_bu)
            seeds.extend(seeds_list)

    frames_all = np.concatenate(frames_batches, axis=0)    # (N, T, C, Hds, Wds)
    pred_all = np.concatenate(pred_batches, axis=0)        # (N, U, T_pred)
    avg_resp_all = np.concatenate(avg_batches, axis=0)     # (N, U)

    return frames_all, pred_all, avg_resp_all, seeds


def compute_sta(
    frames_all: np.ndarray,
    pred_all: np.ndarray,
    *,
    t_frame: int = 5,
    skip_frames: int = 0,
    response_offset: int = 0,
    response_duration: int = 6,
    baseline: float = 0.5,
    normalize_weights: bool = True,
    eps: float = 1e-8,
    show_progress: bool = True,
    subtract_stim_mean: bool = True,
    zscore_normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a spatial spike-triggered average (STA) for each neuron.

    Uses a single stimulus frame ``t_frame`` from each movie, weighted by the
    mean neural response over a window aligned to that frame.

    Parameters
    ----------
    frames_all:
        Stored stimulus frames, shape (N, T, C, H, W), values in [0, 1].
    pred_all:
        Neural predictions, shape (N, U, T_pred).
    t_frame:
        Stimulus frame index to use for the STA.
    skip_frames:
        Frames to skip when aligning stimulus to predictions.
    response_offset:
        Offset from the mapped prediction time to start the response window.
    response_duration:
        Number of prediction time steps in the response window.
    baseline:
        Subtracted from frames before weighting (0.5 centres [0,1] frames at grey).
    normalize_weights:
        If True, divide the weighted sum by the total weight per neuron.
    eps:
        Small constant for numerical stability.
    show_progress:
        Show a tqdm progress bar over neurons.
    subtract_stim_mean:
        If True, subtract the ensemble mean stimulus frame before weighting.
    zscore_normalize:
        If True, z-score each neuron's STA across its spatial dimensions.

    Returns
    -------
    sta : np.ndarray, shape (U, C, H, W)
        Spatial STA per neuron.
    denom : np.ndarray, shape (U,)
        Sum of weights per neuron.
    stim_mean_chw : np.ndarray, shape (C, H, W)
        Mean stimulus frame across the ensemble.
    """
    N, T, C, H, W = frames_all.shape
    N2, U, T_pred = pred_all.shape
    assert N == N2, f"frames_all and pred_all must have the same N: {N} vs {N2}"
    assert 0 <= t_frame < T, f"t_frame={t_frame} out of range [0, {T})"

    frame_nchw = frames_all[:, t_frame].astype(np.float32) - float(baseline)  # (N,C,H,W)
    stim_mean_chw = frame_nchw.mean(axis=0)                                    # (C,H,W)
    if subtract_stim_mean:
        frame_nchw = frame_nchw - stim_mean_chw[None]

    t_pred0 = t_frame - int(skip_frames)
    w0 = max(0, t_pred0 + int(response_offset))
    w1 = min(int(T_pred), w0 + int(response_duration))

    if w1 <= w0:
        return (
            np.zeros((U, C, H, W), dtype=np.float32),
            np.zeros((U,), dtype=np.float32),
            stim_mean_chw.astype(np.float32),
        )

    weights_nu = pred_all[:, :, w0:w1].mean(axis=2).astype(np.float32)  # (N, U)

    sta = np.zeros((U, C, H, W), dtype=np.float32)
    denom = np.zeros((U,), dtype=np.float32)

    iterator = (
        tqdm(range(U), desc="Computing STA", unit="neuron")
        if show_progress
        else range(U)
    )
    for u in iterator:
        w_n = weights_nu[:, u]                                              # (N,)
        sta_num = np.einsum("n,nchw->chw", w_n, frame_nchw).astype(np.float32)
        d = float(w_n.sum())
        denom[u] = d
        sta[u] = sta_num / (d + eps) if normalize_weights else sta_num

    if zscore_normalize:
        std_per_neuron = sta.reshape(U, -1).std(axis=1).reshape(U, 1, 1, 1)
        sta = sta / (std_per_neuron + eps)

    return sta, denom, stim_mean_chw.astype(np.float32)


def sta_to_rgb(
    sta_chw: np.ndarray,
    *,
    mode: str = "robust",
    p: float = 99.0,
    gamma: float = 1.0,
    center: float = 0.5,
    clip: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """Convert a signed STA to a displayable RGB image in [0, 1].

    Parameters
    ----------
    sta_chw:
        STA tensor, shape (C, H, W), signed contrast-like values.
    mode:
        Scaling mode: ``"robust"`` uses a percentile of absolute values;
        ``"maxabs"`` uses the maximum absolute value.
    p:
        Percentile for robust scaling (ignored when ``mode="maxabs"``).
    gamma:
        Optional gamma for display.
    center:
        Display value that maps to zero in the STA (default 0.5 = grey).
    clip:
        Whether to clip final values to [0, 1].
    eps:
        Small constant for numerical stability.

    Returns
    -------
    np.ndarray, shape (H, W, 3)
        Displayable RGB image in [0, 1].
    """
    x = sta_chw.astype(np.float32, copy=False)
    if x.ndim != 3:
        raise ValueError(f"sta_to_rgb expects (C, H, W), got {x.shape}")

    if x.shape[0] == 1:
        x = np.repeat(x, 3, axis=0)
    elif x.shape[0] != 3:
        x = x[:3]

    if mode == "robust":
        s = max(float(np.percentile(np.abs(x), p)), eps)
    elif mode == "maxabs":
        s = max(float(np.max(np.abs(x))), eps)
    else:
        raise ValueError(f"Unknown mode={mode!r}. Use 'robust' or 'maxabs'.")

    x_scaled = np.clip(x / s, -1.0, 1.0)
    disp = float(center) + 0.5 * x_scaled

    if gamma != 1.0:
        disp = np.clip(disp, 0.0, 1.0) ** (1.0 / float(gamma))
    if clip:
        disp = np.clip(disp, 0.0, 1.0)

    return np.transpose(disp, (1, 2, 0)).astype(np.float32, copy=False)


def compute_spatial_sta(
    wrapper,
    *,
    output_path: str | Path | None = None,
    # --- stimulus / prediction parameters ---
    key: str = "37_3843837605846_0_V3A_V4",
    num_samples: int = 12,
    dot_offset_samples: int = 3,
    dot_duration_samples: int = 6,
    fps: float = 30.0,
    square_size_px: int = 25,
    dots_per_frame: int = 100,
    base_seed: int = 61,
    N: int = 5000,
    batch_size: int = 10,
    win_start: int = 2,
    win_dur: int = 6,
    ds_factor: int = 4,
    dtype=np.float32,
    # --- STA parameters ---
    t_frame: int = 5,
    skip_frames: int = 0,
    response_offset: int = 0,
    response_duration: int = 6,
    baseline: float = 0.5,
    normalize_weights: bool = True,
    subtract_stim_mean: bool = True,
    zscore_normalize: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """Run the full dotmapping pipeline and return (and optionally save) the spatial STA.

    Generates sparse-noise stimuli, collects model predictions, and computes a
    spatial spike-triggered average (STA) for every neuron.  The result is
    optionally written to disk as a ``.npy`` file so it can be reloaded without
    re-running the expensive prediction step.

    Parameters
    ----------
    wrapper:
        Wrapped model for generating predictions.
    output_path:
        If provided, the STA array is saved to this path as a ``.npy`` file.
        Parent directories are created automatically.
    key:
        Session key passed to the wrapper.
    num_samples:
        Number of time samples per stimulus.
    dot_offset_samples:
        Dot onset offset in samples.
    dot_duration_samples:
        Dot duration in samples.
    fps:
        Frames per second.
    square_size_px:
        Dot size in pixels.
    dots_per_frame:
        Number of dots per frame.
    base_seed:
        Base random seed.
    N:
        Total number of stimuli to generate.
    batch_size:
        Number of stimuli per batch.
    win_start:
        Start index of the prediction response window.
    win_dur:
        Duration of the prediction response window.
    ds_factor:
        Downsampling factor for stored stimuli.
    dtype:
        Dtype for stored stimulus/prediction arrays.
    t_frame:
        Stimulus frame used for the STA.
    skip_frames:
        Frames to skip when aligning stimulus to predictions.
    response_offset:
        Offset from the aligned prediction time for the STA response window.
    response_duration:
        Duration of the STA response window.
    baseline:
        Subtracted from stimulus frames before weighting.
    normalize_weights:
        Divide weighted sum by total weight per neuron.
    subtract_stim_mean:
        Subtract ensemble mean stimulus frame before weighting.
    zscore_normalize:
        Z-score each neuron's STA across spatial dimensions.
    show_progress:
        Show tqdm progress bars.

    Returns
    -------
    sta : np.ndarray, shape (U, C, H, W)
        Spatial STA per neuron.
    """
    frames_all, pred_all, _, _ = predict_responses(
        wrapper,
        key=key,
        num_samples=num_samples,
        dot_offset_samples=dot_offset_samples,
        dot_duration_samples=dot_duration_samples,
        fps=fps,
        square_size_px=square_size_px,
        dots_per_frame=dots_per_frame,
        base_seed=base_seed,
        N=N,
        batch_size=batch_size,
        win_start=win_start,
        win_dur=win_dur,
        ds_factor=ds_factor,
        dtype=dtype,
    )

    sta, _, _ = compute_sta(
        frames_all,
        pred_all,
        t_frame=t_frame,
        skip_frames=skip_frames,
        response_offset=response_offset,
        response_duration=response_duration,
        baseline=baseline,
        normalize_weights=normalize_weights,
        subtract_stim_mean=subtract_stim_mean,
        zscore_normalize=zscore_normalize,
        show_progress=show_progress,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, sta)

    return sta
