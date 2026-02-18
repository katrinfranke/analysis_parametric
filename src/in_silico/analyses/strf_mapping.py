"""Spatiotemporal receptive field (stRF) mapping via reverse correlation.

Stimuli are block-based independent-frame noise (white or pink).  Each frame
in the 12-frame batch carries a *different* random pattern, giving the model
sufficient temporal variation to estimate a spatiotemporal RF.

The result is a 3-D RF array with shape ``(U, n_lags, C, H, W)``:
  - ``U``      – number of neurons
  - ``n_lags`` – temporal lags (lag 0 = simultaneous; lag k = stimulus
                 k frames before the response, i.e. longer neural latency)
  - ``C``      – colour channels (3 for RGB)
  - ``H, W``   – spatial dimensions of the (down-sampled) stimulus
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from in_silico.analyses.dotmapping import downsample_avg_pool_nchw, normalize_input
from in_silico.stimuli.white_noise import WhiteNoiseSpec, make_white_noise


# ---------------------------------------------------------------------------
# Core accumulation
# ---------------------------------------------------------------------------

def compute_strf_batch_sum(
    frames_btchw: np.ndarray,
    pred_but: np.ndarray,
    lags: tuple[int, ...] = (0, 1, 2, 3),
    skip_frames: int = 0,
    center_stim: bool = True,
    center_resp: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate the stRF reverse-correlation numerator across a batch.

    For each lag ``τ`` the accumulated quantity is::

        strf_sum[u, j] += Σ_{b,t}  resp[b, u, t+τ] · stim[b, t, :, :, :]

    Larger lags correspond to a stimulus that occurred further in the past
    relative to the response, i.e. longer neural latencies.

    Parameters
    ----------
    frames_btchw:
        Stimulus frames, shape ``(B, T, C, H, W)``, values in [0, 1] (or
        baseline-subtracted contrast values).
    pred_but:
        Neural predictions, shape ``(B, U, T_pred)``.
    lags:
        Non-negative integer lags to accumulate.
    skip_frames:
        Number of leading stimulus frames that have no corresponding prediction
        (``= num_frames − T_pred``; e.g. 3 for a 12-frame batch with 9
        predicted frames).
    center_stim:
        Subtract the per-stimulus temporal mean at each pixel before
        correlating (removes the DC offset from the noise pattern).
    center_resp:
        Subtract the per-stimulus per-unit temporal mean before correlating
        (removes mean firing rate from each trial).

    Returns
    -------
    strf_sum : np.ndarray, shape ``(U, L, C, H, W)``
        Accumulated response-weighted stimulus sum.  Divide by ``n_sum`` at
        the end to obtain the normalised stRF.
    n_sum : np.ndarray, shape ``(L,)``
        Number of ``(batch, time)`` pairs contributing per lag.
    """
    assert frames_btchw.ndim == 5, "frames must be (B, T, C, H, W)"
    assert pred_but.ndim == 3, "pred must be (B, U, T_pred)"

    B, T_stim, C, H, W = frames_btchw.shape
    B2, U, T_pred = pred_but.shape
    assert B == B2, f"Batch size mismatch: frames B={B}, pred B={B2}"

    # Align stimulus frames to the prediction time axis
    stim = frames_btchw[:, skip_frames: skip_frames + T_pred]  # (B, T_use, C, H, W)
    T_use = min(stim.shape[1], T_pred)
    stim = stim[:, :T_use]                                     # (B, T_use, C, H, W)
    resp = pred_but[:, :, :T_use]                              # (B, U, T_use)

    if center_stim:
        stim = stim - stim.mean(axis=1, keepdims=True)
    if center_resp:
        resp = resp - resp.mean(axis=2, keepdims=True)

    L = len(lags)
    strf_sum = np.zeros((U, L, C, H, W), dtype=np.float64)
    n_sum = np.zeros((L,), dtype=np.int64)

    for j, lag in enumerate(lags):
        if lag < 0:
            raise ValueError("lags must be non-negative integers")
        if lag >= T_use:
            continue

        # stim at t=0..T_use-lag-1, resp at t=lag..T_use-1
        T_lag = T_use - lag
        stim_l = stim[:, :T_lag]                     # (B, T_lag, C, H, W)
        resp_l = resp[:, :, lag: lag + T_lag]        # (B, U, T_lag)

        # Efficient matrix multiply instead of einsum:
        #   A = resp_l reshaped to (U, B*T_lag)
        #   Bm = stim_l reshaped to (B*T_lag, C*H*W)
        #   result = A @ Bm → (U, C*H*W) → (U, C, H, W)
        A = resp_l.transpose(1, 0, 2).reshape(U, B * T_lag)       # (U, B*T_lag)
        Bm = stim_l.reshape(B * T_lag, C * H * W)                 # (B*T_lag, C*H*W)
        strf_sum[:, j] += (A @ Bm).reshape(U, C, H, W)

        n_sum[j] += B * T_lag

    return strf_sum.astype(np.float32), n_sum


# ---------------------------------------------------------------------------
# Prediction loop
# ---------------------------------------------------------------------------

def predict_responses_strf(
    wrapper,
    *,
    key: str = "37_3843837605846_0_V3A_V4",
    spec: WhiteNoiseSpec,
    N: int = 1000,
    batch_size: int = 10,
    ds_factor: int = 4,
    dtype=np.float32,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Generate spatiotemporal noise stimuli, predict responses, and collect arrays.

    Parameters
    ----------
    wrapper:
        Wrapped model (``ModelWrapper``) for generating predictions.
    key:
        Session key passed to the wrapper.
    spec:
        Base ``WhiteNoiseSpec``.  The ``seed`` field is overridden per stimulus
        as ``spec.seed + stimulus_index`` to ensure each stimulus is unique.
    N:
        Total number of stimuli to generate.
    batch_size:
        Number of stimuli to accumulate before a model forward pass.
    ds_factor:
        Spatial downsampling factor applied to stored stimulus frames
        (does *not* affect the frames passed to the model).  Use ``1`` for
        full resolution.
    dtype:
        Output dtype for stored arrays (default float32).

    Returns
    -------
    frames_all : np.ndarray, shape ``(N, T, 3, H_ds, W_ds)``
        Down-sampled stimulus frames in [0, 1].
    pred_all : np.ndarray, shape ``(N, U, T_pred)``
        Baseline-shifted neural predictions (first time-step subtracted).
    seeds : list[int]
        Random seeds used for each stimulus, length N.
    """
    wrapper.key = key

    frames_batches: list[np.ndarray] = []
    pred_batches: list[np.ndarray] = []
    seeds: list[int] = []

    with tqdm(total=N, desc="Predicting stRF responses", unit="stim") as pbar:
        for start in range(0, N, batch_size):
            b = min(batch_size, N - start)
            frames_list: list[np.ndarray] = []
            preds_list: list[np.ndarray] = []

            for i in range(b):
                seed_i = spec.seed + start + i
                spec_i = replace(spec, seed=seed_i)

                frames_raw = make_white_noise(spec_i)         # (T, 3, H, W) in [0, 1]
                frames_store = downsample_avg_pool_nchw(
                    frames_raw.astype(np.float32, copy=False), ds_factor
                )                                             # (T, 3, H_ds, W_ds)
                frames_norm = normalize_input(frames_raw)     # (T, 3, H, W) normalised
                pred_ut, _ = wrapper.predict(frames_norm)     # (U, T_pred)

                frames_list.append(frames_store.astype(dtype, copy=False))
                preds_list.append(pred_ut.astype(dtype, copy=False))
                seeds.append(seed_i)
                pbar.update(1)

            frames_btchw = np.stack(frames_list, axis=0)     # (b, T, 3, H_ds, W_ds)
            pred_but = np.stack(preds_list, axis=0)          # (b, U, T_pred)

            # Baseline-shift: subtract the first prediction time-step
            pred_but = pred_but - pred_but[:, :, [0]]

            frames_batches.append(frames_btchw)
            pred_batches.append(pred_but.astype(dtype, copy=False))

    frames_all = np.concatenate(frames_batches, axis=0)      # (N, T, 3, H_ds, W_ds)
    pred_all = np.concatenate(pred_batches, axis=0)          # (N, U, T_pred)

    return frames_all, pred_all, seeds


# ---------------------------------------------------------------------------
# stRF computation
# ---------------------------------------------------------------------------

def compute_spatiotemporal_sta(
    frames_all: np.ndarray,
    pred_all: np.ndarray,
    *,
    skip_frames: int = 3,
    n_lags: int = 4,
    center_stim: bool = True,
    center_resp: bool = True,
    baseline: float = 0.5,
    inner_batch: int = 500,
    show_progress: bool = True,
) -> np.ndarray:
    """Compute the spatiotemporal spike-triggered average (stRF) for each neuron.

    Iterates over stored stimuli in chunks, accumulates the reverse-correlation
    numerator via :func:`compute_strf_batch_sum`, and normalises at the end.

    Parameters
    ----------
    frames_all:
        Stored stimulus frames, shape ``(N, T, C, H, W)``, values in [0, 1].
    pred_all:
        Neural predictions, shape ``(N, U, T_pred)``.
    skip_frames:
        Leading stimulus frames with no corresponding prediction
        (= ``num_frames − T_pred``, typically 3 for 12-frame / 9-pred batches).
    n_lags:
        Number of lags to compute (lags ``0, 1, …, n_lags−1``).
        With ``skip_frames=3`` and ``T_pred=9``, lags ``0–3`` each receive the
        same number of contributing samples.
    center_stim:
        Subtract the per-stimulus temporal mean at each pixel before correlating.
    center_resp:
        Subtract the per-stimulus per-unit temporal mean before correlating.
    baseline:
        Subtracted from stored frames before cross-correlation
        (0.5 centres binary [0, 1] frames at grey).
    inner_batch:
        Number of stimuli processed per accumulation step (memory trade-off).
    show_progress:
        Show a tqdm progress bar.

    Returns
    -------
    strf : np.ndarray, shape ``(U, n_lags, C, H, W)``
        Normalised spatiotemporal RF per neuron.
        Positive values indicate ON-preference, negative OFF-preference,
        relative to the grey baseline.
    """
    N, T_stim, C, H, W = frames_all.shape
    N2, U, T_pred = pred_all.shape
    assert N == N2, f"frames_all and pred_all N mismatch: {N} vs {N2}"

    lags = tuple(range(n_lags))
    strf_acc = np.zeros((U, n_lags, C, H, W), dtype=np.float64)
    n_acc = np.zeros((n_lags,), dtype=np.int64)

    # Subtract grey baseline so noise is contrast-centred around zero
    frames_c = frames_all.astype(np.float32) - float(baseline)

    n_chunks = (N + inner_batch - 1) // inner_batch
    iterator = (
        tqdm(range(0, N, inner_batch), total=n_chunks, desc="Computing stRF", unit="chunk")
        if show_progress
        else range(0, N, inner_batch)
    )
    for start in iterator:
        end = min(start + inner_batch, N)
        strf_i, n_i = compute_strf_batch_sum(
            frames_c[start:end],       # (chunk, T, C, H, W)
            pred_all[start:end],       # (chunk, U, T_pred)
            lags=lags,
            skip_frames=skip_frames,
            center_stim=center_stim,
            center_resp=center_resp,
        )
        strf_acc += strf_i
        n_acc += n_i

    # Normalise by the number of contributing (stim, time) pairs per lag
    n_safe = np.maximum(n_acc, 1).astype(np.float64)
    strf = (strf_acc / n_safe[None, :, None, None, None]).astype(np.float32)
    return strf


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def strf_to_rgb(
    strf_lchw: np.ndarray,
    *,
    mode: str = "robust",
    p: float = 99.0,
    gamma: float = 1.0,
    center: float = 0.5,
    clip: bool = True,
    shared_scale: bool = True,
    eps: float = 1e-8,
) -> list[np.ndarray]:
    """Convert a spatiotemporal RF to a displayable RGB image per lag.

    Parameters
    ----------
    strf_lchw:
        stRF for a single neuron, shape ``(n_lags, C, H, W)``, signed values.
    mode:
        ``"robust"`` uses a percentile of absolute values for scaling;
        ``"maxabs"`` uses the maximum absolute value.
    p:
        Percentile used by ``mode="robust"`` (ignored otherwise).
    gamma:
        Optional gamma correction applied after scaling.
    center:
        Display value corresponding to zero in the stRF (0.5 = grey).
    clip:
        Clip final values to [0, 1].
    shared_scale:
        If True, use a single amplitude scale across all lags so that
        magnitudes are comparable between frames.  If False, scale each
        lag independently.
    eps:
        Numerical stability constant.

    Returns
    -------
    list of np.ndarray, each shape ``(H, W, 3)``
        One displayable RGB image per lag, values in [0, 1].
    """
    x = strf_lchw.astype(np.float32, copy=False)
    if x.ndim != 4:
        raise ValueError(f"strf_to_rgb expects (n_lags, C, H, W), got {x.shape}")

    def _scale(arr: np.ndarray) -> float:
        if mode == "robust":
            return max(float(np.percentile(np.abs(arr), p)), eps)
        if mode == "maxabs":
            return max(float(np.abs(arr).max()), eps)
        raise ValueError(f"Unknown mode={mode!r}. Use 'robust' or 'maxabs'.")

    global_scale = _scale(x) if shared_scale else None

    images: list[np.ndarray] = []
    for lag_frame in x:          # lag_frame: (C, H, W)
        c = lag_frame
        if c.shape[0] == 1:
            c = np.repeat(c, 3, axis=0)
        elif c.shape[0] > 3:
            c = c[:3]

        s = global_scale if shared_scale else _scale(c)
        disp = float(center) + 0.5 * np.clip(c / s, -1.0, 1.0)

        if gamma != 1.0:
            disp = np.clip(disp, 0.0, 1.0) ** (1.0 / float(gamma))
        if clip:
            disp = np.clip(disp, 0.0, 1.0)

        images.append(np.transpose(disp, (1, 2, 0)).astype(np.float32, copy=False))

    return images


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def compute_strf(
    wrapper,
    *,
    output_path: str | Path | None = None,
    # --- stimulus / prediction ---
    key: str = "37_3843837605846_0_V3A_V4",
    num_frames: int = 12,
    fps: float = 30.0,
    block_size_px: int = 5,
    noise_type: str = "white",
    contrast: float = 1.0,
    mean_lum: float = 0.5,
    base_seed: int = 0,
    rgb: bool = True,
    N: int = 1000,
    batch_size: int = 10,
    ds_factor: int = 4,
    dtype=np.float32,
    # --- stRF estimation ---
    skip_frames: int = 3,
    n_lags: int = 4,
    center_stim: bool = True,
    center_resp: bool = True,
    baseline: float = 0.5,
    inner_batch: int = 500,
    show_progress: bool = True,
) -> np.ndarray:
    """Run the full spatiotemporal RF mapping pipeline.

    Generates block-based noise stimuli, collects neural predictions through
    the model wrapper, and computes the stRF for every neuron via reverse
    correlation.  The result is optionally saved to disk as a ``.npy`` file.

    Parameters
    ----------
    wrapper:
        Wrapped model (``ModelWrapper``) for generating predictions.
    output_path:
        If provided, the stRF array is saved to this path as a ``.npy`` file.
        Parent directories are created automatically.
    key:
        Session key passed to the wrapper.
    num_frames:
        Frames per stimulus batch (typically 12).
    fps:
        Frames per second.
    block_size_px:
        Side length (pixels) of each noise block unit (the effective spatial
        resolution of the noise; larger values → coarser noise).
    noise_type:
        ``"white"`` for independent binary noise or ``"pink"`` for 1/f
        spatially correlated noise.
    contrast:
        Peak-to-peak noise contrast in [0, 1].
    mean_lum:
        Mean luminance in [0, 1].
    base_seed:
        Starting random seed.  Stimulus ``i`` uses seed ``base_seed + i``.
    rgb:
        Three independent noise channels if True; replicated luminance if False.
    N:
        Total number of stimuli.
    batch_size:
        Stimuli per model forward-pass batch.
    ds_factor:
        Spatial downsampling factor for stored stimulus frames (1 = full res).
    dtype:
        Dtype for intermediate stored arrays.
    skip_frames:
        Leading stimulus frames with no corresponding prediction
        (= ``num_frames − T_pred``).
    n_lags:
        Number of temporal lags in the stRF (lags 0 … n_lags−1).
    center_stim:
        Subtract per-stimulus temporal mean stimulus before correlating.
    center_resp:
        Subtract per-stimulus per-unit temporal mean response before correlating.
    baseline:
        Subtracted from stored frames before cross-correlation.
    inner_batch:
        Stimuli per accumulation chunk inside :func:`compute_spatiotemporal_sta`.
    show_progress:
        Show tqdm progress bars.

    Returns
    -------
    strf : np.ndarray, shape ``(U, n_lags, C, H_ds, W_ds)``
        Spatiotemporal RF per neuron.
    """
    spec = WhiteNoiseSpec(
        num_frames=num_frames,
        fps=fps,
        block_size_px=block_size_px,
        noise_type=noise_type,
        contrast=contrast,
        mean_lum=mean_lum,
        seed=base_seed,
        rgb=rgb,
    )

    frames_all, pred_all, _ = predict_responses_strf(
        wrapper,
        key=key,
        spec=spec,
        N=N,
        batch_size=batch_size,
        ds_factor=ds_factor,
        dtype=dtype,
    )

    strf = compute_spatiotemporal_sta(
        frames_all,
        pred_all,
        skip_frames=skip_frames,
        n_lags=n_lags,
        center_stim=center_stim,
        center_resp=center_resp,
        baseline=baseline,
        inner_batch=inner_batch,
        show_progress=show_progress,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, strf)

    return strf
