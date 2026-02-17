import numpy as np

def reverse_corr_rf_batch_sum(
    frames_btchw: np.ndarray,          # (B,T,C,H,W)
    pred_but: np.ndarray,              # (B,U,T_pred)
    lags: list[int] | tuple[int, ...] = (0, 1, 2, 3),
    channel: int = 0,
    skip_frames: int = 0,
    center_stim: bool = True,
    center_resp: bool = True,
):
    """
    Accumulate reverse-correlation numerator+denominator across a batch.

    Returns:
      rf_sum: (U, L, H, W) sum over (b,t) of resp * stim
      n_sum:  (L,) total number of timepoints contributing per lag
             (so you can divide at the very end across all batches)
    """
    assert frames_btchw.ndim == 5, "frames must be (B,T,C,H,W)"
    assert pred_but.ndim == 3, "pred must be (B,U,T_pred)"

    B, T_stim, C, H, W = frames_btchw.shape
    B2, U, T_pred = pred_but.shape
    assert B == B2

    # pick one channel: (B,T,H,W)
    stim = frames_btchw[:, :, channel, :, :]

    # align stim to predictions: pred corresponds to stim indices [skip_frames : skip_frames+T_pred)
    stim_aligned = stim[:, skip_frames: skip_frames + T_pred, :, :]  # (B, Ta, H, W)
    Ta = stim_aligned.shape[1]
    T_use = min(Ta, T_pred)

    stim_aligned = stim_aligned[:, :T_use, :, :]  # (B,T,H,W)
    resp = pred_but[:, :, :T_use]                 # (B,U,T)

    if center_stim:
        # subtract mean over time (per stimulus) at each pixel
        stim_aligned = stim_aligned - stim_aligned.mean(axis=1, keepdims=True)  # (B,1,H,W)
    if center_resp:
        # subtract mean over time (per stimulus, per unit)
        resp = resp - resp.mean(axis=2, keepdims=True)  # (B,U,1)

    L = len(lags)
    rf_sum = np.zeros((U, L, H, W), dtype=np.float32)
    n_sum = np.zeros((L,), dtype=np.int64)

    for j, lag in enumerate(lags):
        if lag < 0:
            raise ValueError("lags must be non-negative ints")
        if lag >= T_use:
            continue

        # stimulus at time (t-lag) paired with response at time t
        stim_l = stim_aligned[:, :T_use - lag, :, :]          # (B,T-lag,H,W)
        resp_l = resp[:, :, lag:lag + (T_use - lag)]          # (B,U,T-lag)

        # sum over batch and time: (U,H,W)
        rf_sum[:, j] += np.einsum("but,bthw->uhw", resp_l, stim_l).astype(np.float32)
        n_sum[j] += B * (T_use - lag)

    return rf_sum, n_sum
