"""Drifting grating analysis: SF/TF/color tuning and direction selectivity.

Two-phase pipeline
------------------
**Phase 1 – parameter sweep** (:func:`sweep_grating_params`):
  Sweep spatial frequency × temporal frequency × color axis at a small set of
  reference directions and identify the optimal stimulus parameters for each
  neuron (i.e. the (SF, TF, color) combination that maximises the mean firing
  rate, averaged over the reference directions).

**Phase 2 – direction sweep** (:func:`sweep_directions`):
  For each unique (SF, TF, color) combination that is optimal for at least one
  neuron, present drifting gratings at ``n_directions`` evenly spaced directions
  (default 12 × 30°) and record the mean response.  This gives a full direction
  tuning curve per neuron.

Derived quantities
------------------
- Preferred direction  (argmax over direction responses)
- Direction Selectivity Index (DSI) = (R_pref − R_null) / (R_pref + R_null)
- Orientation Selectivity Index (OSI) via vector-sum in orientation space

Sliding-window prediction
-------------------------
The model only accepts ``win_len`` (typically 12) frames at a time and returns
``win_len − skip_frames`` (typically 9) valid predictions.  Longer stimuli are
broken into overlapping windows with stride equal to the number of valid frames,
and all windows are predicted sequentially; the valid predictions are
concatenated along the time axis before computing the mean response.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from in_silico.analyses.dotmapping import normalize_input
from in_silico.stimuli.drifting_grating import (
    COLORS,
    DriftingGratingSpec,
    make_drifting_grating,
)


# ---------------------------------------------------------------------------
# Default sweep values
# ---------------------------------------------------------------------------

DEFAULT_SF_CPD: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
DEFAULT_TF_HZ: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
DEFAULT_COLORS: tuple[str, ...] = COLORS  # ("achromatic", "lm", "s")
DEFAULT_REF_DIRECTIONS: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
DEFAULT_N_DIRECTIONS: int = 12


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _n_total_frames(
    n_seconds: float,
    fps: float,
    win_len: int,
    skip_frames: int,
) -> tuple[int, int]:
    """Return ``(T_total, n_windows)`` needed to cover ``n_seconds``.

    Parameters
    ----------
    n_seconds:
        Desired stimulus duration in seconds.
    fps:
        Frame rate.
    win_len:
        Model input window length (frames).
    skip_frames:
        Leading frames per window that produce no valid prediction.

    Returns
    -------
    T_total : int
        Total grating frames to generate.
    n_windows : int
        Number of windows that will be passed to the model.
    """
    n_valid = win_len - skip_frames          # valid predictions per window
    n_needed = int(np.ceil(n_seconds * fps)) # total valid frames needed
    n_windows = max(1, int(np.ceil(n_needed / n_valid)))
    T_total = win_len + (n_windows - 1) * n_valid
    return T_total, n_windows


def _predict_sliding_window_mean(
    wrapper,
    frames_norm: np.ndarray,
    *,
    win_len: int,
    skip_frames: int,
    fps: float,
    response_start_s: float = 0.0,
) -> np.ndarray:
    """Predict the mean neural response to a long normalised grating sequence.

    Slides a ``win_len``-frame window over ``frames_norm`` with stride equal to
    the number of valid predictions per window (``win_len − skip_frames``).
    Valid predictions from all windows are concatenated and averaged to produce
    one scalar response per neuron.

    Parameters
    ----------
    wrapper:
        :class:`~in_silico.model.wrapper.ModelWrapper` instance.
    frames_norm:
        Normalised grating frames, shape ``(T_total, 3, H, W)``.
    win_len:
        Window length fed to the model (frames).
    skip_frames:
        Leading frames per window without valid predictions.
    fps:
        Frame rate (used to convert ``response_start_s`` to frames).
    response_start_s:
        Seconds of valid predictions to discard at the start before averaging
        (useful to skip the onset transient).

    Returns
    -------
    np.ndarray, shape ``(U,)``
        Mean response per neuron.
    """
    n_valid = win_len - skip_frames
    T_total = frames_norm.shape[0]

    preds: list[np.ndarray] = []
    t = 0
    while t + win_len <= T_total:
        window = frames_norm[t : t + win_len]      # (win_len, 3, H, W)
        pred_ut, _ = wrapper.predict(window)        # (U, n_valid)
        preds.append(pred_ut.astype(np.float32, copy=False))
        t += n_valid

    if not preds:
        raise ValueError(
            f"Not enough grating frames ({T_total}) for one window of length {win_len}."
        )

    preds_all = np.concatenate(preds, axis=1)      # (U, T_pred_total)

    # Optionally skip onset frames
    skip_pred = int(response_start_s * fps)
    preds_response = preds_all[:, skip_pred:]      # (U, T_resp)

    if preds_response.shape[1] == 0:
        raise ValueError(
            f"response_start_s={response_start_s} skips all {preds_all.shape[1]} "
            "valid prediction frames."
        )

    return preds_response.mean(axis=1)             # (U,)


# ---------------------------------------------------------------------------
# Phase 1 – parameter sweep
# ---------------------------------------------------------------------------

def sweep_grating_params(
    wrapper,
    *,
    sf_cpd_list: Sequence[float] = DEFAULT_SF_CPD,
    tf_hz_list: Sequence[float] = DEFAULT_TF_HZ,
    colors: Sequence[str] = DEFAULT_COLORS,
    ref_directions_deg: Sequence[float] = DEFAULT_REF_DIRECTIONS,
    contrast: float = 0.5,
    mean_lum: float = 0.5,
    n_seconds: float = 2.0,
    response_start_s: float = 0.0,
    key: str = "37_3843837605846_0_V3A_V4",
    size_hw: tuple[int, int] = (236, 420),
    fps: float = 30.0,
    px_per_deg: float = 6.7,
    win_len: int = 12,
    skip_frames: int = 3,
    show_progress: bool = True,
) -> np.ndarray:
    """Sweep spatial frequency × temporal frequency × color at reference directions.

    Parameters
    ----------
    wrapper:
        :class:`~in_silico.model.wrapper.ModelWrapper` instance.
    sf_cpd_list:
        Spatial frequencies to test in cycles per degree.
    tf_hz_list:
        Temporal frequencies to test in Hz.
    colors:
        Color axes to test.  Each element must be one of
        ``"achromatic"``, ``"lm"``, ``"s"``.
    ref_directions_deg:
        Directions (degrees) used during parameter search.  Responses are
        averaged over these directions to avoid direction bias.  Defaults to
        four cardinal directions.
    contrast:
        Grating contrast (modulation amplitude, [0, 1]).
    mean_lum:
        Mean luminance in [0, 1].
    n_seconds:
        Duration of each grating presentation (seconds).
    response_start_s:
        Onset period (seconds) excluded from the mean response computation.
    key:
        Session key passed to the model wrapper.
    size_hw:
        Stimulus spatial dimensions ``(H, W)`` in pixels.
    fps:
        Frame rate (frames per second).
    px_per_deg:
        Pixels per degree of visual angle.
    win_len:
        Model input window length (frames).
    skip_frames:
        Leading frames per window with no valid prediction output.
    show_progress:
        Show a tqdm progress bar.

    Returns
    -------
    responses : np.ndarray, shape ``(n_sf, n_tf, n_color, n_ref_dir, U)``
        Mean response per condition per neuron.
    """
    wrapper.key = key

    sf_cpd_list = tuple(sf_cpd_list)
    tf_hz_list = tuple(tf_hz_list)
    colors = tuple(colors)
    ref_directions_deg = tuple(ref_directions_deg)

    n_sf = len(sf_cpd_list)
    n_tf = len(tf_hz_list)
    n_col = len(colors)
    n_dir = len(ref_directions_deg)

    T_total, _ = _n_total_frames(n_seconds, fps, win_len, skip_frames)

    # Base spec (direction/sf/tf/color filled in per condition)
    base_spec = DriftingGratingSpec(
        size_hw=size_hw,
        fps=fps,
        px_per_deg=px_per_deg,
        num_frames=T_total,
        contrast=contrast,
        mean_lum=mean_lum,
    )

    total_conditions = n_sf * n_tf * n_col * n_dir
    responses: np.ndarray | None = None

    with tqdm(
        total=total_conditions,
        desc="Phase 1 – grating param sweep",
        unit="cond",
        disable=not show_progress,
    ) as pbar:
        for i_sf, sf in enumerate(sf_cpd_list):
            for i_tf, tf in enumerate(tf_hz_list):
                for i_col, color in enumerate(colors):
                    for i_dir, direction in enumerate(ref_directions_deg):
                        spec = replace(
                            base_spec,
                            direction_deg=direction,
                            sf_cpd=sf,
                            tf_hz=tf,
                            color=color,
                        )
                        frames_raw = make_drifting_grating(spec)  # (T, 3, H, W)
                        frames_norm = normalize_input(frames_raw)

                        resp = _predict_sliding_window_mean(
                            wrapper,
                            frames_norm,
                            win_len=win_len,
                            skip_frames=skip_frames,
                            fps=fps,
                            response_start_s=response_start_s,
                        )  # (U,)

                        if responses is None:
                            U = resp.shape[0]
                            responses = np.zeros(
                                (n_sf, n_tf, n_col, n_dir, U), dtype=np.float32
                            )

                        responses[i_sf, i_tf, i_col, i_dir] = resp
                        pbar.update(1)

    assert responses is not None, "No conditions were run."
    return responses


# ---------------------------------------------------------------------------
# Phase 1 – find optimal parameters per neuron
# ---------------------------------------------------------------------------

def find_optimal_params(
    param_responses: np.ndarray,
) -> np.ndarray:
    """Find the optimal (SF, TF, color) combination per neuron.

    Parameters
    ----------
    param_responses:
        Output of :func:`sweep_grating_params`, shape
        ``(n_sf, n_tf, n_color, n_ref_dir, U)``.

    Returns
    -------
    optimal_indices : np.ndarray, shape ``(U, 3)``, dtype int
        For each neuron: ``[sf_idx, tf_idx, color_idx]`` that maximises the
        mean response averaged over reference directions.
    """
    # Average over reference directions to remove direction bias
    avg = param_responses.mean(axis=3)          # (n_sf, n_tf, n_color, U)
    n_sf, n_tf, n_col, U = avg.shape

    flat = avg.reshape(-1, U)                   # (n_sf*n_tf*n_col, U)
    flat_argmax = flat.argmax(axis=0)           # (U,)

    # Convert flat index → (sf_idx, tf_idx, col_idx)
    optimal_indices = np.stack(
        np.unravel_index(flat_argmax, (n_sf, n_tf, n_col)), axis=1
    ).astype(np.int32)                          # (U, 3)
    return optimal_indices


# ---------------------------------------------------------------------------
# Phase 2 – direction sweep at optimal parameters
# ---------------------------------------------------------------------------

def sweep_directions(
    wrapper,
    optimal_indices: np.ndarray,
    *,
    sf_cpd_list: Sequence[float] = DEFAULT_SF_CPD,
    tf_hz_list: Sequence[float] = DEFAULT_TF_HZ,
    colors: Sequence[str] = DEFAULT_COLORS,
    n_directions: int = DEFAULT_N_DIRECTIONS,
    contrast: float = 0.5,
    mean_lum: float = 0.5,
    n_seconds: float = 2.0,
    response_start_s: float = 0.0,
    key: str = "37_3843837605846_0_V3A_V4",
    size_hw: tuple[int, int] = (236, 420),
    fps: float = 30.0,
    px_per_deg: float = 6.7,
    win_len: int = 12,
    skip_frames: int = 3,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep ``n_directions`` evenly spaced directions at each neuron's optimal params.

    Only the unique (SF, TF, color) combinations that are optimal for at least
    one neuron are actually run – the model predicts all neurons simultaneously,
    so each unique combo requires one direction-sweep of ``n_directions`` forward
    passes.

    Parameters
    ----------
    wrapper:
        :class:`~in_silico.model.wrapper.ModelWrapper` instance.
    optimal_indices:
        Shape ``(U, 3)`` – output of :func:`find_optimal_params`.
    sf_cpd_list, tf_hz_list, colors:
        Parameter lists used in Phase 1 (must match the same order).
    n_directions:
        Number of evenly spaced directions to test (0°…360° exclusive).
    contrast, mean_lum, n_seconds, response_start_s, key, size_hw, fps,
    px_per_deg, win_len, skip_frames, show_progress:
        Same meaning as in :func:`sweep_grating_params`.

    Returns
    -------
    direction_responses : np.ndarray, shape ``(U, n_directions)``
        Mean response per neuron per direction, at each neuron's optimal params.
    directions_deg : np.ndarray, shape ``(n_directions,)``
        Direction values in degrees (0, 30, 60, … for ``n_directions=12``).
    """
    wrapper.key = key

    sf_cpd_list = tuple(sf_cpd_list)
    tf_hz_list = tuple(tf_hz_list)
    colors = tuple(colors)

    U = optimal_indices.shape[0]
    directions_deg = np.linspace(0.0, 360.0, n_directions, endpoint=False)

    T_total, _ = _n_total_frames(n_seconds, fps, win_len, skip_frames)

    base_spec = DriftingGratingSpec(
        size_hw=size_hw,
        fps=fps,
        px_per_deg=px_per_deg,
        num_frames=T_total,
        contrast=contrast,
        mean_lum=mean_lum,
    )

    direction_responses = np.zeros((U, n_directions), dtype=np.float32)

    # Find unique (sf_idx, tf_idx, col_idx) combos needed
    unique_combos, combo_assignments = np.unique(
        optimal_indices, axis=0, return_inverse=True
    )  # unique_combos: (K, 3), combo_assignments: (U,) → index into unique_combos

    total_conditions = len(unique_combos) * n_directions
    with tqdm(
        total=total_conditions,
        desc="Phase 2 – direction sweep",
        unit="cond",
        disable=not show_progress,
    ) as pbar:
        for combo_idx, (sf_i, tf_i, col_i) in enumerate(unique_combos):
            sf = sf_cpd_list[int(sf_i)]
            tf = tf_hz_list[int(tf_i)]
            color = colors[int(col_i)]

            # Boolean mask of neurons whose optimal is this combo
            neuron_mask = combo_assignments == combo_idx  # (U,)

            for dir_idx, direction in enumerate(directions_deg):
                spec = replace(
                    base_spec,
                    direction_deg=float(direction),
                    sf_cpd=sf,
                    tf_hz=tf,
                    color=color,
                )
                frames_raw = make_drifting_grating(spec)
                frames_norm = normalize_input(frames_raw)

                resp = _predict_sliding_window_mean(
                    wrapper,
                    frames_norm,
                    win_len=win_len,
                    skip_frames=skip_frames,
                    fps=fps,
                    response_start_s=response_start_s,
                )  # (U_all,) — model predicts every neuron simultaneously

                # Assign only to neurons whose optimal is this combo
                direction_responses[neuron_mask, dir_idx] = resp[neuron_mask]
                pbar.update(1)

    return direction_responses, directions_deg


# ---------------------------------------------------------------------------
# Derived tuning metrics
# ---------------------------------------------------------------------------

def compute_dsi(
    direction_responses: np.ndarray,
    directions_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Direction Selectivity Index (DSI) per neuron.

    DSI = (R_pref − R_null) / (R_pref + R_null)

    where R_null is the response at the direction 180° opposite to the
    preferred direction.  Values range from 0 (no direction preference) to
    1 (response only to preferred direction).

    Parameters
    ----------
    direction_responses:
        Shape ``(U, n_directions)``.
    directions_deg:
        Shape ``(n_directions,)``.

    Returns
    -------
    dsi : np.ndarray, shape ``(U,)``
    preferred_dir_deg : np.ndarray, shape ``(U,)``, float
    """
    n_dir = len(directions_deg)
    step = directions_deg[1] - directions_deg[0] if n_dir > 1 else 360.0

    pref_idx = direction_responses.argmax(axis=1)  # (U,)
    r_pref = direction_responses[np.arange(len(pref_idx)), pref_idx]  # (U,)

    # Index of the anti-preferred direction (180° away, wrapping)
    null_idx = (pref_idx + n_dir // 2) % n_dir
    r_null = direction_responses[np.arange(len(null_idx)), null_idx]  # (U,)

    denom = r_pref + r_null
    dsi = np.where(denom > 0, (r_pref - r_null) / denom, 0.0).astype(np.float32)
    preferred_dir_deg = directions_deg[pref_idx].astype(np.float32)

    return dsi, preferred_dir_deg


def compute_osi(
    direction_responses: np.ndarray,
    directions_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Orientation Selectivity Index (OSI) via vector sum.

    Uses the second circular moment in orientation space (direction mod 180°):

        z = Σ_i R_i · exp(2j · θ_i)
        OSI = |z| / Σ_i R_i

    Values range from 0 (non-selective) to 1 (perfectly orientation-tuned).

    Parameters
    ----------
    direction_responses:
        Shape ``(U, n_directions)``.
    directions_deg:
        Shape ``(n_directions,)``.

    Returns
    -------
    osi : np.ndarray, shape ``(U,)``
    preferred_orientation_deg : np.ndarray, shape ``(U,)``
        Preferred orientation in [0°, 180°).
    """
    orientations_rad = np.deg2rad(directions_deg) * 2.0  # double for orientation space

    # Responses clipped to zero for the vector sum (negative spikes ignored)
    r = np.maximum(direction_responses, 0.0)  # (U, n_dir)

    # Complex vector sum
    z = (r * np.exp(1j * orientations_rad[np.newaxis, :])).sum(axis=1)  # (U,)

    total_r = r.sum(axis=1)  # (U,)
    osi = np.where(total_r > 0, np.abs(z) / total_r, 0.0).astype(np.float32)

    # Preferred orientation: angle of z / 2, wrapped to [0°, 180°)
    pref_ori_rad = np.angle(z) / 2.0
    pref_ori_deg = (np.rad2deg(pref_ori_rad) % 180.0).astype(np.float32)

    return osi, pref_ori_deg


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_polar_tuning(
    direction_responses: np.ndarray,
    directions_deg: np.ndarray,
    *,
    neuron_indices: Sequence[int] | None = None,
    titles: Sequence[str] | None = None,
    n_cols: int = 4,
    figsize_per_panel: tuple[float, float] = (3.0, 3.0),
    color: str = "steelblue",
    fill_alpha: float = 0.25,
    normalize: bool = True,
) -> plt.Figure:
    """Plot direction tuning curves as polar plots.

    Parameters
    ----------
    direction_responses:
        Shape ``(U, n_directions)`` or ``(n_directions,)`` for a single neuron.
    directions_deg:
        Direction values in degrees, shape ``(n_directions,)``.
    neuron_indices:
        Which neurons (rows) to plot.  Defaults to all.
    titles:
        Optional title for each panel.
    n_cols:
        Number of columns in the subplot grid.
    figsize_per_panel:
        Width and height of each individual polar panel (inches).
    color:
        Line / fill colour.
    fill_alpha:
        Alpha for the shaded area under the tuning curve.
    normalize:
        If True, normalise each neuron's curve to its maximum before plotting.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Handle single-neuron input
    if direction_responses.ndim == 1:
        direction_responses = direction_responses[np.newaxis, :]  # (1, n_dir)

    U, n_dir = direction_responses.shape

    if neuron_indices is None:
        neuron_indices = list(range(U))
    neuron_indices = list(neuron_indices)
    n_panels = len(neuron_indices)

    n_rows = max(1, int(np.ceil(n_panels / n_cols)))
    fig_w = figsize_per_panel[0] * min(n_cols, n_panels)
    fig_h = figsize_per_panel[1] * n_rows
    fig, axes = plt.subplots(
        n_rows, min(n_cols, n_panels),
        subplot_kw={"projection": "polar"},
        figsize=(fig_w, fig_h),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    # Close the polar curve by repeating the first point
    angles = np.deg2rad(np.append(directions_deg, directions_deg[0]))

    for panel_i, neuron_i in enumerate(neuron_indices):
        ax = axes_flat[panel_i]
        r = direction_responses[neuron_i]  # (n_dir,)
        r_plot = np.append(r, r[0])        # close the curve

        if normalize:
            r_max = r_plot.max()
            if r_max > 0:
                r_plot = r_plot / r_max

        ax.plot(angles, r_plot, color=color, linewidth=1.5)
        ax.fill(angles, r_plot, color=color, alpha=fill_alpha)

        # Mark preferred direction
        pref_i = r.argmax()
        pref_angle = np.deg2rad(directions_deg[pref_i])
        r_pref = r_plot[pref_i]
        ax.plot([pref_angle, pref_angle], [0, r_pref], color="crimson",
                linewidth=1.5, linestyle="--")

        title = (titles[panel_i] if titles else f"Neuron {neuron_i}")
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels(["0°", "90°", "180°", "270°"], fontsize=7)
        ax.tick_params(axis="y", labelsize=6)

    # Hide unused panels
    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_drifting_grating_analysis(
    wrapper,
    *,
    output_path: str | Path | None = None,
    # Model / session
    key: str = "37_3843837605846_0_V3A_V4",
    size_hw: tuple[int, int] = (236, 420),
    fps: float = 30.0,
    px_per_deg: float = 6.7,
    win_len: int = 12,
    skip_frames: int = 3,
    # Stimulus
    contrast: float = 0.5,
    mean_lum: float = 0.5,
    n_seconds: float = 2.0,
    response_start_s: float = 0.0,
    # Phase 1 – parameter sweep
    sf_cpd_list: Sequence[float] = DEFAULT_SF_CPD,
    tf_hz_list: Sequence[float] = DEFAULT_TF_HZ,
    colors: Sequence[str] = DEFAULT_COLORS,
    ref_directions_deg: Sequence[float] = DEFAULT_REF_DIRECTIONS,
    # Phase 2 – direction sweep
    n_directions: int = DEFAULT_N_DIRECTIONS,
    show_progress: bool = True,
) -> dict:
    """Run the full drifting grating analysis pipeline.

    Phase 1: sweep (SF, TF, color) at reference directions, find per-neuron
    optimal parameters.  Phase 2: sweep ``n_directions`` directions at each
    neuron's optimal (SF, TF, color).

    Parameters
    ----------
    wrapper:
        :class:`~in_silico.model.wrapper.ModelWrapper` instance.
    output_path:
        If provided, save the results dict as a ``.npz`` file.
    key:
        Session key passed to the model wrapper.
    size_hw:
        Stimulus dimensions ``(H, W)`` in pixels.
    fps:
        Frame rate (frames per second).
    px_per_deg:
        Pixels per degree of visual angle.
    win_len:
        Model input window length (frames).
    skip_frames:
        Leading frames per window with no valid predictions.
    contrast:
        Grating contrast (modulation amplitude, [0, 1]).
    mean_lum:
        Mean luminance in [0, 1].
    n_seconds:
        Duration of each grating presentation in seconds.
    response_start_s:
        Onset period to exclude from the mean response computation.
    sf_cpd_list:
        Spatial frequencies in cpd (Phase 1).
    tf_hz_list:
        Temporal frequencies in Hz (Phase 1).
    colors:
        Color axes (Phase 1).
    ref_directions_deg:
        Reference directions used in Phase 1.
    n_directions:
        Number of directions in the Phase 2 tuning sweep.
    show_progress:
        Show tqdm progress bars.

    Returns
    -------
    dict with keys:

    ``param_responses``
        np.ndarray ``(n_sf, n_tf, n_color, n_ref_dir, U)`` – Phase 1 responses.
    ``optimal_indices``
        np.ndarray ``(U, 3)`` – optimal (sf_idx, tf_idx, color_idx) per neuron.
    ``direction_responses``
        np.ndarray ``(U, n_directions)`` – Phase 2 direction tuning.
    ``directions_deg``
        np.ndarray ``(n_directions,)`` – direction values used in Phase 2.
    ``preferred_dir_deg``
        np.ndarray ``(U,)`` – preferred direction per neuron.
    ``dsi``
        np.ndarray ``(U,)`` – Direction Selectivity Index per neuron.
    ``preferred_ori_deg``
        np.ndarray ``(U,)`` – preferred orientation per neuron [0°, 180°).
    ``osi``
        np.ndarray ``(U,)`` – Orientation Selectivity Index per neuron.
    ``sf_cpd_list``, ``tf_hz_list``, ``colors``, ``ref_directions_deg``
        Parameter lists used in Phase 1.
    """
    sf_cpd_list = tuple(sf_cpd_list)
    tf_hz_list = tuple(tf_hz_list)
    colors = tuple(colors)
    ref_directions_deg = tuple(ref_directions_deg)

    common_kw = dict(
        key=key,
        size_hw=size_hw,
        fps=fps,
        px_per_deg=px_per_deg,
        win_len=win_len,
        skip_frames=skip_frames,
        contrast=contrast,
        mean_lum=mean_lum,
        n_seconds=n_seconds,
        response_start_s=response_start_s,
        show_progress=show_progress,
    )

    # ------------------------------------------------------------------
    # Phase 1: parameter sweep
    # ------------------------------------------------------------------
    param_responses = sweep_grating_params(
        wrapper,
        sf_cpd_list=sf_cpd_list,
        tf_hz_list=tf_hz_list,
        colors=colors,
        ref_directions_deg=ref_directions_deg,
        **common_kw,
    )

    optimal_indices = find_optimal_params(param_responses)

    # ------------------------------------------------------------------
    # Phase 2: direction sweep
    # ------------------------------------------------------------------
    direction_responses, directions_deg = sweep_directions(
        wrapper,
        optimal_indices,
        sf_cpd_list=sf_cpd_list,
        tf_hz_list=tf_hz_list,
        colors=colors,
        n_directions=n_directions,
        **common_kw,
    )

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------
    dsi, preferred_dir_deg = compute_dsi(direction_responses, directions_deg)
    osi, preferred_ori_deg = compute_osi(direction_responses, directions_deg)

    results = dict(
        param_responses=param_responses,
        optimal_indices=optimal_indices,
        direction_responses=direction_responses,
        directions_deg=directions_deg,
        preferred_dir_deg=preferred_dir_deg,
        dsi=dsi,
        preferred_ori_deg=preferred_ori_deg,
        osi=osi,
        sf_cpd_list=np.array(sf_cpd_list, dtype=np.float32),
        tf_hz_list=np.array(tf_hz_list, dtype=np.float32),
        colors=np.array(colors),
        ref_directions_deg=np.array(ref_directions_deg, dtype=np.float32),
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **results)

    return results
