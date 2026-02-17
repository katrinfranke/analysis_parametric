from __future__ import annotations
import numpy as np
import torch
from torch import autocast


class ModelWrapper:
    """
    Wrapper for FreeViewingHiera-style models.

    Expected model call:
      out = model(video_bcthw, key, gaze=gaze_btx2, gaze_timestamps=gaze_bt)

    Where:
      video_bcthw: (B,C,T,H,W)
      gaze_btx2:   (B,T,2)
      gaze_bt:     (B,T)
      out:         (B,U,T)  (typical)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        key: str = "screen",
        skip_samples: int = 0,
        amp: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
        device: str | None = None,
        default_gaze_time: float = 3600.0,
    ):
        self.model = model
        self.key = key
        self.skip_samples = int(skip_samples)
        self.amp = bool(amp)
        self.amp_dtype = amp_dtype
        self.default_gaze_time = float(default_gaze_time)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(
        self,
        frames_tchw: np.ndarray,
        *,
        gaze_txy: np.ndarray | None = None,
        gaze_time: float | None = None,
        return_full: bool = False,
    ):
        """
        Args:
          frames_tchw: (T,C,H,W) float32 in [0,1]
          gaze_txy: (T,2) float32. If None -> zeros.
          gaze_time: constant used to fill gaze_timestamps (seconds).
                     If None -> default_gaze_time.
          return_full: if True, also return the raw out tensor (B,U,T)

        Returns:
          pred: (U, T_valid) float32  (already sliced by skip_samples)
          info: dict with keys:
              - skip_samples
              - T, T_valid
              - gaze_time
              - pred_full_shape (before skip slicing)
          (optionally) out_full: raw torch output on CPU
        """
        assert frames_tchw.ndim == 4, "frames must be (T,C,H,W)"
        T, C, H, W = frames_tchw.shape

        if gaze_txy is None:
            gaze_txy = np.zeros((T, 2), dtype=np.float32)
        assert gaze_txy.shape == (T, 2), f"gaze must be (T,2), got {gaze_txy.shape}"

        if gaze_time is None:
            gaze_time = self.default_gaze_time

        # Convert to torch + correct layout (B,C,T,H,W)
        video = torch.from_numpy(frames_tchw).to(self.device)         # (T,C,H,W)
        video = video.permute(1, 0, 2, 3).unsqueeze(0).contiguous()   # (1,C,T,H,W)

        gaze = torch.from_numpy(gaze_txy).to(self.device).unsqueeze(0)  # (1,T,2)
        gaze_ts = torch.full((1, T), float(gaze_time), device=self.device)  # (1,T)

        # Forward (matches your snippet)
        with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=(self.device.startswith("cuda") and self.amp)):
            out = self.model(video, self.key, gaze=gaze, gaze_timestamps=gaze_ts)

        # out expected: (B,U,T). Convert to numpy, slice skip_samples on time axis.
        out_cpu = out.detach().float().cpu()
        pred_full = out_cpu[0].numpy()  # (U,T)

        s = self.skip_samples
        pred = pred_full[:, s:]  # (U,T-s)

        info = {
            "skip_samples": s,
            "T": int(T),
            "T_valid": int(pred.shape[1]),
            "gaze_time": float(gaze_time),
            "pred_full_shape": tuple(pred_full.shape),
        }

        if return_full:
            return pred, info, out_cpu
        return pred, info
