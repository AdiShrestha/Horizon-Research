"""Reference-grid comparison before storage quantization."""

import numpy as np


def audit_refinement(coarse, refined):
    if coarse.shape != refined.shape or coarse.ndim != 4 or not coarse.size:
        raise ValueError("matching nonempty (trajectories, times, channels, cells) arrays required")
    if coarse.dtype != np.float64 or refined.dtype != np.float64:
        raise TypeError("audit requires the projected float64 fields")
    errors = np.abs(refined - coarse)
    if not np.isfinite(errors).all():
        raise ValueError("refinement audit encountered non-finite values")
    return errors.max(axis=0), errors.max(axis=(1, 2, 3))
