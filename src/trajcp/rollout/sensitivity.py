"""Trajectory-intrinsic periodic finite-difference sensitivity from §5."""
from hashlib import sha256

import numpy as np
from scipy.ndimage import gaussian_filter1d


def trajectory_key(initial, parameters):
    """Stable for the frozen float32 emulator, independent of pseudonyms and batching."""
    state = np.asarray(initial, dtype="<f4").tobytes()
    params = np.asarray(parameters, dtype="<f4").tobytes()
    return sha256(state + params).digest()


def perturbed_initials(initial, parameters, *, seed, count=8, cutoff=8, epsilon=1e-3):
    """Return (1+M, batch, x), starting with the exactly unperturbed float32 states."""
    initial = np.asarray(initial, dtype=np.float32)
    parameters = np.asarray(parameters, dtype=np.float32)
    if (initial.ndim != 2 or parameters.ndim != 2 or len(parameters) != len(initial)
            or not 1 <= cutoff < initial.shape[1] / 2 or count < 1 or epsilon <= 0
            or not np.isfinite(initial).all() or not np.isfinite(parameters).all()):
        raise ValueError("invalid periodic perturbation inputs")
    x = np.arange(initial.shape[1]) * (2 * np.pi / initial.shape[1])
    phases = np.arange(1, cutoff + 1)[:, None] * x
    basis = np.concatenate((np.cos(phases), np.sin(phases)))
    out = np.broadcast_to(initial, (count + 1, *initial.shape)).copy()
    for i, (u, p) in enumerate(zip(initial, parameters, strict=True)):
        entropy = np.frombuffer(trajectory_key(u, p), dtype="<u4").tolist()
        rng = np.random.default_rng(np.random.SeedSequence([seed, *entropy]))
        xi = rng.standard_normal((count, 2 * cutoff)) @ basis
        xi /= np.sqrt(np.mean(xi**2, axis=-1, keepdims=True))
        rms = np.sqrt(np.mean(u.astype(np.float64)**2))
        out[1:, i] = u + epsilon * rms * xi
    return out


def raw_sensitivity(rollouts, *, epsilon=1e-3):
    """Rollouts are (1+M, n, T, x); accumulate RMS in CPU float64."""
    runs = np.asarray(rollouts, dtype=np.float64)
    if runs.ndim != 4 or len(runs) < 2 or epsilon <= 0 or not np.isfinite(runs).all():
        raise ValueError("invalid finite-difference rollouts")
    return np.sqrt(np.mean((runs[1:] - runs[0])**2, axis=0)) / epsilon


def intrinsic_weights(raw, smoothing, eta, absolute_floor):
    """Smooth across periodic space, apply the step-local relative floor, then add epsilon0."""
    raw = np.asarray(raw, dtype=np.float64)
    if (raw.ndim != 3 or raw.size == 0 or not np.isfinite(raw).all() or np.any(raw < 0)
            or smoothing not in (0, 1, 2, 3) or not 0 <= eta <= 1
            or not np.isfinite(absolute_floor) or absolute_floor <= 0):
        raise ValueError("invalid intrinsic weight inputs")
    smooth = gaussian_filter1d(raw, smoothing, axis=-1, mode="wrap") if smoothing else raw
    floor = eta * np.median(smooth, axis=-1, keepdims=True)
    return np.maximum(smooth, floor) + absolute_floor
