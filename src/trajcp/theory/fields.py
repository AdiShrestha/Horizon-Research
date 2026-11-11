"""Synthetic space-time error fields and their sup scores (float64, NumPy).

These generators exist for simulation studies of the calibration theory. They never feed an
experimental result about an emulator.

The error field of one trajectory is e_t(x) = exp(growth * t) * z_t(x), t = 1..T, where z is
stationary with unit variance, AR(1) in time with coefficient ``rho_t``. With ``length = 0`` the
cells are independent (the AR(1) field); with ``length > 0`` the cells carry periodic
squared-exponential correlation of that length scale (a Gaussian-process field). The fixed
weights w_t = exp(weight_rate * t) deliberately differ from the true growth, so the maximising
step varies from trajectory to trajectory.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _spatial_amplitude(cells: int, length: float) -> FloatArray:
    """Square root of the circulant spectrum of a periodic squared-exponential kernel."""
    lag = np.arange(cells)
    lag = np.minimum(lag, cells - lag).astype(np.float64)
    kernel = np.exp(-(lag**2) / (2.0 * length**2))
    spectrum = np.fft.rfft(kernel).real
    return np.sqrt(np.clip(spectrum, 0.0, None))


def step_scores(
    rng: np.random.Generator,
    n: int,
    *,
    steps: int = 50,
    cells: int = 64,
    rho_t: float = 0.9,
    length: float = 0.0,
    growth: float = 0.04,
    weight_rate: float = 0.03,
    chunk: int = 2000,
) -> FloatArray:
    """Per-step maxima max_x |e_t(x)| / w_t, shape (n, steps), from the field model above."""
    if not 0.0 <= rho_t < 1.0:
        raise ValueError("rho_t must lie in [0, 1)")
    amplitude = _spatial_amplitude(cells, length) if length > 0.0 else None
    innovation = math.sqrt(1.0 - rho_t**2)
    t = np.arange(1, steps + 1, dtype=np.float64)
    ratio = np.exp((growth - weight_rate) * t)

    def noise(size: int) -> FloatArray:
        white = rng.standard_normal((size, cells))
        if amplitude is None:
            return white
        return np.asarray(
            np.fft.irfft(np.fft.rfft(white, axis=-1) * amplitude[None, :], n=cells, axis=-1),
            dtype=np.float64,
        )

    out = np.empty((n, steps), dtype=np.float64)
    for start in range(0, n, chunk):
        size = min(chunk, n - start)
        z = noise(size)
        out[start : start + size, 0] = np.abs(z).max(axis=1) * ratio[0]
        for step in range(1, steps):
            z = rho_t * z + innovation * noise(size)
            out[start : start + size, step] = np.abs(z).max(axis=1) * ratio[step]
    return out


def sup_scores(rng: np.random.Generator, n: int, **kwargs: Any) -> FloatArray:
    """n i.i.d. scores S = max_{t, x} |e_t(x)| / w_t from the field model above."""
    return np.asarray(step_scores(rng, n, **kwargs).max(axis=1), dtype=np.float64)


def uniform_scores(rng: np.random.Generator, n: int) -> FloatArray:
    """n i.i.d. Uniform(0, 1) scores: the exact score law is known."""
    return np.asarray(rng.random(n), dtype=np.float64)
