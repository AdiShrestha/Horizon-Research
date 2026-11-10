"""Viscous Burgers reference solutions: u_t + (u^2 / 2)_x = nu u_xx on a periodic domain.

Initial conditions are random Fourier series with modes 1..max_mode, coefficients N(0, 1) scaled
by k^(-slope / 2), slope ~ N(slope_mean, slope_sd^2), rescaled to a given sup norm. The series is
evaluated analytically, so every solver grid starts from the same function. Solutions use
Exponax ETDRK4 (conservative form, 2/3 dealiasing) in float64 on the CPU with a convective
substep limit, and are projected to the emulator grid by exact Fourier truncation.
"""

from __future__ import annotations

import math

import exponax as ex
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

jax.config.update("jax_enable_x64", True)

FloatArray = npt.NDArray[np.float64]
NORMALISATION_GRID = 8192


def fourier_ic(
    key: jax.Array,
    points: int,
    amplitude: float,
    max_mode: int = 8,
    slope_mean: float = 3.0,
    slope_sd: float = 0.5,
    extent: float = 1.0,
) -> tuple[FloatArray, float]:
    """Initial condition on ``points`` grid points and its spectral slope."""
    k_slope, k_coef = jax.random.split(key)
    slope = float(slope_mean + slope_sd * jax.random.normal(k_slope, dtype=jnp.float64))
    coef = np.asarray(jax.random.normal(k_coef, (2, max_mode), dtype=jnp.float64))
    modes = np.arange(1, max_mode + 1, dtype=np.float64)
    weights = modes ** (-slope / 2.0)

    def evaluate(n: int) -> FloatArray:
        x = np.arange(n, dtype=np.float64) / n * extent
        phase = 2.0 * np.pi * np.outer(x / extent, modes)
        return np.asarray((np.cos(phase) * coef[0] + np.sin(phase) * coef[1]) @ weights)

    scale = amplitude / np.abs(evaluate(NORMALISATION_GRID)).max()
    return evaluate(points) * scale, slope


def substeps_for(
    points: int, macro_dt: float, amplitude: float, courant: float, extent: float = 1.0
) -> int:
    dx = extent / points
    return max(1, math.ceil(macro_dt / (courant * dx / amplitude)))


def solve(
    u0: FloatArray,
    nu: float,
    macro_dt: float,
    steps: int,
    substeps: int,
    extent: float = 1.0,
) -> FloatArray:
    """Trajectories at macro steps 0..steps; u0 has shape (batch, points)."""
    points = u0.shape[-1]
    stepper = ex.stepper.Burgers(
        1, extent, points, macro_dt / substeps, diffusivity=nu, conservative=True, order=4
    )
    macro = ex.RepeatedStepper(stepper, substeps)
    roll = jax.jit(jax.vmap(ex.rollout(macro, steps, include_init=True)))
    out = roll(jnp.asarray(u0, dtype=jnp.float64)[:, None, :])
    return np.asarray(out[:, :, 0, :], dtype=np.float64)


def project(u: FloatArray, points: int) -> FloatArray:
    """Exact Fourier truncation to ``points`` grid points (Nyquist mode discarded)."""
    fine = u.shape[-1]
    spectrum = np.fft.rfft(u, axis=-1)[..., : points // 2]
    padded = np.zeros((*u.shape[:-1], points // 2 + 1), dtype=np.complex128)
    padded[..., : points // 2] = spectrum * (points / fine)
    return np.fft.irfft(padded, n=points, axis=-1)


def solve_many(
    u0: FloatArray,
    nus: FloatArray,
    macro_dt: float,
    steps: int,
    substeps: int,
    extent: float = 1.0,
) -> FloatArray:
    """Like ``solve`` but with one viscosity per trajectory (batched over viscosities)."""
    import equinox as eqx

    points = u0.shape[-1]

    def make(nu: jax.Array) -> ex.RepeatedStepper:
        stepper = ex.stepper.Burgers(
            1, extent, points, macro_dt / substeps, diffusivity=nu, conservative=True, order=4
        )
        return ex.RepeatedStepper(stepper, substeps)

    steppers = eqx.filter_vmap(make)(jnp.asarray(nus, dtype=jnp.float64))

    @eqx.filter_jit
    def run(batch: ex.RepeatedStepper, states: jax.Array) -> jax.Array:
        return eqx.filter_vmap(lambda s, u: ex.rollout(s, steps, include_init=True)(u))(
            batch, states
        )

    out = run(steppers, jnp.asarray(u0, dtype=jnp.float64)[:, None, :])
    return np.asarray(out[:, :, 0, :], dtype=np.float64)
