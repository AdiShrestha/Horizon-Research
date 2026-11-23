"""Kuramoto-Sivashinsky reference solutions (P3): u_t + u u_x + u_xx + nu4 u_xxxx = 0, periodic.

Exponax ETDRK4 (conservative form, 2/3 dealiasing), float64, CPU. ``solve`` maps u0 of shape
(batch, points) to macro steps 0..steps, with ``substeps`` solver steps per macro step.
``scripts/06_p3_validate.py`` checks that Exponax's sign conventions reproduce the equation above.
"""

from __future__ import annotations

import functools

import exponax as ex
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

jax.config.update("jax_enable_x64", True)

FloatArray = npt.NDArray[np.float64]


def make_stepper(extent: float, points: int, dt: float, nu4: float):
    return ex.stepper.KuramotoSivashinskyConservative(
        1,
        extent,
        points,
        dt,
        convection_scale=1.0,
        second_order_scale=1.0,
        fourth_order_scale=nu4,
        order=4,
    )


def solve(
    u0: FloatArray,
    nu4: float,
    extent: float,
    macro_dt: float,
    steps: int,
    substeps: int,
) -> FloatArray:
    """Trajectories at macro steps 0..steps; u0 has shape (batch, points)."""
    stepper = make_stepper(extent, u0.shape[-1], macro_dt / substeps, nu4)
    macro = ex.RepeatedStepper(stepper, substeps)
    roll = jax.jit(jax.vmap(ex.rollout(macro, steps, include_init=True)))
    out = roll(jnp.asarray(u0, dtype=jnp.float64)[:, None, :])
    return np.asarray(out[:, :, 0, :], dtype=np.float64)


@functools.lru_cache(maxsize=16)
def _advance_fn(extent: float, points: int, dt: float, nu4: float, steps: int):
    macro = ex.RepeatedStepper(make_stepper(extent, points, dt, nu4), steps)
    return jax.jit(jax.vmap(macro))


def advance(
    u: FloatArray, nu4: float, extent: float, dt: float, steps: int
) -> FloatArray:
    """Final state after ``steps`` solver steps of size ``dt`` (no trajectory stored)."""
    fn = _advance_fn(extent, u.shape[-1], dt, nu4, steps)
    return np.asarray(fn(jnp.asarray(u, dtype=jnp.float64)[:, None, :])[:, 0, :])


def etdrk4_reference(
    u0: FloatArray, nu4: float, extent: float, dt: float, steps: int
) -> FloatArray:
    """Independent Kassam-Trefethen ETDRK4 for the equation in the module docstring (numpy).

    Linear part L(k) = k^2 - nu4 k^4, nonlinearity -(1/2) d_x(u^2); no dealiasing mask beyond the
    2/3 rule. Returns the final state; used only to validate sign conventions."""
    n = u0.shape[-1]
    k = 2 * np.pi * np.fft.rfftfreq(n, d=extent / n)
    lin = k**2 - nu4 * k**4
    mask = (np.arange(k.size) <= n // 3).astype(float)
    g = -0.5j * k
    e, e2 = np.exp(dt * lin), np.exp(dt * lin / 2)
    m = 32
    r = np.exp(1j * np.pi * (np.arange(1, m + 1) - 0.5) / m)
    lr = dt * lin[:, None] + r[None, :]
    q = dt * np.real(np.mean((np.exp(lr / 2) - 1) / lr, axis=1))
    f1 = dt * np.real(
        np.mean((-4 - lr + np.exp(lr) * (4 - 3 * lr + lr**2)) / lr**3, axis=1)
    )
    f2 = dt * np.real(np.mean((2 + lr + np.exp(lr) * (-2 + lr)) / lr**3, axis=1))
    f3 = dt * np.real(
        np.mean((-4 - 3 * lr - lr**2 + np.exp(lr) * (4 - lr)) / lr**3, axis=1)
    )

    def nl(vh: np.ndarray) -> np.ndarray:
        u = np.fft.irfft(vh, n=n)
        return g * np.fft.rfft(u * u) * mask

    v = np.fft.rfft(u0)
    for _ in range(steps):
        nv = nl(v)
        a = e2 * v + q * nv
        na = nl(a)
        b = e2 * v + q * na
        nb = nl(b)
        c = e2 * a + q * (2 * nb - nv)
        nc = nl(c)
        v = e * v + nv * f1 + 2 * (na + nb) * f2 + nc * f3
    return np.fft.irfft(v, n=n)


def benettin_lambda(
    u_on_attractor: FloatArray,
    nu4: float,
    extent: float,
    dt: float,
    horizon: float,
    renorm: float = 1.0,
    perturbation: float = 1e-8,
    seed: int = 0,
) -> FloatArray:
    """Benettin largest Lyapunov exponent per trajectory from twin trajectories renormalized every
    ``renorm`` time units; ``u_on_attractor`` is the post-burn-in state, shape (batch, points)."""
    sub = round(renorm / dt)
    u = u_on_attractor
    d = np.random.default_rng(seed).standard_normal(u.shape)
    d *= perturbation / np.linalg.norm(d, axis=-1, keepdims=True)
    v = u + d
    total = np.zeros(u.shape[0])
    n_ren = round(horizon / renorm)
    for _ in range(n_ren):
        u = advance(u, nu4, extent, dt, sub)
        v = advance(v, nu4, extent, dt, sub)
        sep = np.linalg.norm(v - u, axis=-1)
        total += np.log(sep / perturbation)
        v = u + (v - u) * (perturbation / sep)[:, None]
    return total / (n_ren * renorm)
