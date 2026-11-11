"""Linear advection-diffusion-dispersion (task P1): u_t + c u_x = nu u_xx + delta u_xxx, periodic.

In Fourier space each mode evolves independently, u_hat_k(t) = u_hat_k(0) exp(m(k) t) with
m(k) = -i c k - nu k^2 - i delta k^3 (k the angular wavenumber 2 pi n / L). Two solvers:
``exact`` evaluates this multiplier on the analytic initial condition (float64, NumPy) and
``solve_exponax`` uses Exponax's general linear stepper (float64, JAX, CPU) for the audit.
"""

from __future__ import annotations

import exponax as ex
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

jax.config.update("jax_enable_x64", True)

FloatArray = npt.NDArray[np.float64]


def multiplier(points: int, c: float, nu: float, delta: float, extent: float = 1.0) -> np.ndarray:
    k = 2.0 * np.pi * np.fft.rfftfreq(points, d=extent / points)
    return -1j * c * k - nu * k**2 - 1j * delta * k**3


def exact(
    u0: FloatArray, c: float, nu: float, delta: float, dt: float, steps: int, extent: float = 1.0
) -> FloatArray:
    """States at macro steps 0..steps from u0 of shape (points,), exact in time."""
    points = u0.shape[-1]
    m = multiplier(points, c, nu, delta, extent)
    spectrum = np.fft.rfft(u0)
    times = dt * np.arange(steps + 1)
    return np.fft.irfft(spectrum[None, :] * np.exp(np.outer(times, m)), n=points, axis=-1)


def solve_exponax(
    u0: FloatArray, c: float, nu: float, delta: float, dt: float, steps: int, extent: float = 1.0
) -> FloatArray:
    """Same as ``exact`` but with Exponax: u_t = sum_j a_j d^j u, a = (0, -c, nu, delta)."""
    points = u0.shape[-1]
    stepper = ex.stepper.generic.GeneralLinearStepper(
        1, extent, points, dt, linear_coefficients=(0.0, -c, nu, delta)
    )
    out = jax.jit(ex.rollout(stepper, steps, include_init=True))(
        jnp.asarray(u0, dtype=jnp.float64)[None, :]
    )
    return np.asarray(out[:, 0, :], dtype=np.float64)
