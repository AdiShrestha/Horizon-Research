"""Deterministic, CPU float64 P2 viscous Burgers trajectories."""

from dataclasses import dataclass

import exponax as ex
import jax
import jax.numpy as jnp
import numpy as np

# Domain-separate pilot keys from the study's fold_in(master_key, index).
PILOT_SEED_TAG = 0x50325031


@dataclass(frozen=True)
class BurgersPilot:
    namespace: str
    master_seed: int
    grid: int
    solver_grid: int
    substeps: int
    steps: int
    domain_extent: float
    macro_dt: float
    nu_low: float
    nu_high: float
    slope_mean: float
    slope_std: float
    max_mode: int
    amplitude_low: float
    amplitude_high: float

    def __post_init__(self):
        if self.namespace != "engineering_pilot":
            raise ValueError("this generator is scoped to the engineering pilot namespace")
        if self.grid < 4 or self.grid % 2 or self.solver_grid != 2 * self.grid:
            raise ValueError("grid must be even and solver_grid must be twice grid")
        if not 1 <= self.max_mode < self.grid // 2:
            raise ValueError("initial Fourier modes must lie below the projected Nyquist")
        if self.steps < 1 or self.substeps < 1 or min(self.domain_extent, self.macro_dt) <= 0:
            raise ValueError("time and domain settings must be positive")
        if not 0 < self.nu_low < self.nu_high or not 0 < self.amplitude_low < self.amplitude_high:
            raise ValueError("viscosity and amplitude bounds must be positive ordered intervals")
        if self.slope_std < 0 or not 0 <= self.master_seed < 2**32:
            raise ValueError("invalid slope or uint32 seed")


def project_fourier(spectrum, source_grid, grid):
    """Project a NumPy-convention real FFT to an even grid, dropping its Nyquist."""
    spectrum = spectrum[..., : grid // 2 + 1].at[..., -1].set(0)
    return jnp.fft.irfft(spectrum / source_grid, n=grid, norm="forward")


def batched_burgers(config, *, spatial_refinement=1, temporal_refinement=1):
    """Return a jitted vmap over uint32 trajectory IDs on the configured CPU."""
    if not jax.config.x64_enabled or jax.default_backend() != "cpu":
        raise RuntimeError("set JAX_PLATFORMS=cpu and JAX_ENABLE_X64=true before importing JAX")
    if spatial_refinement < 1 or temporal_refinement < 1:
        raise ValueError("refinement factors must be positive")
    points = config.solver_grid * spatial_refinement
    substeps = config.substeps * temporal_refinement
    pilot_key = jax.random.fold_in(jax.random.PRNGKey(config.master_seed), PILOT_SEED_TAG)
    modes = jnp.arange(1, config.max_mode + 1, dtype=jnp.float64)
    phase = 2 * jnp.pi * modes[:, None] * jnp.arange(points, dtype=jnp.float64) / points
    basis = jnp.concatenate((jnp.cos(phase), jnp.sin(phase)), axis=0)

    def solve(trajectory_id):
        key = jax.random.fold_in(pilot_key, trajectory_id)
        nu_key, slope_key, amplitude_key, coefficients_key = jax.random.split(key, 4)
        nu = jnp.exp(jax.random.uniform(
            nu_key, (), dtype=jnp.float64, minval=np.log(config.nu_low), maxval=np.log(config.nu_high)
        ))
        slope = config.slope_mean + config.slope_std * jax.random.normal(
            slope_key, (), dtype=jnp.float64
        )
        amplitude = jax.random.uniform(
            amplitude_key, (), dtype=jnp.float64,
            minval=config.amplitude_low, maxval=config.amplitude_high,
        )
        coefficients = jax.random.normal(coefficients_key, (2, modes.size), dtype=jnp.float64)
        coefficients *= modes ** (-0.5 * slope)
        # Parseval gives the same normalizer on both solver grids.
        rms = jnp.sqrt(jnp.sum(coefficients**2) / 2)
        initial = (amplitude / rms) * (coefficients.ravel() @ basis)
        stepper = ex.stepper.Burgers(
            1, config.domain_extent, points, config.macro_dt / substeps,
            diffusivity=nu, conservative=True, order=4, dealiasing_fraction=2 / 3,
        )

        def macro_step(state, _):
            state = jax.lax.fori_loop(0, substeps, lambda _, u: stepper.step_fourier(u), state)
            return state, project_fourier(state, points, config.grid)

        initial_ft = jnp.fft.rfft(initial[None, :], axis=-1)
        _, future = jax.lax.scan(macro_step, initial_ft, None, length=config.steps)
        snapshots = jnp.concatenate((project_fourier(initial_ft, points, config.grid)[None], future))
        return snapshots, (nu, slope, amplitude, key)

    return jax.jit(jax.vmap(solve))
