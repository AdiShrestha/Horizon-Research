"""Apple-GPU checks: Fourier layers and FFT round trips in eager float32 against CPU float64.

The checks reproduce the setting of a reported FFT + Adam divergence on the Metal backend:
a spectral convolution (batch 16, 64 channels, 64 x 64 grid, 12 modes) trained with Adam for
100 steps. Tolerances were fixed before the first run: relative gradient error below 1e-5 at the
first step, relative loss difference below 1e-3 at every step, FFT round-trip error below 1e-6.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from trajcp.emulators.fno import SpectralConv2d

mps_required = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="no Apple GPU available"
)

BATCH, CHANNELS, GRID, MODES, STEPS = 16, 64, 64, 12, 100


def _layer_pair() -> tuple[SpectralConv2d, SpectralConv2d]:
    """The same layer in complex128 on the CPU and complex64 on the GPU."""
    generator = torch.Generator().manual_seed(0)
    reference = SpectralConv2d(CHANNELS, CHANNELS, MODES, MODES)
    shape = reference.weights1.shape
    scale = 1.0 / (CHANNELS * CHANNELS)
    with torch.no_grad():
        for weights in (reference.weights1, reference.weights2):
            real = torch.randn(shape, generator=generator, dtype=torch.float64)
            imag = torch.randn(shape, generator=generator, dtype=torch.float64)
            weights.copy_(scale * torch.complex(real, imag))
    reference.double()
    reference.weights1.data = reference.weights1.data.to(torch.complex128)
    reference.weights2.data = reference.weights2.data.to(torch.complex128)
    device_layer = SpectralConv2d(CHANNELS, CHANNELS, MODES, MODES)
    with torch.no_grad():
        device_layer.weights1.copy_(reference.weights1.to(torch.complex64))
        device_layer.weights2.copy_(reference.weights2.to(torch.complex64))
    return reference, device_layer.to("mps")


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(BATCH, CHANNELS, GRID, GRID, generator=generator, dtype=torch.float64)
    y = torch.randn(BATCH, CHANNELS, GRID, GRID, generator=generator, dtype=torch.float64)
    return x, y


def measure_training_parity() -> dict[str, float]:
    """Train both copies for STEPS Adam steps; return the parity measurements."""
    reference, device_layer = _layer_pair()
    x, y = _data()
    x_dev, y_dev = x.to("mps", torch.float32), y.to("mps", torch.float32)
    opt_ref = torch.optim.Adam(reference.parameters(), lr=1e-3)
    opt_dev = torch.optim.Adam(device_layer.parameters(), lr=1e-3)
    losses_ref: list[float] = []
    losses_dev: list[float] = []
    first_gradient_error = 0.0
    last_gradient_error = 0.0
    for step in range(STEPS):
        opt_ref.zero_grad()
        opt_dev.zero_grad()
        loss_ref = torch.mean((reference(x) - y) ** 2)
        loss_dev = torch.mean((device_layer(x_dev) - y_dev) ** 2)
        loss_ref.backward()
        loss_dev.backward()
        errors = []
        for p_ref, p_dev in zip(reference.parameters(), device_layer.parameters(), strict=True):
            g_ref = p_ref.grad
            g_dev = p_dev.grad.cpu().to(torch.complex128)
            errors.append(float(torch.linalg.norm(g_dev - g_ref) / torch.linalg.norm(g_ref)))
        if step == 0:
            first_gradient_error = max(errors)
        last_gradient_error = max(errors)
        losses_ref.append(float(loss_ref.detach()))
        losses_dev.append(float(loss_dev.detach().cpu()))
        opt_ref.step()
        opt_dev.step()
    ref, dev = np.array(losses_ref), np.array(losses_dev)
    return {
        "first_step_relative_gradient_error": first_gradient_error,
        "last_step_relative_gradient_error": last_gradient_error,
        "max_relative_loss_difference": float(np.max(np.abs(dev - ref) / ref)),
        "loss_first": float(ref[0]),
        "loss_last": float(ref[-1]),
    }


ROUND_TRIP_SHAPES: list[tuple[int, ...]] = [
    (63,),
    (65,),
    (64,),
    (63, 65),
    (31, 33),
    (64, 64),
    (127, 63),
]


def round_trip_error(shape: tuple[int, ...], device: str = "mps") -> float:
    """Relative rfftn -> irfftn(s=shape) error in float32 on ``device``."""
    generator = torch.Generator().manual_seed(2)
    x = torch.randn((4, *shape), generator=generator, dtype=torch.float32).to(device)
    dims = tuple(range(1, x.ndim))
    back = torch.fft.irfftn(torch.fft.rfftn(x, dim=dims), s=shape, dim=dims)
    return float(torch.linalg.norm((back - x).cpu()) / torch.linalg.norm(x.cpu()))


def test_spectral_layer_matches_an_explicit_dft_on_the_cpu() -> None:
    torch.manual_seed(0)
    layer = SpectralConv2d(2, 3, 3, 2).double()
    layer.weights1.data = layer.weights1.data.to(torch.complex128)
    layer.weights2.data = layer.weights2.data.to(torch.complex128)
    x = torch.randn(2, 2, 9, 7, dtype=torch.float64)
    spectrum = np.fft.rfft2(x.numpy())
    out = np.zeros((2, 3, 9, 4), dtype=np.complex128)
    w1, w2 = layer.weights1.detach().numpy(), layer.weights2.detach().numpy()
    out[:, :, :3, :2] = np.einsum("bixy,ioxy->boxy", spectrum[:, :, :3, :2], w1)
    out[:, :, -3:, :2] = np.einsum("bixy,ioxy->boxy", spectrum[:, :, -3:, :2], w2)
    expected = np.fft.irfft2(out, s=(9, 7))
    np.testing.assert_allclose(layer(x).detach().numpy(), expected, rtol=1e-12, atol=1e-14)


LARGE_PROBE = pytest.mark.xfail(
    strict=True,
    reason=(
        "the round trip is slightly above 1e-6 in float32 on the GPU at 8,001 points while float32 "
        "on the CPU stays near 4e-7; reported, tolerance unchanged"
    ),
)


@pytest.mark.mps
@mps_required
@pytest.mark.parametrize(
    "shape",
    [pytest.param(s, marks=LARGE_PROBE) if s == (127, 63) else s for s in ROUND_TRIP_SHAPES],
)
def test_fft_round_trip_with_explicit_size_on_the_gpu(shape: tuple[int, ...]) -> None:
    assert round_trip_error(shape) < 1e-6


@pytest.mark.mps
@mps_required
def test_spectral_layer_training_matches_the_cpu_reference() -> None:
    result = measure_training_parity()
    assert result["first_step_relative_gradient_error"] < 1e-5, result
    assert result["max_relative_loss_difference"] < 1e-3, result
