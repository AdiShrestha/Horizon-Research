# Ported from the parallel Horizon workspace, commit a1500f9 (1D-relevant parts only).
"""1D Fourier-layer and FFT checks: CPU float64 reference, and eager float32 on the Apple GPU.

Only the 1D parts of the parallel file are ported; the 2D spectral-layer training parity test is not.
Tests that need the GPU carry the ``mps`` marker and skip when no Apple GPU is available.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

mps_required = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="no Apple GPU available"
)

ROUND_TRIP_SHAPES_1D = [(256,), (1024,), (4096,)]


def round_trip_error(shape: tuple[int, ...], device: str = "mps") -> float:
    """Relative rfftn -> irfftn(s=shape) error in float32 on ``device``."""
    generator = torch.Generator().manual_seed(2)
    x = torch.randn((4, *shape), generator=generator, dtype=torch.float32).to(device)
    dims = tuple(range(1, x.ndim))
    back = torch.fft.irfftn(torch.fft.rfftn(x, dim=dims), s=shape, dim=dims)
    return float(torch.linalg.norm((back - x).cpu()) / torch.linalg.norm(x.cpu()))


def test_spectral_1d_layer_matches_an_explicit_dft_on_the_cpu() -> None:
    from trajcp.emulators.fno import SpectralConv1d

    torch.manual_seed(0)
    layer = SpectralConv1d(2, 3, 4)
    layer.weights.data = layer.weights.data.to(torch.complex128)
    x = torch.randn(2, 2, 11, dtype=torch.float64)
    out = np.zeros((2, 3, 6), dtype=np.complex128)
    out[:, :, :4] = np.einsum(
        "bix,iox->box", np.fft.rfft(x.numpy())[:, :, :4], layer.weights.detach().numpy()
    )
    expected = np.fft.irfft(out, n=11)
    np.testing.assert_allclose(layer(x).detach().numpy(), expected, rtol=1e-12, atol=1e-14)


def test_fno_1d_maps_the_grid_to_itself() -> None:
    from trajcp.emulators.fno import FNO

    model = FNO(1, in_channels=2, out_channels=1, width=8, modes=4, layers=2)
    out = model(torch.randn(3, 2, 256))
    assert out.shape == (3, 1, 256)
    assert torch.isfinite(out).all()


@pytest.mark.mps
@mps_required
@pytest.mark.parametrize("shape", ROUND_TRIP_SHAPES_1D)
def test_fft_round_trip_with_explicit_size_on_the_gpu(shape: tuple[int, ...]) -> None:
    assert round_trip_error(shape) < 1e-6
