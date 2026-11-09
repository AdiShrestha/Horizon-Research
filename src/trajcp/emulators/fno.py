"""Spectral convolution layers for Fourier neural operators (eager PyTorch).

The layers follow the standard construction: transform with a real FFT, multiply the retained
low modes by learned complex weights, transform back. Every inverse FFT passes the explicit
output size ``s`` so that odd grid sizes round-trip.
"""

from __future__ import annotations

import torch
from torch import nn


def complex_dtype_for(real_dtype: torch.dtype) -> torch.dtype:
    """complex64 for float32 and complex128 for float64."""
    if real_dtype == torch.float32:
        return torch.complex64
    if real_dtype == torch.float64:
        return torch.complex128
    raise ValueError(f"unsupported real dtype {real_dtype}")


class SpectralConv2d(nn.Module):
    """Fourier layer on (batch, channels, H, W) real input with ``modes1 x modes2`` kept modes."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        shape = (in_channels, out_channels, modes1, modes2)
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(shape, dtype=torch.complex64))
        self.weights2 = nn.Parameter(scale * torch.randn(shape, dtype=torch.complex64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        spectrum = torch.fft.rfft2(x)
        out = torch.zeros(
            batch, self.out_channels, height, width // 2 + 1, dtype=spectrum.dtype, device=x.device
        )
        m1, m2 = self.modes1, self.modes2
        out[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy", spectrum[:, :, :m1, :m2], self.weights1
        )
        out[:, :, -m1:, :m2] = torch.einsum(
            "bixy,ioxy->boxy", spectrum[:, :, -m1:, :m2], self.weights2
        )
        return torch.fft.irfft2(out, s=(height, width))
