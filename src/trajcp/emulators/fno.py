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


class SpectralConv1d(nn.Module):
    """Fourier layer on (batch, channels, N) real input with ``modes`` kept modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes)
        self.weights = nn.Parameter(scale * torch.randn(shape, dtype=torch.complex64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, size = x.shape
        spectrum = torch.fft.rfft(x)
        out = torch.zeros(
            batch, self.out_channels, size // 2 + 1, dtype=spectrum.dtype, device=x.device
        )
        out[:, :, : self.modes] = torch.einsum(
            "bix,iox->box", spectrum[:, :, : self.modes], self.weights
        )
        return torch.fft.irfft(out, n=size)


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


class FNO(nn.Module):
    """Fourier neural operator in 1D or 2D: pointwise lift, ``layers`` Fourier blocks with a
    pointwise skip and GELU, pointwise projection. Input (batch, in_channels, *grid).

    With ``cond_dim > 0`` every block output is modulated by FiLM,
    h <- h * (1 + gamma(c)) + beta(c),
    from a conditioning vector c of shape (batch, cond_dim); gamma and beta start at zero."""

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        width: int,
        modes: int,
        layers: int = 4,
        projection: int = 128,
        cond_dim: int = 0,
    ) -> None:
        super().__init__()
        if dim not in (1, 2):
            raise ValueError("dim must be 1 or 2")
        conv = nn.Conv1d if dim == 1 else nn.Conv2d
        self.lift = conv(in_channels, width, 1)
        self.spectral = nn.ModuleList(
            SpectralConv1d(width, width, modes)
            if dim == 1
            else SpectralConv2d(width, width, modes, modes)
            for _ in range(layers)
        )
        self.pointwise = nn.ModuleList(conv(width, width, 1) for _ in range(layers))
        self.project = nn.Sequential(
            conv(width, projection, 1), nn.GELU(), conv(projection, out_channels, 1)
        )
        self.film = nn.ModuleList(
            nn.Linear(cond_dim, 2 * width) for _ in range(layers if cond_dim else 0)
        )
        for film in self.film:
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        if (cond is None) != (len(self.film) == 0):
            raise ValueError("pass cond exactly when the model was built with cond_dim > 0")
        h = self.lift(x)
        expand = (slice(None), slice(None)) + (None,) * (x.dim() - 2)
        last = len(self.spectral) - 1
        blocks = zip(self.spectral, self.pointwise, strict=True)
        for i, (spectral, pointwise) in enumerate(blocks):
            h = spectral(h) + pointwise(h)
            if cond is not None:
                gamma, beta = self.film[i](cond).chunk(2, dim=1)
                h = h * (1.0 + gamma[expand]) + beta[expand]
            if i < last:
                h = nn.functional.gelu(h)
        return self.project(h)
