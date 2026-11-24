"""Modern U-Net for the E2 tier (brief §8.1): GroupNorm, GELU, residual blocks, circular padding,
FiLM on the task parameters in every residual block.

Input (batch, in_channels, *grid), output (batch, out_channels, *grid). Each resolution level has
two residual blocks on the way down and two on the way up; downsampling is a stride-2 circular
convolution and upsampling is nearest-neighbour followed by a circular convolution. FiLM maps the
conditioning vector to (gamma, beta) per block, h <- h * (1 + gamma) + beta, initialized at zero.
"""

from __future__ import annotations

import torch
from torch import nn

GROUPS = 8


def _conv(dim: int, cin: int, cout: int, kernel: int = 3, stride: int = 1) -> nn.Module:
    cls = nn.Conv1d if dim == 1 else nn.Conv2d
    return cls(cin, cout, kernel, stride=stride, padding=kernel // 2, padding_mode="circular")


class ResBlock(nn.Module):
    def __init__(self, dim: int, cin: int, cout: int, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(GROUPS, cin)
        self.conv1 = _conv(dim, cin, cout)
        self.norm2 = nn.GroupNorm(GROUPS, cout)
        self.conv2 = _conv(dim, cout, cout)
        self.skip = _conv(dim, cin, cout, kernel=1) if cin != cout else nn.Identity()
        self.film = nn.Linear(cond_dim, 2 * cout) if cond_dim else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        h = self.conv1(nn.functional.gelu(self.norm1(x)))
        if self.film is not None and cond is not None:
            gamma, beta = self.film(cond).chunk(2, dim=1)
            expand = (slice(None), slice(None)) + (None,) * (h.dim() - 2)
            h = h * (1.0 + gamma[expand]) + beta[expand]
        h = self.conv2(nn.functional.gelu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        channels: tuple[int, ...] = (32, 64, 128),
        blocks: int = 2,
        cond_dim: int = 0,
    ) -> None:
        super().__init__()
        if dim not in (1, 2):
            raise ValueError("dim must be 1 or 2")
        self.cond_dim = cond_dim
        self.lift = _conv(dim, in_channels, channels[0])
        self.down = nn.ModuleList()
        self.pool = nn.ModuleList()
        width = channels[0]
        skips = []
        for level, ch in enumerate(channels):
            for _ in range(blocks):
                self.down.append(ResBlock(dim, width, ch, cond_dim))
                width = ch
                skips.append(ch)
            if level < len(channels) - 1:
                self.pool.append(_conv(dim, width, width, kernel=3, stride=2))
        self.mid = nn.ModuleList(
            [ResBlock(dim, width, width, cond_dim), ResBlock(dim, width, width, cond_dim)]
        )
        self.up = nn.ModuleList()
        self.unpool = nn.ModuleList()
        for level, ch in reversed(list(enumerate(channels))):
            for _ in range(blocks):
                self.up.append(ResBlock(dim, width + skips.pop(), ch, cond_dim))
                width = ch
            if level > 0:
                self.unpool.append(_conv(dim, width, width))
        self.out_norm = nn.GroupNorm(GROUPS, width)
        self.project = _conv(dim, width, out_channels, kernel=1)
        self.blocks, self.levels = blocks, len(channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        if (cond is None) != (self.cond_dim == 0):
            raise ValueError("pass cond exactly when the model was built with cond_dim > 0")
        h = self.lift(x)
        saved = []
        it = iter(self.down)
        for level in range(self.levels):
            for _ in range(self.blocks):
                h = next(it)(h, cond)
                saved.append(h)
            if level < self.levels - 1:
                h = self.pool[level](h)
        for block in self.mid:
            h = block(h, cond)
        it_up, it_unpool = iter(self.up), iter(self.unpool)
        for level in reversed(range(self.levels)):
            for _ in range(self.blocks):
                h = next(it_up)(torch.cat([h, saved.pop()], dim=1), cond)
            if level > 0:
                h = nn.functional.interpolate(h, scale_factor=2, mode="nearest")
                h = next(it_unpool)(h)
        return self.project(nn.functional.gelu(self.out_norm(h)))
