"""FFT and Adam parity checks for the Apple Silicon diagnostic."""

import time

import torch
from torch import nn
from torch.nn import functional as F

GRADIENT_LIMIT = 1e-5
LOSS_LIMIT = 1e-5
ROUNDTRIP_LIMIT = 1e-6


class SpectralConv2d(nn.Module):
    """Fourier layer with interpolated complex weights, as in issue #151740."""

    def __init__(self, channels, modes, rng):
        super().__init__()
        self.weights = nn.Parameter(
            torch.rand(channels, channels, modes, modes, 2, generator=rng) / channels**2
        )

    def forward(self, values):
        shape = values.shape[-2:]
        values_ft = torch.fft.rfftn(values, dim=(-2, -1))
        frequency_shape = values_ft.shape[-2:]
        weights = torch.view_as_complex(self.weights)
        weights_ft = torch.complex(
            F.interpolate(weights.real, size=frequency_shape),
            F.interpolate(weights.imag, size=frequency_shape),
        )
        output_ft = torch.einsum("bixy,ioxy->boxy", values_ft, weights_ft)
        return torch.fft.irfftn(output_ft, s=shape, dim=(-2, -1))


def _training_step(model, optimizer, features, targets):
    optimizer.zero_grad(set_to_none=True)
    loss = F.mse_loss(model(features), targets)
    loss.backward()
    gradient = model.weights.grad
    optimizer.step()
    return float(loss.detach()), gradient.detach().cpu().double()


def fft_adam_parity(device="mps", *, batch=16, channels=64, grid=64, modes=12, steps=100):
    """Compare independently trained float32 and CPU float64 models each step."""
    settings = {"batch": batch, "channels": channels, "grid": grid, "modes": modes, "steps": steps}
    rng = torch.Generator().manual_seed(42119)
    inputs = torch.randn(batch, channels, grid, grid, generator=rng)
    targets = torch.randn(batch, channels, grid, grid, generator=rng)
    candidate = SpectralConv2d(channels, modes, rng)
    reference = SpectralConv2d(channels, modes, rng).double()
    reference.load_state_dict(candidate.state_dict())
    candidate = candidate.to(device=device)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=1e-3, foreach=False)
    candidate_optimizer = torch.optim.Adam(candidate.parameters(), lr=1e-3, foreach=False)
    cpu_inputs, cpu_targets = inputs.double(), targets.double()
    device_inputs, device_targets = inputs.to(device), targets.to(device)
    cpu_losses, candidate_losses, relative_gradients = [], [], []
    cpu_seconds, candidate_seconds = 0.0, 0.0
    for _ in range(steps):
        start = time.perf_counter()
        cpu_loss, cpu_gradient = _training_step(reference, reference_optimizer, cpu_inputs, cpu_targets)
        cpu_seconds += time.perf_counter() - start
        start = time.perf_counter()
        candidate_loss, candidate_gradient = _training_step(
            candidate, candidate_optimizer, device_inputs, device_targets
        )
        if device == "mps":
            torch.mps.synchronize()
        candidate_seconds += time.perf_counter() - start
        cpu_losses.append(cpu_loss)
        candidate_losses.append(candidate_loss)
        relative_gradients.append(
            float(torch.linalg.vector_norm(cpu_gradient - candidate_gradient)
                  / torch.linalg.vector_norm(cpu_gradient))
        )
    cpu_losses_tensor = torch.tensor(cpu_losses, dtype=torch.float64)
    candidate_losses_tensor = torch.tensor(candidate_losses, dtype=torch.float64)
    loss_difference = float(((cpu_losses_tensor - candidate_losses_tensor).abs()
                             / cpu_losses_tensor).max())
    max_gradient = float(torch.tensor(relative_gradients, dtype=torch.float64).max())
    passed = max_gradient < GRADIENT_LIMIT and loss_difference < LOSS_LIMIT
    return {
        "status": "passed" if passed else "failed", "device": f"{device} float32",
        "reference": "cpu float64", "fixture": settings,
        "max_relative_gradient_l2_error": max_gradient, "gradient_limit": GRADIENT_LIMIT,
        "worst_gradient_step": relative_gradients.index(max_gradient) + 1,
        "max_relative_loss_error": loss_difference, "loss_limit": LOSS_LIMIT,
        "relative_gradient_by_step": relative_gradients,
        "candidate_losses": candidate_losses, "cpu_float64_losses": cpu_losses,
        "elapsed_seconds": {"cpu_float64": cpu_seconds, f"{device}_float32": candidate_seconds},
    }


def odd_fft_roundtrip(device="mps"):
    """Check odd spatial extents with an explicit inverse FFT shape."""
    rng = torch.Generator().manual_seed(184)
    cases = []
    for shape in ((31,), (63, 65), (33, 64), (65, 33)):
        values = torch.randn(2, *shape, generator=rng).to(device)
        dims = tuple(range(-len(shape), 0))
        reconstructed = torch.fft.irfftn(torch.fft.rfftn(values, dim=dims), s=shape, dim=dims)
        err = float((values - reconstructed).abs().max())
        cases.append({"shape": list(shape), "max_absolute_error": err})
    passed = all(case["max_absolute_error"] < ROUNDTRIP_LIMIT for case in cases)
    return {
        "status": "passed" if passed else "failed", "device": f"{device} float32",
        "absolute_error_limit": ROUNDTRIP_LIMIT, "cases": cases,
    }
