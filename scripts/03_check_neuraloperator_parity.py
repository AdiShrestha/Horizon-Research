"""Check the existing 1D FNO fixture against neuraloperator 2.0.0 on CPU."""

import argparse
import json
import platform
from functools import partial
from importlib.metadata import version
from pathlib import Path

import torch
from neuralop.layers.channel_mlp import ChannelMLP
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.models import FNO
from torch import nn

from trajcp.emulators.hardware_benchmarks import Ops, _forward, _parameters, _spectral

ROOT = Path(__file__).resolve().parents[1]
WIDTH, MODES, BLOCKS = 32, 16, 4
ABSOLUTE_LIMIT, RELATIVE_LIMIT = 2e-5, 2e-6


def copy_pointwise(layer, values):
    # Local layout is (input, output); Conv1d uses (output, input, kernel).
    layer.weight.copy_(values.T.unsqueeze(-1))
    if layer.bias is not None:
        layer.bias.zero_()


def reference_fno(params):
    # neuraloperator converts the final real-FFT resolution to n // 2 + 1 bins.
    reference = FNO(
        n_modes=(2 * (MODES - 1),), n_layers=BLOCKS,
        in_channels=1, out_channels=1, hidden_channels=WIDTH,
        lifting_channel_ratio=0, positional_embedding=None, domain_padding=None,
        non_linearity=torch.relu, use_channel_mlp=False, fno_skip="linear",
        norm=None, factorization=None, implementation="reconstructed",
        conv_module=partial(SpectralConv, fft_norm="backward", bias=False),
    ).eval()
    projection = ChannelMLP(in_channels=WIDTH, out_channels=1, n_layers=1, n_dim=1)
    # The native last block omits ReLU, and the stock projection has two layers.
    reference.projection = nn.Sequential(nn.ReLU(), projection)
    copy_pointwise(reference.lifting.fcs[0], params["lift"])
    copy_pointwise(projection.fcs[0], params["readout"])
    for i, (spectral, skip) in enumerate(zip(
        reference.fno_blocks.convs, reference.fno_blocks.fno_skips, strict=True
    )):
        weights = torch.complex(params[f"real{i}_0"], params[f"imag{i}_0"])
        destination = spectral.weight.tensor
        if tuple(spectral.n_modes) != (MODES,) or destination.shape != (WIDTH, WIDTH, MODES):
            raise AssertionError("reference mode count or spectral weight layout changed")
        destination.copy_(weights.permute(1, 2, 0))
        copy_pointwise(skip.conv, params[f"pointwise{i}"])
    return reference


def input_fixture(grid):
    rng = torch.Generator(device="cpu").manual_seed(12850 + grid)
    position = torch.arange(grid, dtype=torch.float32) / grid
    harmonic = (
        0.3 + torch.sin(2 * torch.pi * position)
        + 0.2 * torch.cos(2 * torch.pi * (MODES - 1) * position)
        + 0.4 * torch.sin(2 * torch.pi * MODES * position)
    )
    impulse = torch.zeros(grid, dtype=torch.float32)
    impulse[grid // 3] = 1
    step = (position < 0.35).float() - 0.5
    return torch.stack((torch.randn(grid, generator=rng), harmonic, impulse, step))[..., None]


def errors(actual, expected):
    if actual.shape != expected.shape or not actual.isfinite().all() or not expected.isfinite().all():
        raise AssertionError("shape mismatch or nonfinite values")
    diff = (actual.double() - expected.double()).flatten(1)
    ref = expected.double().flatten(1)
    reference_norm = torch.linalg.vector_norm(ref, dim=1)
    diff_norm = torch.linalg.vector_norm(diff, dim=1)
    # Float64 reductions measure CPU float32 arithmetic without adding metric roundoff.
    relative = torch.where(
        reference_norm > 0, diff_norm / reference_norm,
        torch.where(diff_norm == 0, 0.0, float("inf")),
    )
    return {
        "max_abs": diff.abs().max().item(),
        "max_relative_l2_per_example": relative.max().item(),
    }


def compare(ops, params, reference, inputs):
    local = inputs @ params["lift"]
    lifted = reference.lifting(inputs.transpose(1, 2))
    upstream = lifted
    measurements = {"lift": errors(local, lifted.transpose(1, 2))}
    for i in range(BLOCKS):
        spectral = _spectral(ops, local, params, i, MODES)
        pointwise = local @ params[f"pointwise{i}"]
        local_channels_first = local.transpose(1, 2)
        oracle_spectral = reference.fno_blocks.convs[i](local_channels_first)
        oracle_skip = reference.fno_blocks.fno_skips[i](local_channels_first)
        measurements[f"block_{i + 1}_spectral"] = errors(spectral, oracle_spectral.transpose(1, 2))
        measurements[f"block_{i + 1}_pointwise"] = errors(pointwise, oracle_skip.transpose(1, 2))
        local = torch.relu(spectral + pointwise)
        upstream = reference.fno_blocks(upstream, index=i)
        activated = upstream if i < BLOCKS - 1 else torch.relu(upstream)
        measurements[f"block_{i + 1}_output"] = errors(local, activated.transpose(1, 2))
    measurements["readout"] = errors(
        local @ params["readout"], reference.projection(upstream).transpose(1, 2)
    )
    local_output = _forward(ops, params, inputs, "fno")
    oracle_output = 0.9 * inputs + 0.1 * reference(inputs.transpose(1, 2)).transpose(1, 2)
    measurements["full_output"] = errors(local_output, oracle_output)
    return measurements


@torch.inference_mode()
def run():
    if version("neuraloperator") != "2.0.0":
        raise RuntimeError("this reference check requires neuraloperator==2.0.0")
    torch.manual_seed(12850)
    torch.set_num_threads(4)
    ops = Ops("cpu")
    params = {k: ops.array(v) for k, v in _parameters("fno", 1, WIDTH).items()}
    reference = reference_fno(params)
    grids = (63, 64, 255, 256)
    results = {str(n): compare(ops, params, reference, input_fixture(n)) for n in grids}
    for grid, layers in results.items():
        for name, observed in layers.items():
            if observed["max_abs"] > ABSOLUTE_LIMIT or observed["max_relative_l2_per_example"] > RELATIVE_LIMIT:
                raise AssertionError(f"grid {grid}, {name}: {observed}")
    return {
        "status": "passed",
        "python": platform.python_version(),
        "versions": {k: version(k) for k in ("torch", "neuraloperator", "numpy", "tensorly", "tensorly-torch")},
        "device": "cpu", "dtype": "float32 (complex64 FFT)",
        "fixture": {
            "grids": list(grids), "examples": ["seeded Gaussian", "boundary-frequency harmonics", "impulse", "step"],
            "channels": [1, WIDTH, 1], "blocks": BLOCKS, "retained_real_fft_bins": [0, MODES - 1],
            "neuraloperator_n_modes_argument": [2 * (MODES - 1)],
            "fft_norm": "backward in both directions", "domain_padding": None,
            "spectral_padding": "zero unretained coefficients; inverse FFT is given the original grid length",
        },
        "mapped_layers": [
            "linear lift, four dense spectral convolutions, four linear pointwise skips, and linear readout",
            "ReLU after each spectral-plus-skip sum, including the final ReLU supplied by the projection adapter",
            "the local 0.9 input + 0.1 readout residual blend applied around the adapted neuraloperator model",
        ],
        "upstream_defaults_outside_comparison": [
            "stock two-layer lifting and projection MLPs, positional embedding, and channel MLPs",
            "default GELU and omission of activation after the last FNO block",
            "additional padding, normalization, or resampling configurations",
        ],
        "scope": "one-step deterministic 1D forward equivalence of the adapted architectures",
        "not_tested": ["training or gradient parity", "long rollouts", "float64 upstream", "2D or U-Net", "official cohorts"],
        "limits": {"max_abs": ABSOLUTE_LIMIT, "max_relative_l2_per_example": RELATIVE_LIMIT},
        "measurements": results,
        "reproduction_from_workspace_root": [
            "uv venv --no-project --python 3.12.8 /tmp/horizon-neuralop",
            "uv pip install --python /tmp/horizon-neuralop/bin/python -r source/requirements-neuraloperator.lock",
            "PYTHONPATH=source/src /tmp/horizon-neuralop/bin/python source/scripts/03_check_neuraloperator_parity.py "
            "--out /tmp/horizon-neuraloperator-parity-repeat.json",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/neuraloperator_1d_parity.json")
    args = parser.parse_args()
    report = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    for grid, measurements in report["measurements"].items():
        print(grid, measurements["full_output"])
    print(args.out)
