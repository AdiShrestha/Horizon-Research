"""Small reproducible FNO and U-Net fixtures for local backend measurements."""

import re
import resource
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F



class Ops:
    """Common channels-last operations with backend-specific convolution and FFT."""

    def __init__(self, device):
        self.device = device
        if device == "mlx":
            import mlx.core as mx

            self.xp = mx
        else:
            self.xp = torch

    def array(self, array):
        if self.device == "mlx":
            return self.xp.array(array)
        return torch.as_tensor(array, device=self.device)

    def concat(self, arrays, axis=-1):
        return self.xp.concatenate(arrays, axis=axis)

    def transpose(self, array, axes):
        if self.device == "mlx":
            return self.xp.transpose(array, axes)
        return array.permute(axes)

    def fft(self, array, axes):
        if self.device == "mlx":
            return self.xp.fft.rfftn(array, axes=axes)
        return torch.fft.rfftn(array, dim=axes)

    def ifft(self, array, shape, axes):
        if self.device == "mlx":
            return self.xp.fft.irfftn(array, s=shape, axes=axes)
        return torch.fft.irfftn(array, s=shape, dim=axes)

    def complex(self, real, imag):
        if self.device == "mlx":
            return real + 1j * imag
        return torch.complex(real, imag)

    def zeros(self, shape, dtype):
        if self.device == "mlx":
            return self.xp.zeros(shape, dtype=dtype)
        return torch.zeros(shape, dtype=dtype, device=self.device)

    def conv(self, array, weights, stride=1):
        dims = array.ndim - 2
        # Periodic padding, including on the downsampling convolution.
        for axis in range(1, 1 + dims):
            slices = [slice(None)] * array.ndim
            slices[axis] = slice(-1, None)
            left = array[tuple(slices)]
            slices[axis] = slice(0, 1)
            array = self.concat((left, array, array[tuple(slices)]), axis=axis)
        if self.device == "mlx":
            conv = self.xp.conv1d if dims == 1 else self.xp.conv2d
            return conv(array, weights, stride=stride)
        conv = F.conv1d if dims == 1 else F.conv2d
        channels_first = (0, dims + 1, *range(1, dims + 1))
        output = conv(array.permute(channels_first), weights.permute(channels_first), stride=stride)
        return output.permute(0, *range(2, dims + 2), 1)

    def relu(self, array):
        if self.device == "mlx":
            return self.xp.maximum(array, 0)
        return torch.relu(array)

    def double_spatial(self, array):
        for axis in range(1, array.ndim - 1):
            if self.device == "mlx":
                array = self.xp.repeat(array, 2, axis=axis)
            else:
                array = torch.repeat_interleave(array, 2, dim=axis)
        return array

    def sync(self, *arrays):
        if self.device == "mlx":
            self.xp.eval(*arrays)
            self.xp.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    def numpy(self, array):
        if self.device == "mlx":
            return np.asarray(array)
        return array.detach().cpu().numpy()


def _parameters(model, dims, width):
    rng = np.random.default_rng(47902)

    def weight(shape, fan_in):
        return rng.standard_normal(shape, dtype=np.float32) * np.float32(fan_in ** -0.5)

    if model == "fno":
        modes = 16 if dims == 1 else 12
        shape = (modes,) * dims + (width, width)
        values = {"lift": weight((1, width), 1), "readout": weight((width, 1), width)}
        for block in range(4):
            values[f"pointwise{block}"] = weight((width, width), width)
            for part in range(dims):
                values[f"real{block}_{part}"] = weight(shape, 4 * width)
                values[f"imag{block}_{part}"] = weight(shape, 4 * width)
        return values

    channels = (("enc", 1, width), ("down", width, 2 * width),
                ("bottleneck", 2 * width, 2 * width),
                ("dec", 3 * width, width), ("head", width, 1))
    return {
        name: weight((outputs, *(3,) * dims, inputs), inputs * 3**dims)
        for name, inputs, outputs in channels
    }


def _spectral(ops, array, p, block, modes):
    shape = array.shape[1:-1]
    dims = len(shape)
    axes = tuple(range(1, 1 + dims))
    spectrum = ops.fft(array, axes)
    def mix(band, part):
        real, imag = band.real, band.imag
        wr, wi = p[f"real{block}_{part}"], p[f"imag{block}_{part}"]
        # Per-frequency real GEMM shared by PyTorch and MLX.
        left = ops.concat((real, imag))
        right = ops.concat((ops.concat((wr, wi)), ops.concat((-wi, wr))), axis=-2)
        projected = ops.xp.einsum("b...i,...io->b...o", left, right)
        width = array.shape[-1]
        return ops.complex(projected[..., :width], projected[..., width:])

    def pad_half(band):
        missing = shape[-1] // 2 + 1 - modes
        if missing == 0:
            return band
        pad_shape = (*band.shape[:-2], missing, band.shape[-1])
        return ops.concat((band, ops.zeros(pad_shape, band.dtype)), axis=-2)

    if dims == 1:
        filtered = pad_half(mix(spectrum[:, :modes], 0))
    else:
        positive = pad_half(mix(spectrum[:, :modes, :modes], 0))
        negative = pad_half(mix(spectrum[:, -modes:, :modes], 1))
        pad_shape = (array.shape[0], shape[0] - 2 * modes, positive.shape[2], array.shape[-1])
        filtered = ops.concat((positive, ops.zeros(pad_shape, positive.dtype), negative), axis=1)
    return ops.ifft(filtered, shape, axes)


def _forward(ops, p, array, model):
    if model == "fno":
        state = array @ p["lift"]
        modes = 16 if array.ndim == 3 else 12
        for block in range(4):
            state = ops.relu(_spectral(ops, state, p, block, modes) + state @ p[f"pointwise{block}"])
        # Residual factor keeps the untrained 100-step fixture bounded.
        return 0.9 * array + 0.1 * (state @ p["readout"])

    skip = ops.relu(ops.conv(array, p["enc"]))
    down = ops.relu(ops.conv(skip, p["down"], stride=2))
    middle = ops.relu(ops.conv(down, p["bottleneck"]))
    up = ops.double_spatial(middle)
    decoded = ops.relu(ops.conv(ops.concat((up, skip)), p["dec"]))
    return ops.conv(decoded, p["head"])


def _relative_error(actual, reference):
    actual, reference = actual.astype(np.float64), reference.astype(np.float64)
    return float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))


def _train(model, dims, device, *, compiled=False, steps=100):
    width = 32
    batch, grid = (8, 256) if dims == 1 else (4, 64)
    rng = np.random.default_rng(11224 + dims)
    shape = (batch, *(grid,) * dims, 1)
    ops = Ops(device)
    params = {key: ops.array(value) for key, value in _parameters(model, dims, width).items()}
    x, y = (ops.array(rng.standard_normal(shape, dtype=np.float32)) for _ in range(2))

    def loss(p):
        return ((_forward(ops, p, x, model) - y) ** 2).mean()

    initial = _forward(ops, params, x, model)
    ops.sync(initial)
    initial_values = ops.numpy(initial)
    if device == "mlx":
        import mlx.optimizers as optim

        optimizer = optim.Adam(learning_rate=1e-3, eps=1e-8, bias_correction=True)
        optimizer.init(params)
        gradient = ops.xp.value_and_grad(loss)
        if compiled:
            gradient = ops.xp.compile(gradient)

        def step():
            nonlocal params
            value, grads = gradient(params)
            params = optimizer.apply_gradients(grads, params)
            ops.sync(value, params, optimizer.state)
            return float(value.item())

    else:
        for parameter in params.values():
            parameter.requires_grad_(True)
        optimizer = torch.optim.Adam(params.values(), lr=1e-3, foreach=False)

        def step():
            optimizer.zero_grad(set_to_none=True)
            value = loss(params)
            value.backward()
            optimizer.step()
            ops.sync()
            return float(value.detach())

    start = time.perf_counter()
    warmup_loss = step()
    warmup_seconds = time.perf_counter() - start
    start = time.perf_counter()
    losses = [step() for _ in range(steps)]
    elapsed = time.perf_counter() - start
    return {
        "status": "measured", "batch": batch, "grid": grid, "width": width,
        "warmup_steps": 1, "warmup_seconds": warmup_seconds, "steps": steps,
        "elapsed_seconds": elapsed, "milliseconds_per_step": 1000 * elapsed / steps,
        "losses": [warmup_loss, *losses],
    }, initial_values


def _training_benchmarks():
    for dims in (1, 2):
        for model in ("fno", "unet"):
            cases = {}
            cpu_reference = None
            for name, device, compiled in (
                ("pytorch_cpu", "cpu", False), ("pytorch_mps", "mps", False),
                ("mlx_eager", "mlx", False), ("mlx_compiled_loss_gradient", "mlx", True),
            ):
                try:
                    if device == "mlx":
                        import mlx.core as mx

                        if not mx.metal.is_available():
                            raise RuntimeError("MLX Metal is unavailable")
                        mx.set_default_device(mx.gpu)
                    result, initial = _train(model, dims, device, compiled=compiled)
                    if cpu_reference is None and name == "pytorch_cpu":
                        cpu_reference = initial, np.array(result["losses"])
                    if cpu_reference is not None:
                        result["initial_output_relative_l2_error"] = _relative_error(initial, cpu_reference[0])
                        result["max_relative_loss_curve_error"] = float(
                            (abs(np.array(result["losses"]) - cpu_reference[1]) / cpu_reference[1]).max()
                        )
                    cases[name] = result
                except (RuntimeError, ValueError, ImportError, NotImplementedError) as error:
                    cases[name] = {"status": "blocked", "error": f"{type(error).__name__}: {error}"}
                finally:
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
            yield f"training_{model}_{dims}d", cases


def cpu_1d_fallback():
    result, output = _train("fno", 1, "cpu")
    have_finite_losses = bool(np.isfinite(result["losses"]).all())
    result.update(
        device="cpu float32",
        status="passed" if have_finite_losses and result["losses"][-1] < result["losses"][0]
        and np.isfinite(output).all() else "failed",
    )
    return result


def _vm_counters():
    recorded = subprocess.check_output(["vm_stat"], text=True, timeout=5)
    return {key: int(value) for key, value in re.findall(
        r'^"?(Pageins|Pages reactivated|Translation faults)"?:\s+(\d+)\.', recorded, flags=re.M
    )}


@torch.no_grad()
def batch_ceiling():
    if not torch.backends.mps.is_available():
        return {"status": "blocked", "error": "PyTorch cannot access Metal", "ceilings": {}}
    horizon = 100
    ops = Ops("mps")
    params = {k: ops.array(v) for k, v in _parameters("fno", 2, 32).items()}
    results = {}
    for grid in (64, 128):
        records = []
        for batch in (16, 32, 64, 128, 256):
            torch.mps.empty_cache()
            before = None
            record = {"batch": batch}
            peak_driver = 0
            peak_pytorch = 0
            try:
                coordinate = torch.linspace(0, 2 * torch.pi, grid + 1, device="mps")[:-1]
                initial = coordinate.sin()[None, :] + coordinate.cos()[:, None]
                state = initial[None, :, :, None].expand(batch, -1, -1, -1).contiguous()
                state = _forward(ops, params, state, "fno")
                ops.sync()
                before = _vm_counters()
                process_faults = resource.getrusage(resource.RUSAGE_SELF).ru_majflt
                start = time.perf_counter()
                for _ in range(horizon):
                    state = _forward(ops, params, state, "fno")
                    ops.sync()
                    peak_driver = max(peak_driver, torch.mps.driver_allocated_memory())
                    peak_pytorch = max(peak_pytorch, torch.mps.current_allocated_memory())
                record.update(
                    status="completed", elapsed_seconds=time.perf_counter() - start,
                    finite=bool(torch.isfinite(state).all()), peak_sampled_driver_bytes=peak_driver,
                    peak_sampled_live_pytorch_bytes=peak_pytorch,
                    process_major_faults=resource.getrusage(resource.RUSAGE_SELF).ru_majflt - process_faults,
                )
            except (RuntimeError, OSError) as error:
                record.update(status="blocked", error=f"{type(error).__name__}: {error}")
            if before:
                after = _vm_counters()
                record["vm_stat_deltas"] = {k: after[k] - value for k, value in before.items()}
            records.append(record)
            print(f"2D FNO ceiling grid={grid} {record}", flush=True)
            state = None
        zero_pageins = [
            r["batch"] for r in records if r["status"] == "completed" and r["finite"]
            and r.get("vm_stat_deltas", {}).get("Pageins") == 0
        ]
        zero_faults = [
            r["batch"] for r in records if r["status"] == "completed" and r["finite"]
            and r.get("vm_stat_deltas", {}).get("Translation faults") == 0
        ]
        results[str(grid)] = {
            "cases": records, "largest_completed_with_zero_pageins": max(zero_pageins, default=None),
            "largest_completed_with_zero_system_translation_faults": max(zero_faults, default=None),
            "largest_completed_with_zero_process_major_faults": max([
                r["batch"] for r in records if r["status"] == "completed" and r["finite"]
                and r["process_major_faults"] == 0
            ], default=None),
        }
    return {"status": "measured", "horizon": horizon, "fno_width": 32, "grids": results,
            "vm_scope": "system-wide counters; cannot attribute page-ins or translation faults to this process"}


@torch.no_grad()
def _w3_cost():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ops = Ops(device)
    params = {k: ops.array(v) for k, v in _parameters("fno", 1, 32).items()}
    rng = np.random.default_rng(8877)
    points = np.arange(256) / 256
    modes = np.arange(1, 9)[:, None] * 2 * np.pi * points
    tangents = rng.normal(size=(8, 8)) @ np.cos(modes) + rng.normal(size=(8, 8)) @ np.sin(modes)
    tangents /= np.sqrt(np.mean(tangents**2, axis=1, keepdims=True))
    x = ops.array(np.sin(2 * np.pi * points).astype(np.float32)[None, :, None])
    directions = ops.array(tangents.astype(np.float32)[..., None]) * x.square().mean().sqrt()
    eps = 1e-3
    horizon = 100

    def rollout(state):
        outputs = []
        for _ in range(horizon):
            state = _forward(ops, params, state, "fno")
            outputs.append(state)
        return torch.stack(outputs)

    def measure(fn):
        ops.sync()
        start = time.perf_counter()
        result = fn()
        ops.sync()
        return result, time.perf_counter() - start

    _forward(ops, params, x, "fno")
    all_values, batched = measure(lambda: rollout(torch.cat((x, x + eps * directions))))
    individual, sequential = measure(lambda: torch.cat(
        [rollout(item[None]) for item in torch.cat((x, x + eps * directions))], dim=1
    ))
    fd = (all_values[:, 1:] - all_values[:, :1]) / eps
    result = {
        "status": "measured", "device": device, "horizon": horizon, "directions": len(directions),
        "batched_seconds": batched, "sequential_seconds": sequential,
        "batched_vs_sequential_max_abs_value_error": float((all_values - individual).abs().max()),
        "batched_vs_sequential_seconds_ratio": sequential / batched,
    }
    try:
        jacobian, elapsed = measure(lambda: torch.cat([
            torch.func.jvp(rollout, (x,), (direction[None],))[1] for direction in directions
        ], dim=1))
        numerator = torch.linalg.vector_norm(jacobian - fd)
        denominator = torch.linalg.vector_norm(jacobian)
        result["jvp"] = {"status": "measured", "seconds": elapsed,
                         "relative_l2_error_vs_fd": float(numerator / denominator)}
    except (NotImplementedError, RuntimeError) as error:
        result["jvp"] = {"status": "blocked", "error": f"{type(error).__name__}: {error}"}
    return result


def run_benchmarks():
    # Keep parity, speed and memory runs sequential on the shared device.
    yield from _training_benchmarks()
    try:
        yield "w3_finite_difference_and_jvp", _w3_cost()
    except (RuntimeError, NotImplementedError) as error:
        yield "w3_finite_difference_and_jvp", {"status": "blocked", "error": str(error)}
    yield "fno_batch_ceiling", batch_ceiling()
