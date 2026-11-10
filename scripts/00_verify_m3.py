"""Hardware and Apple-GPU measurements for this machine, written to a JSON file.

Sections (all by default):
  system      chip, cores, memory, macOS build, GPU and Metal support, library versions
  speed       1D and 2D FNO training-step time on the GPU and on the CPU
  ceiling     2D FNO rollout (T = 100) batch ceiling at two GPU memory-limit pairs, with
              allocated memory, page-ins and swap use before and after each rollout
  sustained   continuous 1D FNO training for a fixed number of minutes (median step per minute)
  epoch2d     2D FNO training-step time at 64 x 64, scaled to hours per training run
  w3          M perturbation rollouts at T steps, batched against sequential, and forward-mode
              Jacobian-vector products against finite differences

Every timing is a raw list of repeated measurements after warm-up, with the GPU synchronised
around each one. Inputs are random tensors: the measurements concern speed and memory, not
accuracy. GPU memory limits default to 0.75/0.65; the CPU fallback and fast-math settings must be
unset (the script refuses to run otherwise).

Usage: python scripts/00_verify_m3.py --out artifacts/hardware.json [--sections ...] [--minutes 20]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

LIMITS = {"0.75/0.65": ("0.75", "0.65"), "1.0/0.85": ("1.0", "0.85")}
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.75")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.65")
for _name in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH"):
    if os.environ.get(_name):
        sys.exit(f"{_name} must be unset for measurements")

import torch  # noqa: E402

from trajcp.emulators.fno import FNO  # noqa: E402

GPU = torch.device("mps")
CPU = torch.device("cpu")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {error}"


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def timings(fn: Callable[[], Any], device: torch.device, repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    sync(device)
    out = []
    for _ in range(repeats):
        sync(device)
        start = time.perf_counter()
        fn()
        sync(device)
        out.append(time.perf_counter() - start)
    return out


def summary(values: list[float]) -> dict[str, Any]:
    return {"raw_seconds": values, "median": statistics.median(values), "min": min(values)}


def vm_pageins() -> int:
    match = re.search(r"Pageins:\s+(\d+)", run(["vm_stat"]))
    return int(match.group(1)) if match else -1


def swap_used_mb() -> float:
    match = re.search(r"used = ([\d.]+)M", run(["sysctl", "vm.swapusage"]))
    return float(match.group(1)) if match else -1.0


def memory_state() -> dict[str, Any]:
    return {
        "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
        "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
        "pageins": vm_pageins(),
        "swap_used_mb": swap_used_mb(),
    }


def model_1d(device: torch.device) -> FNO:
    # E1 1D: 4 layers, 16 modes, width 64; input state and log-viscosity channel.
    return FNO(1, in_channels=2, out_channels=1, width=64, modes=16).to(device)


def model_2d(device: torch.device, width: int = 32) -> FNO:
    # E1 2D: 4 layers, 12 modes, width 32 to 48.
    return FNO(2, in_channels=2, out_channels=1, width=width, modes=12).to(device)


def train_step_fn(
    model: FNO, batch: int, grid: tuple[int, ...], device: torch.device
) -> Callable[[], float]:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(batch, 2, *grid, generator=generator).to(device)
    y = torch.randn(batch, 1, *grid, generator=generator).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = torch.linalg.norm(prediction - y) / torch.linalg.norm(y)
        loss.backward()
        optimizer.step()
        return 0.0

    return step


def rollout(model: FNO, u0: torch.Tensor, aux: torch.Tensor, steps: int) -> torch.Tensor:
    """Residual rollout u_{t+1} = u_t + 0.05 * model([u_t, aux]); returns the final state."""
    u = u0
    for _ in range(steps):
        u = u + 0.05 * model(torch.cat([u, aux], dim=1))
    return u


# --------------------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------------------


def section_system() -> dict[str, Any]:
    displays = run(["system_profiler", "SPDisplaysDataType"])
    hardware = run(["system_profiler", "SPHardwareDataType"])
    pick = lambda text, key: (re.search(rf"{key}:\s*(.+)", text) or [None, None])[1]  # noqa: E731
    sudo_free = subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0
    import importlib.metadata as md

    return {
        "chip": pick(hardware, "Chip"),
        "model": pick(hardware, "Model Name"),
        "cpu_cores": pick(hardware, "Total Number of Cores"),
        "memory": pick(hardware, "Memory"),
        "gpu_cores": pick(displays, "Total Number of Cores"),
        "metal_support": pick(displays, "Metal Support"),
        "metal_compiler": "not installed",
        "sw_vers": run(["sw_vers"]).strip().splitlines(),
        "kernel": platform.release(),
        "python": sys.version.split()[0],
        "versions": {p: md.version(p) for p in ("torch", "numpy", "jax", "exponax")},
        "mps_available": torch.backends.mps.is_available(),
        "mps_recommended_max_memory_bytes": int(torch.mps.recommended_max_memory()),
        "environment": {
            k: os.environ.get(k)
            for k in (
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
                "PYTORCH_MPS_LOW_WATERMARK_RATIO",
                "PYTORCH_ENABLE_MPS_FALLBACK",
                "PYTORCH_MPS_FAST_MATH",
            )
        },
        "powermetrics": "available" if sudo_free else "not available without a password",
        "thermal_state": run(["pmset", "-g", "therm"]).strip().splitlines(),
        "load_average": os.getloadavg(),
        "free_disk_bytes": os.statvfs(".").f_bavail * os.statvfs(".").f_frsize,
    }


def section_speed() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, dim, batch, grid in (("fno1d_b64", 1, 64, (256,)), ("fno2d_b16_64", 2, 16, (64, 64))):
        for device, repeats in ((GPU, 30), (CPU, 5)):
            model = model_1d(device) if dim == 1 else model_2d(device)
            fn = train_step_fn(model, batch, grid, device)
            out[f"{name}_{device.type}"] = summary(timings(fn, device, repeats, warmup=5))
    return out


def ceiling_worker(limit: str) -> None:
    """Runs in a subprocess so that the memory limits apply from the first GPU allocation."""
    result: dict[str, Any] = {"limit": limit, "runs": []}
    for grid in (64, 128):
        for batch in (16, 32, 64, 128, 256):
            torch.mps.empty_cache()
            entry: dict[str, Any] = {"grid": grid, "batch": batch, "before": memory_state()}
            try:
                model = model_2d(GPU, width=48)
                with torch.no_grad():
                    u0 = torch.randn(batch, 1, grid, grid, device=GPU)
                    aux = torch.zeros_like(u0)
                    sync(GPU)
                    start = time.perf_counter()
                    final = rollout(model, u0, aux, 100)
                    sync(GPU)
                    entry["seconds"] = time.perf_counter() - start
                    entry["finite"] = bool(torch.isfinite(final).all())
                entry["status"] = "ok"
            except RuntimeError as error:
                entry["status"] = f"error: {str(error)[:200]}"
            entry["after"] = memory_state()
            entry["pageins_delta"] = entry["after"]["pageins"] - entry["before"]["pageins"]
            swap_before = entry["before"]["swap_used_mb"]
            entry["swap_delta_mb"] = entry["after"]["swap_used_mb"] - swap_before
            result["runs"].append(entry)
            del model
            if entry["status"] != "ok":
                break
    print(json.dumps(result))


def section_ceiling() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, (high, low) in LIMITS.items():
        env = dict(os.environ)
        env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = high
        env["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = low
        proc = subprocess.run(
            [sys.executable, "-B", __file__, "--ceiling-worker", label],
            capture_output=True,
            text=True,
            env=env,
        )
        lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
        out[label] = json.loads(lines[-1]) if lines else {"error": proc.stderr[-2000:]}
    return out


def section_sustained(minutes: float) -> dict[str, Any]:
    model = model_1d(GPU)
    step = train_step_fn(model, 64, (256,), GPU)
    for _ in range(10):
        step()
    sync(GPU)
    per_minute: dict[int, list[float]] = {}
    begin = time.perf_counter()
    thermal = [run(["pmset", "-g", "therm"]).strip()]
    while (elapsed := time.perf_counter() - begin) < minutes * 60:
        start = time.perf_counter()
        step()
        sync(GPU)
        per_minute.setdefault(int(elapsed // 60), []).append(time.perf_counter() - start)
    thermal.append(run(["pmset", "-g", "therm"]).strip())
    medians = [statistics.median(per_minute[m]) for m in sorted(per_minute)]
    early = statistics.median(medians[:2])
    late_window = medians[15:20] if len(medians) >= 20 else medians[-5:]
    late = statistics.median(late_window)
    return {
        "batch": 64,
        "minutes": minutes,
        "median_step_seconds_per_minute": medians,
        "steps_per_minute": [len(per_minute[m]) for m in sorted(per_minute)],
        "ratio_late_to_early": late / early,
        "late_window": "minutes 16-20" if len(medians) >= 20 else "last 5 minutes",
        "thermal_state_before_after": thermal,
    }


def section_epoch2d() -> dict[str, Any]:
    out: dict[str, Any] = {}
    pairs_per_epoch = 1000 * 100  # D_train trajectories x one-step pairs per trajectory
    for width, batch in ((32, 16), (32, 32), (48, 16)):
        model = model_2d(GPU, width=width)
        fn = train_step_fn(model, batch, (64, 64), GPU)
        times = timings(fn, GPU, repeats=60, warmup=10)
        steps = pairs_per_epoch / batch
        epoch_hours = statistics.median(times) * steps / 3600
        out[f"width{width}_batch{batch}"] = {
            **summary(times),
            "steps_per_epoch": steps,
            "epoch_hours": epoch_hours,
            "hours_per_seed_200_epochs": epoch_hours * 200,
            "scaling": "median of 60 timed steps x steps per epoch; validation not included",
        }
    return out


def section_w3(
    steps: int = 200, members: int = 8, batch: int = 64, eps: float = 1e-3
) -> dict[str, Any]:
    torch.manual_seed(0)
    model = model_1d(GPU).eval()
    generator = torch.Generator().manual_seed(1)
    u0 = torch.randn(batch, 1, 256, generator=generator).to(GPU)
    aux = torch.full_like(u0, -6.0)
    xi = torch.randn(members, batch, 1, 256, generator=generator).to(GPU)
    rms = u0.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()

    def single() -> None:
        with torch.no_grad():
            rollout(model, u0, aux, steps)

    def batched() -> None:
        with torch.no_grad():
            states = torch.cat([u0, *(u0 + eps * rms * xi[m] for m in range(members))])
            rollout(model, states, aux.repeat(members + 1, 1, 1), steps)

    def sequential() -> None:
        with torch.no_grad():
            rollout(model, u0, aux, steps)
            for m in range(members):
                rollout(model, u0 + eps * rms * xi[m], aux, steps)

    out: dict[str, Any] = {
        "steps": steps,
        "members": members,
        "batch": batch,
        "single": summary(timings(single, GPU, 5, 1)),
        "batched": summary(timings(batched, GPU, 5, 1)),
        "sequential": summary(timings(sequential, GPU, 3, 1)),
    }
    out["batched_over_single"] = out["batched"]["median"] / out["single"]["median"]
    out["sequential_over_batched"] = out["sequential"]["median"] / out["batched"]["median"]

    jvp: dict[str, Any] = {}
    short = 50
    try:
        direction = rms * xi[0]
        f = lambda u: rollout(model, u, aux, short)  # noqa: E731
        with torch.no_grad():
            primal, tangent = torch.func.jvp(f, (u0,), (direction,))
            fd = (f(u0 + eps * direction) - primal) / eps
        error = float(torch.linalg.norm(tangent - fd) / torch.linalg.norm(tangent))
        jvp_time = timings(lambda: torch.func.jvp(f, (u0,), (direction,)), GPU, 3, 1)
        fd_time = timings(lambda: (f(u0 + eps * direction), f(u0)), GPU, 3, 1)
        jvp = {
            "runs": True,
            "steps": short,
            "relative_difference_to_finite_differences": error,
            "jvp": summary(jvp_time),
            "finite_differences": summary(fd_time),
        }
    except Exception as error:  # record whatever the backend reports
        jvp = {"runs": False, "error": f"{type(error).__name__}: {str(error)[:300]}"}
    out["jvp"] = jvp
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure this machine")
    parser.add_argument("--out", type=Path, default=Path("artifacts/hardware.json"))
    parser.add_argument(
        "--sections",
        nargs="+",
        default=["system", "speed", "w3", "epoch2d", "ceiling", "sustained"],
    )
    parser.add_argument("--minutes", type=float, default=20.0)
    parser.add_argument("--ceiling-worker", default=None)
    args = parser.parse_args()
    if args.ceiling_worker:
        ceiling_worker(args.ceiling_worker)
        return
    if not torch.backends.mps.is_available():
        sys.exit("no Apple GPU available")
    result: dict[str, Any] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if args.out.exists():
        result = json.loads(args.out.read_text()) | result
    for name in args.sections:
        print(f"[{time.strftime('%H:%M:%S')}] {name}", flush=True)
        start = time.perf_counter()
        if name == "system":
            value = section_system()
        elif name == "speed":
            value = section_speed()
        elif name == "ceiling":
            value = section_ceiling()
        elif name == "sustained":
            value = section_sustained(args.minutes)
        elif name == "epoch2d":
            value = section_epoch2d()
        elif name == "w3":
            value = section_w3()
        else:
            raise SystemExit(f"unknown section {name}")
        result[name] = value
        result.setdefault("section_wall_seconds", {})[name] = time.perf_counter() - start
        result["load_average_after_" + name] = os.getloadavg()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
    result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
