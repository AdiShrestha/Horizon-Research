"""Measure the host, run the PyTorch parity checks and benchmark local backends."""

import argparse
import json
import math
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Apply the brief's memory controls before importing PyTorch and require native MPS kernels.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.85")
os.environ.pop("PYTORCH_MPS_FAST_MATH", None)
os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)

import torch

from trajcp.emulators.mps_diagnostics import (
    GRADIENT_LIMIT,
    LOSS_LIMIT,
    ROUNDTRIP_LIMIT,
    fft_adam_parity,
    odd_fft_roundtrip,
)

ROOT = Path(__file__).resolve().parents[1]


def command(*args, timeout=30):
    try:
        result = subprocess.run(args, check=False, capture_output=True, text=True, timeout=timeout)
        return (result.stdout if result.returncode == 0 else result.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return str(error)


def host_metadata():
    profile = command("system_profiler", "SPDisplaysDataType", "SPHardwareDataType")
    wanted = ("Model Name", "Chip:", "Chipset Model", "Total Number of Cores", "Memory", "Metal")
    return {
        "architecture": platform.machine(),
        "macos": command("sw_vers"),
        "hardware": [line.strip() for line in profile.splitlines() if line.strip().startswith(wanted)],
        "unified_ram_bytes": int(ram) if (ram := command("sysctl", "-n", "hw.memsize")).isdigit() else None,
        "xcode": command("xcodebuild", "-version"),
        "command_line_tools": command("pkgutil", "--pkg-info", "com.apple.pkg.CLTools_Executables"),
        "metal_compiler": command("xcrun", "metal", "--version"),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "metal_recommended_working_set_bytes": (
            torch.mps.recommended_max_memory() if torch.backends.mps.is_available() else None
        ),
        "power": [re.sub(r"\(id=\d+\)", "", line).strip() for line in command("pmset", "-g", "batt").splitlines()],
    }


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    return value


def save(report, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    staged = out.with_suffix(".tmp")
    staged.write_text(json.dumps(_finite_json(report), indent=2, allow_nan=False) + "\n")
    staged.replace(out)


def attempt(function, **kwargs):
    try:
        return function(**kwargs)
    except (OSError, RuntimeError, NotImplementedError, ImportError) as error:
        return {"status": "blocked", "error": f"{type(error).__name__}: {error}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/hardware.json")
    parser.add_argument("--section", choices=("all", "parity", "fft", "benchmarks", "ceiling", "merge"), default="all")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--p2-report", type=Path, default=ROOT / "artifacts/p2_engineering_pilot/report.json")
    parser.add_argument("--p5-report", type=Path, default=ROOT / "artifacts/p5_timing.json")
    parser.add_argument("--comparison-report", type=Path, default=ROOT / "artifacts/mps_fft_torch2141.json")
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    versions = {"python": platform.python_version(), "torch": torch.__version__}
    for name in ("numpy", "mlx", "jax", "jaxlib", "exponax", "pyarrow"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    report = json.loads(args.out.read_text()) if args.out.exists() else {}
    if report and report["versions"] != versions:
        parser.error("existing report has different dependency versions; supply another --out")
    report.update(versions=versions, cpu_threads=args.cpu_threads)
    if "host" not in report:
        report["host"] = host_metadata()
    if args.section != "merge":
        report.setdefault("run_context", {})[args.section] = {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "power": command("pmset", "-g", "batt").splitlines()[0],
        }
    if args.p2_report.exists():
        p2 = json.loads(args.p2_report.read_text())
        report["p2_engineering_pilot"] = {
            "path": str(args.p2_report.relative_to(ROOT)) if args.p2_report.is_relative_to(ROOT) else str(args.p2_report),
            "timings": p2["timings"], "refinement": p2["refinement"], "validity": p2["validity"],
        }
    if args.p5_report.exists():
        report["p5_timing_fixture"] = json.loads(args.p5_report.read_text())
    if args.comparison_report != args.out and args.comparison_report.exists():
        comparison = json.loads(args.comparison_report.read_text())
        report["alternate_torch_fft"] = {
            "torch": comparison["versions"]["torch"],
            "T-mps-2": comparison["mps_checks"]["T-mps-2"],
        }
    save(report, args.out)
    if args.section in ("all", "parity", "fft"):
        has_mps = report["host"]["mps_available"]
        unavailable = {"status": "blocked", "error": "PyTorch cannot access Metal in this process"}
        parity = report.setdefault("mps_checks", {})
        parity["T-mps-2"] = attempt(odd_fft_roundtrip) if has_mps else unavailable.copy()
        parity["T-mps-2"].update(device="mps float32", absolute_error_limit=ROUNDTRIP_LIMIT)
        save(report, args.out)
        print("T-mps-2", parity["T-mps-2"], flush=True)
        if args.section == "fft":
            return
        parity["T-mps-1"] = attempt(fft_adam_parity) if has_mps else unavailable.copy()
        parity["T-mps-1"].update(device="mps float32", gradient_limit=GRADIENT_LIMIT, loss_limit=LOSS_LIMIT)
        print("T-mps-1", parity["T-mps-1"]["status"], flush=True)
        if any(parity[test]["status"] != "passed" for test in ("T-mps-1", "T-mps-2")):
            save(report, args.out)
            from trajcp.emulators.hardware_benchmarks import cpu_1d_fallback

            parity["cpu_float32_1d_fallback"] = attempt(cpu_1d_fallback)
            parity["cpu_float32_odd_fft"] = attempt(odd_fft_roundtrip, device="cpu")
        save(report, args.out)
    if args.section in ("all", "benchmarks"):
        from trajcp.emulators.hardware_benchmarks import run_benchmarks

        for name, result in run_benchmarks():
            report[name] = result
            save(report, args.out)
            print(name, "recorded", flush=True)
    if args.section == "ceiling":
        from trajcp.emulators.hardware_benchmarks import batch_ceiling

        report["fno_batch_ceiling"] = batch_ceiling()
        save(report, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
