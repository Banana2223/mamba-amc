"""P1-T4: collect environment information and validate the Mamba CUDA stack."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "P1-T4"
OUTPUT_FILE = OUTPUT_DIR / "environment_report.json"


def get_package_version(package_name: str) -> str | None:
    """Return an installed distribution version without importing it."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def get_memory_info() -> dict[str, Any]:
    """Read Linux memory statistics from /proc/meminfo."""
    result: dict[str, Any] = {}
    try:
        values: dict[str, int] = {}
        with Path("/proc/meminfo").open("r", encoding="utf-8") as file:
            for line in file:
                key, value = line.split(":", 1)
                values[key] = int(value.strip().split()[0])

        result["total_gib"] = round(values["MemTotal"] / 1024**2, 2)
        result["available_gib"] = round(values["MemAvailable"] / 1024**2, 2)
        result["swap_total_gib"] = round(values["SwapTotal"] / 1024**2, 2)
    except (OSError, KeyError, ValueError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def get_gpu_info() -> dict[str, Any]:
    """Query the first NVIDIA GPU through nvidia-smi."""
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.free,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        first_gpu = process.stdout.strip().splitlines()[0]
        fields = [value.strip() for value in first_gpu.split(",")]
        return {
            "name": fields[0],
            "driver_version": fields[1],
            "memory_total_mib": int(fields[2]),
            "memory_free_mib": int(fields[3]),
            "compute_capability": fields[4],
        }
    except (OSError, subprocess.SubprocessError, IndexError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_mamba_smoke_test() -> dict[str, Any]:
    """Run a small forward/backward pass with the project's target sequence length."""
    result: dict[str, Any] = {}
    try:
        import torch
        from mamba_ssm import Mamba

        result["torch_version"] = torch.__version__
        result["torch_cuda_build"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        result["device_count"] = torch.cuda.device_count()

        if not torch.cuda.is_available():
            result["passed"] = False
            result["error"] = "PyTorch cannot use CUDA"
            return result

        device = torch.device("cuda")
        result["device_name"] = torch.cuda.get_device_name(0)

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        model = Mamba(
            d_model=64,
            d_state=16,
            d_conv=4,
            expand=2,
        ).to(device)
        model.train()

        x = torch.randn(8, 1024, 64, device=device, requires_grad=True)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        y = model(x)
        loss = y.square().mean()
        loss.backward()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        output_is_finite = bool(torch.isfinite(y).all().item())

        result.update(
            {
                "input_shape": list(x.shape),
                "output_shape": list(y.shape),
                "loss": float(loss.detach().item()),
                "elapsed_seconds": round(elapsed, 4),
                "peak_vram_mib": round(
                    torch.cuda.max_memory_allocated() / 1024**2,
                    2,
                ),
                "output_is_finite": output_is_finite,
                "passed": tuple(y.shape) == tuple(x.shape) and output_is_finite,
            }
        )
    except Exception as exc:  # Report dependency/runtime failures in the artifact.
        result["passed"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def build_report() -> dict[str, Any]:
    disk = shutil.disk_usage(PROJECT_ROOT)
    packages = [
        "torch",
        "numpy",
        "h5py",
        "scipy",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "tqdm",
        "einops",
        "causal-conv1d",
        "mamba-ssm",
    ]

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "memory": get_memory_info(),
        "disk": {
            "total_gib": round(disk.total / 1024**3, 2),
            "free_gib": round(disk.free / 1024**3, 2),
        },
        "gpu": get_gpu_info(),
        "packages": {
            package: get_package_version(package) for package in packages
        },
        "mamba_smoke_test": run_mamba_smoke_test(),
    }


def main() -> int:
    report = build_report()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nEnvironment report saved to: {OUTPUT_FILE}")
    return 0 if report["mamba_smoke_test"].get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
