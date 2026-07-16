"""
AdFrame 2.0 — System Information Collector
===========================================
Collects complete reproducibility metadata:
  GPU, CUDA, PyTorch, Python, OS, drivers, FFmpeg, git state, timing.
"""

import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("adframe2.system_info")


def _run(cmd: list, fallback: str = "unavailable") -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() or result.stderr.strip() or fallback
    except Exception:
        return fallback


def collect_system_info(seed: int = 42, git_root: str = ".") -> dict:
    """Collect all reproducibility metadata into a single dict."""
    now = datetime.now(timezone.utc)

    # --- GPU & CUDA ---
    gpu_info = {}
    try:
        import torch
        gpu_info["available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            gpu_info["device_count"] = torch.cuda.device_count()
            gpu_info["device_name"] = torch.cuda.get_device_name(0)
            gpu_info["total_vram_mb"] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            gpu_info["allocated_vram_mb"] = torch.cuda.memory_allocated(0) // (1024 * 1024)
            gpu_info["reserved_vram_mb"] = torch.cuda.memory_reserved(0) // (1024 * 1024)
            gpu_info["cuda_version"] = torch.version.cuda
            gpu_info["cudnn_version"] = str(torch.backends.cudnn.version())
    except Exception as e:
        gpu_info["error"] = str(e)

    # --- Driver ---
    nvidia_smi = _run(["nvidia-smi", "--query-gpu=driver_version,name,memory.total,compute_cap",
                        "--format=csv,noheader,nounits"])

    # --- PyTorch / Transformers / Diffusers ---
    packages = {}
    for pkg in ["torch", "transformers", "diffusers", "accelerate", "qwen_vl_utils",
                 "PIL", "cv2", "numpy", "sentencepiece"]:
        try:
            mod = __import__(pkg)
            packages[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            packages[pkg] = "not_installed"

    # Flash Attention
    try:
        import flash_attn
        packages["flash_attn"] = flash_attn.__version__
    except ImportError:
        packages["flash_attn"] = "not_installed"

    # --- CPU & RAM ---
    cpu_info = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cores": os.cpu_count(),
    }
    try:
        with open("/proc/meminfo") as f:
            memlines = f.readlines()
        for line in memlines:
            if line.startswith("MemTotal"):
                total_kb = int(line.split()[1])
                cpu_info["ram_total_gb"] = round(total_kb / (1024 * 1024), 1)
            if line.startswith("MemAvailable"):
                avail_kb = int(line.split()[1])
                cpu_info["ram_available_gb"] = round(avail_kb / (1024 * 1024), 1)
    except Exception:
        pass

    # --- OS ---
    os_info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "python_version": sys.version,
        "python_executable": sys.executable,
    }

    # --- FFmpeg ---
    ffmpeg_version = _run(["ffmpeg", "-version"]).split("\n")[0]

    # --- Git ---
    git_info = {
        "commit": _run(["git", "-C", git_root, "rev-parse", "HEAD"]),
        "branch": _run(["git", "-C", git_root, "rev-parse", "--abbrev-ref", "HEAD"]),
        "remote": _run(["git", "-C", git_root, "remote", "get-url", "origin"]),
        "status": _run(["git", "-C", git_root, "status", "--short"]),
    }

    return {
        "timestamp_utc": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "seed": seed,
        "gpu": gpu_info,
        "nvidia_smi_csv": nvidia_smi,
        "cpu": cpu_info,
        "os": os_info,
        "packages": packages,
        "ffmpeg": ffmpeg_version,
        "git": git_info,
    }


def collect_model_registry_entry(
    model_id: str,
    role: str,
    precision: str,
    load_time_s: float,
    vram_before_mb: int,
    vram_after_mb: int,
) -> dict:
    """Record a single model's registry entry."""
    import torch

    # Try to get commit hash from HF Hub
    commit_hash = "unknown"
    try:
        from huggingface_hub import model_info
        info = model_info(model_id)
        commit_hash = info.sha or "unknown"
    except Exception:
        pass

    return {
        "role": role,
        "model_id": model_id,
        "hf_repo": f"https://huggingface.co/{model_id}",
        "commit_hash": commit_hash,
        "precision": precision,
        "dtype": precision,
        "load_time_seconds": round(load_time_s, 2),
        "vram_before_mb": vram_before_mb,
        "vram_after_mb": vram_after_mb,
        "vram_used_mb": vram_after_mb - vram_before_mb,
    }


def current_vram_mb() -> int:
    """Return current allocated VRAM in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(0) // (1024 * 1024)
    except Exception:
        pass
    return 0


def peak_vram_mb() -> int:
    """Return peak reserved VRAM in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_reserved(0) // (1024 * 1024)
    except Exception:
        pass
    return 0
