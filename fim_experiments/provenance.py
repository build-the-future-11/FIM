from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tensor_sha256(tensors: Iterable[torch.Tensor]) -> str:
    """Hash tensor values, shapes, and dtypes without relying on pickle bytes."""
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def source_identity() -> str:
    """Hash executable research sources when a Git commit is unavailable."""
    paths = []
    for relative in ("fim", "fim_experiments", "scripts", "configs", "research"):
        root = PROJECT_ROOT / relative
        if root.exists():
            paths.extend(p for p in root.rglob("*") if p.is_file() and p.suffix in {".py", ".sh", ".yaml", ".yml", ".md"})
    paths.extend(p for p in (PROJECT_ROOT / "pyproject.toml", PROJECT_ROOT / "requirements.txt") if p.exists())
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def git_state() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"available": True, "commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "commit": None, "dirty": None}


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("torch", "numpy", "matplotlib", "PyYAML", "scipy", "pandas"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "NOT_INSTALLED"
    return versions


def runtime_metadata(*, device: torch.device, config: dict[str, Any], started_utc: str) -> dict[str, Any]:
    git = git_state()
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "started_utc": started_utc,
        "ended_utc": utc_now(),
        "git": git,
        "source_identity": source_identity(),
        "identity_kind": "git_and_sha256_source_tree" if git["available"] else "sha256_source_tree",
        "config_sha256": sha256_json(config),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "UNKNOWN",
        "torch_device": str(device),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "dependencies": dependency_versions(),
    }
    if device.type == "cuda":
        metadata["accelerator"] = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        metadata["accelerator"] = "Apple Metal Performance Shaders"
    else:
        metadata["accelerator"] = platform.processor() or platform.machine()
    return metadata
