"""Shared CPU/CUDA selection for offline audio processing."""

from typing import Union

import torch


def resolve_device(device: Union[str, torch.device] = "cpu") -> torch.device:
    """Resolve an explicit device or choose an available GPU for ``auto``.

    Explicit CUDA requests never silently fall back to CPU. CUDA indices refer
    to visible devices, so CUDA_VISIBLE_DEVICES is respected.
    """
    if str(device) == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        selected = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError("Use cpu, cuda, cuda:N, or auto for device.") from exc
    if selected.type not in ("cpu", "cuda"):
        raise ValueError("Use cpu, cuda, cuda:N, or auto for device.")
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable. Check the server's "
                "PyTorch CUDA build and NVIDIA driver, or use --device cpu."
            )
        index = selected.index
        if index is None:
            index = torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is not available.")
        selected = torch.device("cuda", index)
    return selected


def check_device(device: torch.device) -> None:
    """Fail before processing files if a CUDA kernel cannot run on this GPU."""
    if device.type == "cuda":
        try:
            torch.ones(1, device=device).square().sum().item()
        except RuntimeError as exc:
            raise RuntimeError(
                f"Cannot execute CUDA kernels on {device}. Check that the "
                "installed PyTorch build supports this GPU and its driver."
            ) from exc
