"""Cross-platform hint for available PyTorch acceleration (CUDA, MPS, CPU)."""

from __future__ import annotations


def accelerator_label() -> str:
    try:
        import torch
    except ImportError:
        return "unknown (chưa có torch — cài ultralytics sẽ kéo PyTorch)"

    if torch.cuda.is_available():
        try:
            return f"CUDA: {torch.cuda.get_device_name(0)}"
        except Exception:
            return "CUDA"

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "MPS (Apple Silicon)"

    return "CPU"
