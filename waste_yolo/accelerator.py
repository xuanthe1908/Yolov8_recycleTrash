"""Cross-platform hint for available PyTorch acceleration (CUDA, MPS, CPU)."""

from __future__ import annotations

from typing import Optional


def preferred_device_for_ultralytics() -> Optional[str]:
    """Device string to pass to Ultralytics when the user omits ``--device``.

    Prefer NVIDIA GPU (``"0"``), then Apple ``"mps"``, else ``None`` so
    Ultralytics falls back to CPU. This makes Mac Silicon runs use Metal
    consistently instead of leaving the choice implicit.
    """
    try:
        import torch
    except ImportError:
        return None

    if torch.cuda.is_available():
        return "0"

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"

    return None


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
