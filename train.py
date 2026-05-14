#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from waste_yolo.accelerator import accelerator_label, preferred_device_for_ultralytics

ROOT = Path(__file__).resolve().parent
SAVE_DIR = ROOT / "runs" / "detect" / "waste"


def _flush_print(*args: Any, **kwargs: Any) -> None:
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def _fit_epoch_end(trainer: Any) -> None:
    if getattr(trainer, "metrics", None) is None:
        return
    cur = int(getattr(trainer, "epoch", 0)) + 1
    total = int(getattr(trainer, "epochs", 0))
    fit = getattr(trainer, "fitness", None)
    m = trainer.metrics
    parts: list[str] = []
    if isinstance(m, dict):
        for key in (
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "metrics/precision(B)",
            "metrics/recall(B)",
        ):
            if key in m and m[key] is not None:
                short = key.replace("metrics/", "").replace("(B)", "")
                parts.append(f"{short}={float(m[key]):.4f}")
    if fit is not None:
        parts.insert(0, f"fitness={float(fit):.4f}")
    line = " | ".join(parts) if parts else str(m)
    _flush_print(f"\n>>> Tiến độ: epoch {cur}/{total} | {line}")
    _flush_print(f"    Thư mục log: {SAVE_DIR}\n")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Train YOLOv8 — recyclable / non_recyclable")
    parser.add_argument(
        "--data",
        type=str,
        default=str(ROOT / "dataset" / "dataset.yaml"),
        help="File dataset.yaml",
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Weights gốc (n/s/m/l/x)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="", help="cpu | mps | 0 | 0,1 ...")
    args = parser.parse_args()

    resolved_device = (args.device or "").strip() or preferred_device_for_ultralytics()
    dev = resolved_device if resolved_device else "(auto — Ultralytics chọn)"
    _flush_print("=" * 60)
    _flush_print("BẮT ĐẦU TRAIN")
    _flush_print(f"  data     : {args.data}")
    _flush_print(f"  model    : {args.model}")
    _flush_print(f"  epochs   : {args.epochs}")
    _flush_print(f"  imgsz    : {args.imgsz}")
    _flush_print(f"  batch    : {args.batch}")
    _flush_print(f"  device   : {dev}")
    if not (args.device or "").strip():
        _flush_print(f"  phát hiện: {accelerator_label()}")
        if resolved_device:
            _flush_print(f"  (mặc định Apple Silicon / CUDA: dùng {resolved_device})")
    _flush_print(f"  save_dir : {SAVE_DIR}")
    _flush_print("  (Ultralytics in loss từng batch; sau mỗi epoch có dòng >>> Tiến độ)")
    _flush_print("=" * 60)

    model = YOLO(args.model)
    model.add_callback("on_fit_epoch_end", _fit_epoch_end)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=resolved_device,
        project=str(ROOT / "runs" / "detect"),
        name="waste",
        exist_ok=True,
        plots=True,
        verbose=True,
        cls=1.2,
    )


if __name__ == "__main__":
    main()
