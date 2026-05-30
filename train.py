#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from waste_yolo.accelerator import accelerator_label, preferred_device_for_ultralytics

ROOT = Path(__file__).resolve().parent
SAVE_DIR_DETECT = ROOT / "runs" / "detect" / "waste"
SAVE_DIR_CLS = ROOT / "runs" / "classify" / "waste"


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
            "metrics/accuracy_top1",
            "metrics/accuracy_top5",
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
    _flush_print(f"\n>>> Tien do: epoch {cur}/{total} | {line}\n")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    parser = argparse.ArgumentParser(description="Train YOLOv8 — recyclable / non_recyclable")
    parser.add_argument(
        "--task",
        type=str,
        default="cls",
        choices=["cls", "detect"],
        help="cls: classification (khong can bbox, dung cho dataset nay); detect: detection co bbox",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="",
        help=(
            "cls: thu muc chua cls_train/ (de trong = tu dong tim dataset/cls_train); "
            "detect: file dataset.yaml"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Weights goc: de trong = yolov8n-cls.pt (cls) hoac yolov8n.pt (detect)",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=224, help="224 cho cls, 640 cho detect")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="", help="cpu | 0 | 0,1 ...")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="So dataloader workers (0 = main process, tranh loi WinError 1455 tren Windows)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Tiep tuc tu checkpoint last.pt cua lan train truoc",
    )
    args = parser.parse_args()

    resolved_device = (args.device or "").strip() or preferred_device_for_ultralytics()
    dev = resolved_device if resolved_device else "(auto)"

    if args.task == "cls":
        data_path = args.data or str(ROOT / "dataset" / "cls_train")
        model_name = args.model or "yolov8n-cls.pt"
        imgsz = args.imgsz if args.imgsz != 224 else 224
        save_dir = SAVE_DIR_CLS
        project_dir = str(ROOT / "runs" / "classify")
        extra_kwargs: dict[str, Any] = {}
    else:
        data_path = args.data or str(ROOT / "dataset" / "dataset.yaml")
        model_name = args.model or "yolov8n.pt"
        imgsz = args.imgsz if args.imgsz != 224 else 640
        save_dir = SAVE_DIR_DETECT
        project_dir = str(ROOT / "runs" / "detect")
        extra_kwargs = {"cls": 1.2}

    _flush_print("=" * 60)
    _flush_print(f"BAT DAU TRAIN  [task={args.task}]")
    _flush_print(f"  data     : {data_path}")
    _flush_print(f"  model    : {model_name}")
    _flush_print(f"  epochs   : {args.epochs}")
    _flush_print(f"  imgsz    : {imgsz}")
    _flush_print(f"  batch    : {args.batch}")
    _flush_print(f"  device   : {dev}")
    if not (args.device or "").strip():
        _flush_print(f"  phat hien: {accelerator_label()}")
    _flush_print(f"  save_dir : {save_dir}")
    _flush_print(f"  workers  : {args.workers}")
    if args.resume:
        last_pt = save_dir / "weights" / "last.pt"
        _flush_print(f"  resume   : {last_pt}")
    _flush_print("=" * 60)

    if args.resume:
        last_pt = save_dir / "weights" / "last.pt"
        if not last_pt.is_file():
            _flush_print(f"[ERROR] Khong tim thay {last_pt} de resume.")
            return
        model = YOLO(str(last_pt))
        model.add_callback("on_fit_epoch_end", _fit_epoch_end)
        model.train(resume=True, workers=args.workers)
    else:
        model = YOLO(model_name)
        model.add_callback("on_fit_epoch_end", _fit_epoch_end)
        model.train(
            data=data_path,
            epochs=args.epochs,
            imgsz=imgsz,
            batch=args.batch,
            device=resolved_device,
            project=project_dir,
            name="waste",
            exist_ok=True,
            plots=True,
            verbose=True,
            workers=args.workers,
            **extra_kwargs,
        )


if __name__ == "__main__":
    main()
