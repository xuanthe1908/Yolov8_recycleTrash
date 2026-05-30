#!/usr/bin/env python3
"""
Tạo cấu trúc thư mục YOLO Classification từ dataset detection hiện có.

Đầu ra:
  dataset/
    cls_train/
      recyclable/      <-- ảnh class 0
      non_recyclable/  <-- ảnh class 1
    cls_val/
      recyclable/
      non_recyclable/
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLASS_NAMES = {0: "recyclable", 1: "non_recyclable"}


def _build_split(split: str) -> dict[str, int]:
    img_dir = ROOT / "images" / split
    lbl_dir = ROOT / "labels" / split

    counts: dict[str, int] = {name: 0 for name in CLASS_NAMES.values()}

    if not img_dir.is_dir():
        print(f"[WARN] Không tìm thấy {img_dir}")
        return counts

    out_root = ROOT / f"cls_{split}"

    for cls_name in CLASS_NAMES.values():
        (out_root / cls_name).mkdir(parents=True, exist_ok=True)

    for img_path in sorted(img_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.is_file():
            print(f"[WARN] Thiếu label cho {img_path.name} — bỏ qua")
            continue

        first_line = lbl_path.read_text(encoding="utf-8").strip().splitlines()[0]
        try:
            cls_idx = int(first_line.split()[0])
        except (ValueError, IndexError):
            print(f"[WARN] Label không hợp lệ: {lbl_path.name} — bỏ qua")
            continue

        cls_name = CLASS_NAMES.get(cls_idx)
        if cls_name is None:
            print(f"[WARN] Class index {cls_idx} không xác định — bỏ qua")
            continue

        dst = out_root / cls_name / img_path.name
        shutil.copy2(img_path, dst)
        counts[cls_name] += 1

    return counts


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== Build YOLO Classification split ===")
    for split in ("train", "val"):
        counts = _build_split(split)
        total = sum(counts.values())
        detail = ", ".join(f"{k}={v}" for k, v in counts.items())
        print(f"  cls_{split}: {total} ảnh ({detail})")

    print("\nĐường dẫn dùng để train classification:")
    print(f"  {ROOT / 'cls_train'}")
    print(f"  {ROOT / 'cls_val'}")
    print("\nChạy train:")
    print("  python train.py --task cls")


if __name__ == "__main__":
    main()
