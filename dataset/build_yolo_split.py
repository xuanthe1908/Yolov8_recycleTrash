#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect(folder: Path, cls: int) -> list[tuple[Path, int]]:
    out: list[tuple[Path, int]] = []
    if not folder.is_dir():
        return out
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            out.append((p, cls))
    return out


def _clear_split_dirs() -> None:
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            d = ROOT / kind / split
            if d.is_dir():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
    lbl = ROOT / "labels"
    if lbl.is_dir():
        for cache in lbl.glob("*.cache"):
            cache.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo images/ + labels/ từ raw (YOLO)")
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Không cân bằng hai lớp (giữ số lượng gốc — dễ khiến model luôn dự đoán lớp đa số).",
    )
    args = parser.parse_args()

    random.seed(42)
    rec = ROOT / "raw" / "recyclable"
    non = ROOT / "raw" / "non_recyclable"
    if not rec.is_dir() or not non.is_dir():
        print("Cần có raw/recyclable và raw/non_recyclable.")
        return

    rec_items = _collect(rec, 0)
    non_items = _collect(non, 1)

    if not rec_items or not non_items:
        print("Thiếu ảnh trong raw/recyclable hoặc raw/non_recyclable.")
        return

    if args.no_balance:
        items = rec_items + non_items
        print(f"Chế độ không cân bằng: recyclable={len(rec_items)}, non_recyclable={len(non_items)}")
    else:
        n = min(len(rec_items), len(non_items))
        random.shuffle(rec_items)
        random.shuffle(non_items)
        rec_items = rec_items[:n]
        non_items = non_items[:n]
        items = rec_items + non_items
        print(f"Chế độ cân bằng: mỗi lớp {n} ảnh (tổng {2 * n}).")

    random.shuffle(items)
    n_val = max(1, len(items) // 5)
    val_items = items[:n_val]
    train_items = items[n_val:]

    _clear_split_dirs()

    def emit(split_name: str, pairs: list[tuple[Path, int]]) -> None:
        img_d = ROOT / "images" / split_name
        lbl_d = ROOT / "labels" / split_name
        for src, cls in pairs:
            dst = img_d / src.name
            if dst.exists():
                dst = img_d / f"{src.stem}_dup{src.suffix}"
            shutil.copy2(src, dst)
            lab = lbl_d / (dst.stem + ".txt")
            lab.write_text(f"{cls} 0.5 0.5 1 1\n", encoding="utf-8")

    emit("train", train_items)
    emit("val", val_items)

    def count_cls(pairs: list[tuple[Path, int]], c: int) -> int:
        return sum(1 for _, x in pairs if x == c)

    print(
        f"Train: {len(train_items)} (recyclable={count_cls(train_items, 0)}, "
        f"non_recyclable={count_cls(train_items, 1)})"
    )
    print(
        f"Val:   {len(val_items)} (recyclable={count_cls(val_items, 0)}, "
        f"non_recyclable={count_cls(val_items, 1)})"
    )


if __name__ == "__main__":
    main()
