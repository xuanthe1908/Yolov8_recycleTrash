#!/usr/bin/env python3

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

RECYCLABLE_SRC = ("cardboard", "glass", "metal", "paper", "plastic")
NON_SRC = ("battery", "biological", "clothes", "shoes", "trash")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    raw = ROOT / "raw"
    original = raw / "original"
    if original.is_dir():
        out_rec = raw / "recyclable"
        out_non = raw / "non_recyclable"
        out_rec.mkdir(parents=True, exist_ok=True)
        out_non.mkdir(parents=True, exist_ok=True)

        def move_group(folder_names: tuple[str, ...], dest: Path, tag: str) -> int:
            n = 0
            for name in folder_names:
                src_dir = original / name
                if not src_dir.is_dir():
                    continue
                for p in src_dir.iterdir():
                    if not p.is_file() or p.suffix.lower() not in IMG_EXT:
                        continue
                    target = dest / f"{name}_{p.name}"
                    if target.exists():
                        stem, suf = p.stem, p.suffix
                        k = 1
                        while target.exists():
                            target = dest / f"{name}_{stem}_{k}{suf}"
                            k += 1
                    shutil.move(str(p), str(target))
                    n += 1
                shutil.rmtree(src_dir)
            print(f"{tag}: {n} ảnh.")
            return n

        move_group(RECYCLABLE_SRC, out_rec, "Tái chế")
        move_group(NON_SRC, out_non, "Không tái chế")

        try:
            original.rmdir()
        except OSError:
            shutil.rmtree(original)

    for name in ("standardized_256", "standardized_384"):
        d = raw / name
        if d.is_dir():
            shutil.rmtree(d)
            print(f"Đã xóa: raw/{name}/")


if __name__ == "__main__":
    main()
