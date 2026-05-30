#!/usr/bin/env python3

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "runs" / "detect" / "waste" / "results.csv"
OUT_PNG = ROOT / "runs" / "detect" / "waste" / "training_curves.png"


def _nums(rows: list[dict[str, str]], key: str) -> list[float]:
    out: list[float] = []
    for r in rows:
        v = (r.get(key) or "").strip()
        if not v:
            continue
        try:
            out.append(float(v))
        except ValueError:
            pass
    return out


def _pick(keys: set[str], rows: list[dict[str, str]], *candidates: str) -> list[float]:
    for c in candidates:
        if c in keys:
            v = _nums(rows, c)
            if v:
                return v
    return []


def _load_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _build_figure(rows: list[dict[str, str]]) -> Any:
    keys = set(rows[0].keys())
    epochs = list(range(1, len(rows) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Kết quả huấn luyện YOLOv8 (waste)")

    box_l = _pick(keys, rows, "train/box_loss")
    cls_l = _pick(keys, rows, "train/cls_loss")
    if box_l:
        axes[0, 0].plot(epochs[: len(box_l)], box_l, label="box_loss")
    if cls_l:
        axes[0, 0].plot(epochs[: len(cls_l)], cls_l, label="cls_loss")
    axes[0, 0].set_title("Train loss")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    dfl = _pick(keys, rows, "train/dfl_loss")
    if dfl:
        axes[0, 1].plot(epochs[: len(dfl)], dfl, color="green")
    axes[0, 1].set_title("DFL loss")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].grid(True, alpha=0.3)

    mp = _pick(keys, rows, "metrics/mAP50(B)", "metrics/mAP50")
    mp50_95 = _pick(keys, rows, "metrics/mAP50-95(B)", "metrics/mAP50-95")
    if mp:
        axes[1, 0].plot(epochs[: len(mp)], mp, label="mAP50")
    if mp50_95:
        axes[1, 0].plot(epochs[: len(mp50_95)], mp50_95, label="mAP50-95")
    axes[1, 0].set_title("mAP (val)")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    p = _pick(keys, rows, "metrics/precision(B)", "metrics/precision")
    rec = _pick(keys, rows, "metrics/recall(B)", "metrics/recall")
    if p:
        axes[1, 1].plot(epochs[: len(p)], p, label="precision")
    if rec:
        axes[1, 1].plot(epochs[: len(rec)], rec, label="recall")
    axes[1, 1].set_title("Precision / Recall (val)")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not DEFAULT_CSV.is_file():
        print(f"Không tìm thấy {DEFAULT_CSV}. Chạy train xong sẽ có file này.")
        return

    rows = _load_rows(DEFAULT_CSV)
    if not rows:
        print("File results.csv rỗng.")
        return

    fig = _build_figure(rows)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"Đã lưu: {OUT_PNG}")


if __name__ == "__main__":
    main()
