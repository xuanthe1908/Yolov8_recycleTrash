#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib
import sys
from pathlib import Path
from typing import Any
import time

import cv2
from ultralytics import YOLO

from waste_yolo.accelerator import accelerator_label, preferred_device_for_ultralytics
from waste_yolo.recycling import load_recycling_config

ROOT = Path(__file__).resolve().parent
BEST_PT_CLS = ROOT / "runs" / "classify" / "waste" / "weights" / "best.pt"
BEST_PT_DETECT = ROOT / "runs" / "detect" / "waste" / "weights" / "best.pt"
BELT_TRACKER = ROOT / "config" / "tracker_belt.yaml"


class ArduinoSortLink:
    """Best-effort serial link to ESP8266 / Arduino-class MCU sorter."""

    def __init__(self, port: str, baud: int, cooldown_s: float) -> None:
        self._port = port
        self._baud = baud
        self._cooldown_s = max(cooldown_s, 0.0)
        self._last_send_at = 0.0
        self._last_payload = ""
        self._ser = None

        try:
            serial_module = importlib.import_module("serial")
        except Exception as exc:
            print(f"[WARN] pyserial chưa có ({exc}). Bỏ qua gửi Arduino.")
            return

        try:
            self._ser = serial_module.Serial(port=self._port, baudrate=self._baud, timeout=0.02)
            print(f"[INFO] Serial sorter opened: {self._port} @ {self._baud}")
            time.sleep(2.0)  # USB-UART (CH340/CP2102) often resets ESP8266 on open.
        except Exception as exc:
            self._ser = None
            print(f"[WARN] Không mở được serial {self._port}: {exc}")

    @property
    def ready(self) -> bool:
        return self._ser is not None

    def send_class(self, recyclable: bool, class_name: str, confidence: float) -> bool:
        if not self._ser:
            return False

        payload = f"C:{'R' if recyclable else 'N'}\n"
        now = time.time()
        if payload == self._last_payload and (now - self._last_send_at) < self._cooldown_s:
            return False

        try:
            self._ser.write(payload.encode("ascii"))
            self._last_payload = payload
            self._last_send_at = now
            tag = "RECYCLABLE" if recyclable else "NON_RECYCLABLE"
            print(f"[SERIAL] {payload.strip()} ({class_name}, {tag}, conf={confidence:.3f})")
            return True
        except Exception as exc:
            print(f"[WARN] Gửi serial thất bại: {exc}")
            return False


def _default_weights() -> str:
    if BEST_PT_CLS.is_file():
        return str(BEST_PT_CLS)
    if BEST_PT_DETECT.is_file():
        return str(BEST_PT_DETECT)
    return str(ROOT / "yolov8n-cls.pt")


def _is_cls_model(weights_path: str) -> bool:
    """Kiem tra xem model co phai classification model khong."""
    try:
        import torch
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        task = ckpt.get("train_args", {}).get("task", "") or ckpt.get("task", "")
        return str(task).lower() in ("classify", "cls")
    except Exception:
        return "cls" in Path(weights_path).stem.lower()


def _resolve_inference_device(cli_device: str) -> str | None:
    """Ultralytics ``device``: user override, else MPS/CUDA default, else CPU path."""
    manual = (cli_device or "").strip()
    if manual:
        return manual
    return preferred_device_for_ultralytics()


def _list_opencv_camera_indices(max_index: int) -> list[int]:
    """OpenCV camera indices that open and return a frame (hữu ích tìm iPhone / Continuity Camera)."""
    apis: list[int | None] = [None]
    if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        apis.append(int(cv2.CAP_AVFOUNDATION))

    out: list[int] = []
    for i in range(max(0, max_index) + 1):
        opened = False
        for api in apis:
            cap = cv2.VideoCapture(i) if api is None else cv2.VideoCapture(i, api)
            if not cap.isOpened():
                cap.release()
                continue
            ok = False
            for _ in range(5):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    break
                time.sleep(0.02)
            cap.release()
            if ok:
                opened = True
                break
        if opened:
            out.append(i)
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict + phân loại tái chế")
    parser.add_argument(
        "source",
        nargs="?",
        default="0",
        help=(
            "Ảnh, thư mục, video, chỉ mục camera (0,1,…), hoặc URL (rtsp/http). "
            "Mac + iPhone: Continuity/USB thường là 1 nếu 0 là FaceTime HD — "
            "chạy --list-cameras để xem index."
        ),
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="File .pt (mặc định: best.pt nếu đã train, không thì yolov8n.pt)",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help=(
            "cpu | mps | 0 | … Để trống: ưu tiên CUDA rồi MPS (Apple Silicon); "
            "trên Mac inference dùng Metal ổn định."
        ),
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Liệt kê index camera OpenCV mở được (iPhone/Continuity thường khác 0), thoát.",
    )
    parser.add_argument(
        "--list-camera-max",
        type=int,
        default=10,
        help="Thử index 0..N với --list-cameras (mặc định 10).",
    )
    parser.add_argument("--save", action="store_true", help="Lưu ảnh/video kết quả")
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "predict", "track"],
        help="auto: webcam/video dùng track, ảnh dùng predict",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default=str(BELT_TRACKER),
        help="File YAML tracker (mặc định: config/tracker_belt.yaml)",
    )
    parser.add_argument(
        "--track-confirm-frames",
        type=int,
        default=4,
        help="Chỉ log track khi ID xuất hiện >= N frame để giảm nhảy ID",
    )
    parser.add_argument(
        "--track-log-every",
        type=int,
        default=10,
        help="In log tracking mỗi N frame",
    )
    parser.add_argument(
        "--track-max-missed",
        type=int,
        default=45,
        help="Xóa lịch sử ID khi vắng mặt quá N frame",
    )
    parser.add_argument(
        "--center-only",
        action="store_true",
        help="Chi tracking 1 vat the gan tam khung hinh",
    )
    parser.add_argument(
        "--center-window-ratio",
        type=float,
        default=0.35,
        help="Ti le vung trung tam de chon muc tieu (0-1)",
    )
    parser.add_argument(
        "--target-max-missed",
        type=int,
        default=20,
        help="So frame mat muc tieu truoc khi chon ID moi",
    )
    parser.add_argument("--show", action="store_true", help="Hiển thị khung hình")
    parser.add_argument(
        "--serial-port",
        type=str,
        default="",
        help="Cổng serial ESP8266/USB-UART, ví dụ /dev/cu.*, /dev/ttyUSB0 hoặc COM3",
    )
    parser.add_argument("--serial-baud", type=int, default=115200, help="Baud rate serial")
    parser.add_argument(
        "--serial-cooldown",
        type=float,
        default=0.8,
        help="Khoảng nghỉ tối thiểu giữa 2 lệnh giống nhau gửi Arduino (giây)",
    )
    return parser.parse_args()


def _print_cls_result(r: Any, mapping: dict[str, bool]) -> None:
    """In ket qua classification (khong co bbox)."""
    if r.probs is None:
        print("  (khong co ket qua)")
        return
    top1_idx = int(r.probs.top1)
    top1_conf = float(r.probs.top1conf)
    label = r.names[top1_idx]
    rec = mapping.get(label, False)
    tag = "TAI CHE" if rec else "KHONG TAI CHE"
    print(f"  {label} -> {tag}  (conf={top1_conf:.3f})")


def _print_frame_labels(r: Any, mapping: dict[str, bool]) -> None:
    names = r.names
    if r.boxes is None or len(r.boxes) == 0:
        print("  (khong co box)")
        return
    clss = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()
    for j, c in enumerate(clss):
        label = names[int(c)]
        rec = mapping.get(label, False)
        tag = "TAI CHE" if rec else "KHONG TAI CHE"
        cf = float(confs[j]) if j < len(confs) else 0.0
        print(f"  #{j} {label} -> {tag}  (conf={cf:.3f})")


def _print_labels(results: list[Any], mapping: dict[str, bool], is_cls: bool = False) -> None:
    for ri, r in enumerate(results):
        print(f"[{ri}]")
        if is_cls:
            _print_cls_result(r, mapping)
        else:
            _print_frame_labels(r, mapping)


def _show_last_frame(results: list[Any]) -> None:
    img = results[-1].plot()
    cv2.imshow("YOLOv8 waste", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _draw_cls_overlay(
    img: Any,
    label: str,
    tag: str,
    conf: float,
    rec: bool,
    zone: tuple[int, int, int, int],
    active: bool,
) -> None:
    """Ve vung target zone va ket qua classification len frame."""
    import numpy as np

    zx1, zy1, zx2, zy2 = zone
    zone_color = (0, 255, 255)  # cyan khi cho vat the
    result_color = (0, 220, 0) if rec else (0, 0, 220)

    # Ve khung vung target
    cv2.rectangle(img, (zx1, zy1), (zx2, zy2), zone_color, 2)

    # Ve goc khung (nhat manh hon)
    corner = 18
    t = 3
    for px, py, dx, dy in [
        (zx1, zy1, 1, 1),
        (zx2, zy1, -1, 1),
        (zx1, zy2, 1, -1),
        (zx2, zy2, -1, -1),
    ]:
        cv2.line(img, (px, py), (px + dx * corner, py), zone_color, t)
        cv2.line(img, (px, py), (px, py + dy * corner), zone_color, t)

    # Nhan vung
    cv2.putText(img, "CONVEYOR ZONE", (zx1 + 6, zy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, zone_color, 1)

    h, w = img.shape[:2]

    if not active:
        # Khong co vat (conf thap)
        cv2.putText(img, "WAITING FOR OBJECT...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2)
        return

    # Banner nen mo
    banner_h = 70
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    # Nhan chinh
    main_text = f"{label}  |  {tag}  |  {conf:.0%}"
    cv2.putText(img, main_text, (16, 42), cv2.FONT_HERSHEY_DUPLEX, 1.0, result_color, 2)

    # Thanh confidence
    bar_x, bar_y, bar_w, bar_h2 = 16, 54, int(w * 0.45), 10
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h2), (60, 60, 60), -1)
    filled = int(bar_w * conf)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h2), result_color, -1)


def _run_cls_stream(
    model: YOLO,
    args: argparse.Namespace,
    mapping: dict[str, bool],
    arduino: "ArduinoSortLink | None",
    device: "str | None",
) -> None:
    """Stream classification model: crop vung trung tam truoc khi classify."""
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Khong mo duoc source: {args.source}")
        return

    imgsz = args.imgsz if args.imgsz != 640 else 224
    ratio = max(min(args.center_window_ratio, 0.95), 0.1)
    frame_id = 0
    should_show = args.show or args.source.isdigit() if isinstance(args.source, str) else True

    print(f"[INFO] Classification stream | zone_ratio={ratio:.2f} | imgsz={imgsz} | q=thoat")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        half_w = max(20, int(w * ratio / 2))
        half_h = max(20, int(h * ratio / 2))
        zx1, zy1 = cx - half_w, cy - half_h
        zx2, zy2 = cx + half_w, cy + half_h

        # Crop vung trung tam de classify
        crop = frame[zy1:zy2, zx1:zx2]

        results = model.predict(
            crop,
            conf=args.conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
            save=False,
        )
        r = results[0]

        label, tag, top1_conf, rec = "unknown", "?", 0.0, False
        active = False
        if r.probs is not None:
            top1_idx = int(r.probs.top1)
            top1_conf = float(r.probs.top1conf)
            label = r.names[top1_idx]
            rec = mapping.get(label, False)
            tag = "TAI CHE" if rec else "KHONG TAI CHE"
            active = top1_conf >= args.conf

        if active and frame_id % max(args.track_log_every, 1) == 0:
            print(f"[F{frame_id}] {label} -> {tag} (conf={top1_conf:.3f})")

        if active and arduino and arduino.ready:
            arduino.send_class(rec, label, top1_conf)

        if should_show:
            disp = frame.copy()
            _draw_cls_overlay(
                disp, label, tag, top1_conf, rec,
                (zx1, zy1, zx2, zy2), active,
            )
            cv2.imshow("YOLOv8 waste classify - q to quit", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if should_show:
        cv2.destroyAllWindows()


def _is_track_source(source: str) -> bool:
    s = source.strip().lower()
    if s.isdigit() or s.startswith(("rtsp://", "rtmp://", "http://", "https://")):
        return True
    return Path(s).suffix in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _run_track_stream(
    model: YOLO,
    args: argparse.Namespace,
    mapping: dict[str, bool],
    arduino: ArduinoSortLink | None,
    device: str | None,
) -> None:
    stream = model.track(
        source=int(args.source) if args.source.isdigit() else args.source,
        stream=True,
        conf=args.conf,
        imgsz=args.imgsz,
        device=device,
        tracker=args.tracker,
        persist=True,
        save=args.save,
        project=str(ROOT / "runs" / "predict"),
        name="waste",
        exist_ok=True,
    )

    seen_count: dict[int, int] = defaultdict(int)
    missed_count: dict[int, int] = defaultdict(int)
    frame_id = 0
    should_show = args.show or args.source.isdigit()
    target_id: int | None = None
    target_missed = 0

    for r in stream:
        frame_id += 1

        active_ids: set[int] = set()
        target_row: tuple[int, int, float, float, float, float, float] | None = None
        rows: list[tuple[int, int, float, float, float, float, float]] = []

        if r.boxes is not None and len(r.boxes) > 0 and r.boxes.id is not None:
            names = r.names
            track_ids = r.boxes.id.int().cpu().tolist()
            clss = r.boxes.cls.int().cpu().tolist()
            confs = r.boxes.conf.cpu().tolist()
            xyxy = r.boxes.xyxy.cpu().tolist()

            for tid, cls_idx, cf, box in zip(track_ids, clss, confs, xyxy):
                x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                rows.append((tid, cls_idx, float(cf), x1, y1, x2, y2))

            for tid, cls_idx, cf, _x1, _y1, _x2, _y2 in rows:
                active_ids.add(tid)
                seen_count[tid] += 1
                missed_count[tid] = 0

            if args.center_only and rows:
                h, w = r.orig_shape
                cx, cy = w / 2.0, h / 2.0
                half_w = max(10.0, (w * max(min(args.center_window_ratio, 1.0), 0.05)) / 2.0)
                half_h = max(10.0, (h * max(min(args.center_window_ratio, 1.0), 0.05)) / 2.0)

                target_candidates = []
                for row in rows:
                    tid, _cls_idx, _cf, x1, y1, x2, y2 = row
                    bx, by = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    if abs(bx - cx) <= half_w and abs(by - cy) <= half_h:
                        dist2 = (bx - cx) ** 2 + (by - cy) ** 2
                        target_candidates.append((dist2, row))
                    elif target_id is not None and tid == target_id:
                        # Cho phep giu lock hien tai du box ra khoi vung tam trong ngan han.
                        dist2 = (bx - cx) ** 2 + (by - cy) ** 2
                        target_candidates.append((dist2, row))

                if target_id is not None:
                    for _d2, row in target_candidates:
                        if row[0] == target_id:
                            target_row = row
                            target_missed = 0
                            break

                if target_row is None and target_candidates:
                    _dist, best = min(target_candidates, key=lambda x: x[0])
                    target_id = best[0]
                    target_row = best
                    target_missed = 0

                if target_row is None and target_id is not None:
                    target_missed += 1
                    if target_missed > max(args.target_max_missed, 1):
                        target_id = None
                        target_missed = 0
            else:
                target_row = None

            if args.center_only and target_row is not None:
                tid, cls_idx, cf, _x1, _y1, _x2, _y2 = target_row
                if seen_count[tid] >= args.track_confirm_frames and frame_id % max(args.track_log_every, 1) == 0:
                    label = names[int(cls_idx)]
                    rec = mapping.get(label, False)
                    tag = "TAI CHE" if rec else "KHONG TAI CHE"
                    print(f"[F{frame_id}] TARGET ID {tid:03d} {label} -> {tag} (conf={float(cf):.3f})")
                if seen_count[tid] >= args.track_confirm_frames and arduino and arduino.ready:
                    label = names[int(cls_idx)]
                    rec = mapping.get(label, False)
                    arduino.send_class(rec, label, float(cf))
            elif not args.center_only:
                for tid, cls_idx, cf, _x1, _y1, _x2, _y2 in rows:
                    # Chỉ công bố track khi đã ổn định vài frame, giảm track ảo trên băng tải.
                    if seen_count[tid] >= args.track_confirm_frames and frame_id % max(args.track_log_every, 1) == 0:
                        label = names[int(cls_idx)]
                        rec = mapping.get(label, False)
                        tag = "TAI CHE" if rec else "KHONG TAI CHE"
                        print(f"[F{frame_id}] ID {tid:03d} {label} -> {tag} (conf={float(cf):.3f})")
                    if seen_count[tid] >= args.track_confirm_frames and arduino and arduino.ready:
                        label = names[int(cls_idx)]
                        rec = mapping.get(label, False)
                        arduino.send_class(rec, label, float(cf))

        for tid in list(seen_count.keys()):
            if tid not in active_ids:
                missed_count[tid] += 1
                if missed_count[tid] > args.track_max_missed:
                    seen_count.pop(tid, None)
                    missed_count.pop(tid, None)

        if should_show:
            img = r.plot()
            if args.center_only:
                h, w = r.orig_shape
                cx, cy = int(w / 2), int(h / 2)
                half_w = int(max(10.0, (w * max(min(args.center_window_ratio, 1.0), 0.05)) / 2.0))
                half_h = int(max(10.0, (h * max(min(args.center_window_ratio, 1.0), 0.05)) / 2.0))
                cv2.rectangle(img, (cx - half_w, cy - half_h), (cx + half_w, cy + half_h), (255, 255, 0), 2)
                cv2.putText(img, "CENTER TARGET MODE", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                if target_id is not None:
                    cv2.putText(img, f"Target ID: {target_id}", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("YOLOv8 waste tracking - q to quit", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if should_show:
        cv2.destroyAllWindows()


def main() -> None:
    args = _parse_args()
    if args.list_cameras:
        mx = max(0, args.list_camera_max)
        print(f"[INFO] Đang thử mở camera index 0..{mx} (OpenCV; Mac dùng thêm AVFoundation nếu cần)…")
        ids = _list_opencv_camera_indices(mx)
        if not ids:
            print("  (không mở được camera nào — cấp quyền Camera cho Terminal/Python, thử cắm iPhone)")
        else:
            print("  Index mở được:", ", ".join(str(i) for i in ids))
            print("  Gợi ý: python predict.py <index> --mode track --show")
        return

    run_device = _resolve_inference_device(args.device)
    if (args.device or "").strip():
        print(f"[INFO] device={run_device} — {accelerator_label()}")
    elif run_device:
        print(f"[INFO] device={run_device} (mặc định Mac Silicon/CUDA) — {accelerator_label()}")
    else:
        print(f"[INFO] device=(CPU) — {accelerator_label()}")

    weights = args.weights or _default_weights()
    is_cls = _is_cls_model(weights)
    model = YOLO(weights)
    mapping = load_recycling_config()
    arduino = ArduinoSortLink(args.serial_port, args.serial_baud, args.serial_cooldown) if args.serial_port else None

    task_label = "CLASSIFICATION" if is_cls else "DETECTION"
    print(f"[INFO] weights={Path(weights).name}  task={task_label}")

    is_stream_source = _is_track_source(args.source)

    if is_cls:
        if is_stream_source:
            _run_cls_stream(model, args, mapping, arduino, run_device)
        else:
            results = model.predict(
                source=args.source,
                conf=args.conf,
                imgsz=args.imgsz if args.imgsz != 640 else 224,
                device=run_device,
                save=args.save,
                project=str(ROOT / "runs" / "predict"),
                name="waste",
                exist_ok=True,
            )
            _print_labels(results, mapping, is_cls=True)
            if args.show and results:
                _show_last_frame(results)
        return

    use_track = args.mode == "track" or (args.mode == "auto" and is_stream_source)
    if use_track:
        _run_track_stream(model, args, mapping, arduino, run_device)
        return

    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=run_device,
        save=args.save,
        project=str(ROOT / "runs" / "predict"),
        name="waste",
        exist_ok=True,
    )
    _print_labels(results, mapping, is_cls=False)
    if args.show and results:
        _show_last_frame(results)


if __name__ == "__main__":
    main()
