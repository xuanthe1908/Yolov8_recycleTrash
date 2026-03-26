#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any
import time

import cv2
from ultralytics import YOLO

from waste_yolo.recycling import load_recycling_config

ROOT = Path(__file__).resolve().parent
BEST_PT = ROOT / "runs" / "detect" / "waste" / "weights" / "best.pt"
BELT_TRACKER = ROOT / "config" / "tracker_belt.yaml"


class ArduinoSortLink:
    """Best-effort serial link to Arduino sorter."""

    def __init__(self, port: str, baud: int, cooldown_s: float) -> None:
        self._port = port
        self._baud = baud
        self._cooldown_s = max(cooldown_s, 0.0)
        self._last_send_at = 0.0
        self._last_payload = ""
        self._ser = None

        try:
            import serial  # type: ignore
        except Exception as exc:
            print(f"[WARN] pyserial chưa có ({exc}). Bỏ qua gửi Arduino.")
            return

        try:
            self._ser = serial.Serial(port=self._port, baudrate=self._baud, timeout=0.02)
            print(f"[INFO] Arduino serial opened: {self._port} @ {self._baud}")
            time.sleep(2.0)  # Nano CH340 often resets when serial opens.
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
    if BEST_PT.is_file():
        return str(BEST_PT)
    return str(ROOT / "yolov8n.pt")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict + phân loại tái chế")
    parser.add_argument("source", help="Ảnh, thư mục, video, hoặc 0 = webcam")
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="File .pt (mặc định: best.pt nếu đã train, không thì yolov8n.pt)",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="")
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
        help="Cổng serial Arduino, ví dụ /dev/cu.wchusbserial* hoặc COM3",
    )
    parser.add_argument("--serial-baud", type=int, default=115200, help="Baud rate serial")
    parser.add_argument(
        "--serial-cooldown",
        type=float,
        default=0.8,
        help="Khoảng nghỉ tối thiểu giữa 2 lệnh giống nhau gửi Arduino (giây)",
    )
    return parser.parse_args()


def _print_frame_labels(r: Any, mapping: dict[str, bool]) -> None:
    names = r.names
    if r.boxes is None or len(r.boxes) == 0:
        print("  (không có box)")
        return
    clss = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()
    for j, c in enumerate(clss):
        label = names[int(c)]
        rec = mapping.get(label, False)
        tag = "TÁI CHẾ" if rec else "KHÔNG TÁI CHẾ"
        cf = float(confs[j]) if j < len(confs) else 0.0
        print(f"  #{j} {label} -> {tag}  (conf={cf:.3f})")


def _print_labels(results: list[Any], mapping: dict[str, bool]) -> None:
    for ri, r in enumerate(results):
        print(f"[{ri}]")
        _print_frame_labels(r, mapping)


def _show_last_frame(results: list[Any]) -> None:
    img = results[-1].plot()
    cv2.imshow("YOLOv8 waste", img)
    cv2.waitKey(0)
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
) -> None:
    stream = model.track(
        source=int(args.source) if args.source.isdigit() else args.source,
        stream=True,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device or None,
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
    weights = args.weights or _default_weights()
    model = YOLO(weights)
    mapping = load_recycling_config()
    arduino = ArduinoSortLink(args.serial_port, args.serial_baud, args.serial_cooldown) if args.serial_port else None

    use_track = args.mode == "track" or (args.mode == "auto" and _is_track_source(args.source))
    if use_track:
        _run_track_stream(model, args, mapping, arduino)
        return

    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device or None,
        save=args.save,
        project=str(ROOT / "runs" / "predict"),
        name="waste",
        exist_ok=True,
    )
    _print_labels(results, mapping)
    if args.show and results:
        _show_last_frame(results)


if __name__ == "__main__":
    main()
