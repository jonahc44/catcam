#!/usr/bin/env python3
"""
catcam - RTSP + YOLO cat detection with debug overlay.

Keys (in debug window):
  q / esc : quit
  s       : save current frame to snapshots/
  n       : name and store the clicked points as a zone (prompts in terminal)
  z       : print the full ZONES block, ready to paste
  c       : clear pending clicked points
  x       : delete the most recently added zone
  space   : pause/unpause

Workflow: click the outline of one counter, press 'n', type a name.
Repeat for the next counter. Press 'z' and paste the output into config.yaml.
"""

import os
import re
import math
import time
import threading
import socket
import socketserver
import http.server
import subprocess
from collections import deque
from pathlib import Path
import yaml
import requests
from datetime import datetime, timezone

# Must be set before cv2 import touches the FFMPEG backend.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|reorder_queue_size;0"
)

import cv2
import numpy as np
import onnxruntime as ort
from soco import SoCo

from webview import WebView

_cfg_path = Path(__file__).with_name("config.yaml")
if not _cfg_path.exists():
    _cfg_path.write_text(Path(__file__).with_name("config.example.yaml").read_text())
    print("[init] created config.yaml from config.example.yaml")
cfg = yaml.safe_load(_cfg_path.read_text())
cfg.setdefault("zones", {})


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def start_audio_server(port):
    handler = http.server.SimpleHTTPRequestHandler
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("", port), handler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    return httpd

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------


class Detector:
    def __init__(self, model_path: str, imgsz: int = 640):
        so = ort.SessionOptions()
        so.intra_op_num_threads = cfg["model"]["ort_threads"]
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sess = ort.InferenceSession(
            model_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.iname = self.sess.get_inputs()[0].name
        self.imgsz = imgsz

        oshape = self.sess.get_outputs()[0].shape
        dims = [d if isinstance(d, int) else -1 for d in oshape]
        if len(dims) != 3 or dims[1] < 5:
            raise SystemExit(
                f"[model] unexpected output shape {tuple(oshape)}.\n"
                "  This script targets YOLOv8/v11 grid output, e.g. (1, 84, 8400).\n"
                "  Export with: yolo export model=yolo11s.pt format=onnx imgsz=640 opset=12"
            )
        self.num_classes = dims[1] - 4
        print(f"[model] {model_path} {tuple(oshape)} "
              f"- {self.num_classes} classes")

        # Input size comes from the model when it's static.
        ishape = self.sess.get_inputs()[0].shape
        if isinstance(ishape[-1], int) and ishape[-1] > 0:
            if ishape[-1] != imgsz:
                print(f"[model] overriding imgsz {imgsz} -> {ishape[-1]} from model")
            self.imgsz = ishape[-1]

    def _letterbox(self, img):
        """Resize preserving aspect ratio, pad to square. Returns img, scale, pads."""
        h, w = img.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        top = (self.imgsz - nh) // 2
        left = (self.imgsz - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, r, left, top

    def __call__(self, frame, roi=None):
        """Detect on frame, optionally cropping to roi first.

        roi is (x1, y1, x2, y2) as fractions of frame size. Returned boxes are
        always in FULL-FRAME coordinates regardless of cropping.
        """
        h_full, w_full = frame.shape[:2]

        if roi:
            ox = int(roi[0] * w_full)
            oy = int(roi[1] * h_full)
            ex = int(roi[2] * w_full)
            ey = int(roi[3] * h_full)
            ox, oy = max(0, ox), max(0, oy)
            ex, ey = min(w_full, ex), min(h_full, ey)
            if ex - ox < 32 or ey - oy < 32:
                ox, oy, ex, ey = 0, 0, w_full, h_full
            crop = frame[oy:ey, ox:ex]
        else:
            ox, oy = 0, 0
            crop = frame

        lb, r, pad_x, pad_y = self._letterbox(crop)

        x = lb[:, :, ::-1].transpose(2, 0, 1)[None]
        x = np.ascontiguousarray(x, dtype=np.float32) / 255.0

        out = self.sess.run(None, {self.iname: x})[0][0]
        ch, cw = crop.shape[:2]

        # (4 + num_classes, num_anchors): box coords then per-class scores.
        scores_all = out[4:]

        # Merge the animal classes into one score, so a cat split between
        # "cat" and "dog" isn't thresholded out of existence.
        valid_animals = [c for c in cfg["classes"]["animal"] if c < self.num_classes]
        
        if valid_animals:
            animal_conf = scores_all[valid_animals].max(axis=0)
        else:
            # Fallback if the model has no animal classes mapped
            animal_conf = np.zeros(scores_all.shape[1], dtype=np.float32)

        conf_all = np.maximum(animal_conf, scores_all.max(axis=0))

        keep = conf_all > cfg["model"]["conf_thresh"]
        if not keep.any():
            return []

        cx, cy, bw, bh = out[:4, keep]
        lx1, ly1 = cx - bw / 2, cy - bh / 2
        lx2, ly2 = cx + bw / 2, cy + bh / 2
        conf = conf_all[keep]
        acon = animal_conf[keep]
        cls = scores_all[:, keep].argmax(axis=0)

        # letterboxed crop space -> full frame space
        x1 = np.clip((lx1 - pad_x) / r, 0, cw - 1) + ox
        y1 = np.clip((ly1 - pad_y) / r, 0, ch - 1) + oy
        x2 = np.clip((lx2 - pad_x) / r, 0, cw - 1) + ox
        y2 = np.clip((ly2 - pad_y) / r, 0, ch - 1) + oy

        rects = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
        idxs = cv2.dnn.NMSBoxes(
            rects.tolist(), conf.tolist(), cfg["model"]["conf_thresh"], cfg["model"]["iou_thresh"]
        )
        if len(idxs) == 0:
            return []
        idxs = np.array(idxs).flatten()

        return [
            (int(cls[i]), float(conf[i]),
             (int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])),
             float(acon[i]))
            for i in idxs
        ]


# ----------------------------------------------------------------------------
# Capture
# ----------------------------------------------------------------------------


class FreshestFrame(threading.Thread):
    """Drains the decoder at full rate, keeps only the most recent frame."""

    def __init__(self, url: str):
        super().__init__(daemon=True)
        self.url = url
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.fail_count = 0
        self.cap = self._open()
        self.start()

    def _open(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self._reconnect()
                continue

            ok, f = self.cap.read()
            if not ok:
                self.fail_count += 1
                if self.fail_count > 30:
                    self._reconnect()
                else:
                    time.sleep(0.05)
                continue

            self.fail_count = 0
            with self.lock:
                self.frame = f

    def _reconnect(self):
        print("[capture] reconnecting...")
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        time.sleep(2.0)
        self.cap = self._open()
        self.fail_count = 0

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()


# ----------------------------------------------------------------------------
# Zone logic
# ----------------------------------------------------------------------------


def check_zone_scale(zones, frame_shape, auto_fix=True):
    """Warn (and optionally rescale) when zones were clicked at another resolution.

    Zone points are absolute pixels, so switching stream2 -> stream1 leaves them
    covering a small corner of the frame. Detect that and offer the 4x scale.
    """
    if not zones:
        return
    h, w = frame_shape[:2]
    max_x = max(p[0] for poly in zones.values() for p in poly)
    max_y = max(p[1] for poly in zones.values() for p in poly)

    frac_x, frac_y = max_x / w, max_y / h
    if frac_x > 0.35 or frac_y > 0.35:
        return  # zones plausibly match this frame

    ratio = round(w / max(max_x, 1))
    print(f"\n[zones] WARNING: zone points reach only "
          f"({max_x}, {max_y}) in a {w}x{h} frame "
          f"({frac_x:.0%} x {frac_y:.0%} of it).")
    print("[zones] These were almost certainly clicked at a lower stream "
          "resolution.")

    if auto_fix:
        scale = w / 640
        if scale > 1.5 and abs(scale - round(scale)) < 0.05:
            scale = round(scale)
            for name, poly in list(zones.items()):
                zones[name] = [(int(x * scale), int(y * scale)) for x, y in poly]
            print(f"[zones] auto-scaled all zones by {scale}x "
                  f"(assuming they were made on a 640-wide preview).")
            print("[zones] VERIFY on the web view - re-click them if wrong.\n")
        else:
            print("[zones] Re-click them in the web view against this stream.\n")


def zone_rois(zones, frame_shape):
    """Derive one inference crop per zone, as (name, (fx1, fy1, fx2, fy2)).

    Fractions of the full frame, padded and clamped. Returns [(None, None)]
    when there are no zones, meaning "use the whole frame".
    """
    h, w = frame_shape[:2]
    if not zones:
        return [(None, None)]

    out = []
    for name, poly in zones.items():
        if len(poly) < 3:
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        bw, bh = x2 - x1, y2 - y1

        x1 -= bw * cfg["roi"]["pad_side"]
        x2 += bw * cfg["roi"]["pad_side"]
        y1 -= bh * cfg["roi"]["pad_top"]
        y2 += bh * cfg["roi"]["pad_bottom"]

        # Enforce a minimum crop size, growing about the centre.
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cw, chh = max(x2 - x1, cfg["roi"]["min_edge"]), max(y2 - y1, cfg["roi"]["min_edge"])
        x1, x2 = cx - cw / 2, cx + cw / 2
        y1, y2 = cy - chh / 2, cy + chh / 2

        x1 = max(0.0, min(x1, w - 1))
        y1 = max(0.0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, float(w)))
        y2 = max(y1 + 1, min(y2, float(h)))

        out.append((name, (x1 / w, y1 / h, x2 / w, y2 / h)))

    return out or [(None, None)]


def merge_detections(groups, iou_thresh=0.55):
    """Flatten per-crop detections and drop duplicates from overlapping crops."""
    flat = [d for g in groups for d in g]
    if len(flat) < 2:
        return flat

    boxes = [[d[2][0], d[2][1], d[2][2] - d[2][0], d[2][3] - d[2][1]] for d in flat]
    scores = [d[1] for d in flat]
    idxs = cv2.dnn.NMSBoxes(boxes, scores, cfg["model"]["conf_thresh"], iou_thresh)
    if len(idxs) == 0:
        return []
    return [flat[i] for i in np.array(idxs).flatten()]


def anchor_point(box):
    """Bottom-center of the box - roughly where the animal contacts a surface."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, y2)


def order_polygon(points):
    """Sort points counter-clockwise about their centroid.

    Clicking corners out of order produces a self-intersecting 'bowtie' whose
    area is a fraction of what was intended, so normalise on the way in.
    """
    pts = [(float(x), float(y)) for x, y in points]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    pts.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return [(int(round(x)), int(round(y))) for x, y in pts]


def which_zone(point, zones) -> str | None:
    """Name of the first zone containing point, or None.

    With no zones defined the whole frame is active, reported as "frame".
    """
    if not zones:
        return "frame"
    px, py = float(point[0]), float(point[1])
    for name, poly in zones.items():
        if len(poly) < 3:
            continue
        if cv2.pointPolygonTest(np.array(poly, dtype=np.int32), (px, py), False) >= 0:
            return name
    return None


class DatasetCapture:
    """Writes full-res frames + YOLO-format pre-labels for later fine-tuning.

    Layout (Ultralytics standard):
        dataset/images/<name>.jpg
        dataset/labels/<name>.txt      one line per box:
                                       <class> <cx> <cy> <w> <h>   (normalized)
        dataset/data.yaml              ready for `yolo train data=...`
        dataset/notes.csv              model conf per capture, for triage

    Labels are the model's guesses. Correct them in a labeling tool before
    training - the whole point is to fix what the model currently gets wrong.
    """

    def __init__(self, root: Path, classes: list[str]):
        self.root = root
        self.classes = classes
        self.img_dir = root / "images"
        self.lbl_dir = root / "labels"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.lbl_dir.mkdir(parents=True, exist_ok=True)

        self.last_capture = 0.0
        self.last_negative = 0.0
        self.count = len(list(self.img_dir.glob("*.jpg")))

        self._write_yaml()
        self.notes = root / "notes.csv"
        if not self.notes.exists():
            self.notes.write_text("file,n_boxes,max_conf,classes\n")
        print(f"[dataset] {root} ({self.count} existing images)")

    def _write_yaml(self):
        names = "\n".join(f"  {i}: {n}" for i, n in enumerate(self.classes))
        (self.root / "data.yaml").write_text(
            "# Edit train/val paths after splitting, then:\n"
            "#   yolo train model=yolo11s.pt data=data.yaml epochs=100 imgsz=640\n"
            "path: .\n"
            "train: images\n"
            "val: images\n"
            f"nc: {len(self.classes)}\n"
            f"names:\n{names}\n"
        )

    def maybe_capture(self, frame, detections, force=False, tag=""):
        now = time.time()
        labeled = [
            d for d in detections
            if d[1] >= cfg["dataset"]["label_thresh"] 
            and d[0] in cfg["dataset"]["coco_to_dataset"]
            and cfg["dataset"]["coco_to_dataset"][d[0]] < len(self.classes)
        ]

        if not force:
            if labeled:
                if now - self.last_capture < cfg["dataset"]["min_interval"]:
                    return None
            else:
                if now - self.last_negative < cfg["dataset"]["negative_every"]:
                    return None

        h, w = frame.shape[:2]
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = f"{stamp}-{self.count:05d}" + (f"-{tag}" if tag else "")

        cv2.imwrite(str(self.img_dir / f"{name}.jpg"), frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        lines = []
        for cls, conf, box, _ in labeled:
            did = cfg["dataset"]["coco_to_dataset"][cls]
            x1, y1, x2, y2 = box
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"{did} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # An empty .txt is a valid negative sample - YOLO expects the file
        # to exist even with no boxes.
        (self.lbl_dir / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

        max_conf = max((d[1] for d in labeled), default=0.0)
        cls_names = "|".join(sorted({self.classes[cfg["dataset"]["coco_to_dataset"][d[0]]] for d in labeled}))
        with self.notes.open("a") as f:
            f.write(f"{name}.jpg,{len(lines)},{max_conf:.3f},{cls_names}\n")

        self.count += 1
        if labeled:
            self.last_capture = now
        else:
            self.last_negative = now
        return name

class ClipRecorder:
    """Rolling pre-roll buffer + post-roll capture, written off the hot path.

    push() is called every loop iteration. trigger() marks 'something is
    happening'; recording continues until CLIP_POST_SECONDS pass with no
    further trigger, then the frames are encoded on a worker thread.
    """

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.pre = deque(maxlen=max(1, int(cfg["clips"]["pre_seconds"] * cfg["model"]["infer_fps"]) + 2))
        self.frames: list[np.ndarray] | None = None
        self.last_trigger = 0.0
        self.started = 0.0
        self.zone = None
        self.lock = threading.Lock()
        print(f"[clips] {root} ({len(list(root.glob('*.mp4')))} existing)")

    @staticmethod
    def _prep(frame):
        if cfg["clips"]["scale"] != 1.0:
            frame = cv2.resize(frame, None, fx=cfg["clips"]["scale"], fy=cfg["clips"]["scale"],
                               interpolation=cv2.INTER_AREA)
        return frame

    def push(self, frame):
        f = self._prep(frame)
        with self.lock:
            if self.frames is None:
                self.pre.append(f)
                return
            self.frames.append(f)
            now = time.time()
            over = now - self.started > cfg["clips"]["max_seconds"]
            idle = now - self.last_trigger > cfg["clips"]["post_seconds"]
            if over or idle:
                frames, zone, started = self.frames, self.zone, self.started
                self.frames = None
                self.pre.clear()
                threading.Thread(target=self._write,
                                 args=(frames, zone, started, over),
                                 daemon=True).start()

    def trigger(self, zone):
        with self.lock:
            self.last_trigger = time.time()
            if self.frames is None:
                self.frames = list(self.pre)
                self.started = self.last_trigger
                self.zone = zone or "frame"
                print(f"[clip] recording ({len(self.frames)} pre-roll frames)")
            elif zone and zone not in self.zone.split("+"):
                self.zone += f"+{zone}"

    def flush(self):
        """Write whatever is in progress - call on shutdown."""
        with self.lock:
            if self.frames:
                self._write(self.frames, self.zone, self.started, False)
                self.frames = None

    def _write(self, frames, zone, started, truncated):
        if len(frames) < 2:
            return
        h, w = frames[0].shape[:2]
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(started))
        path = self.root / f"{stamp}-{zone}.mp4"
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             cfg["clips"]["fps"], (w, h))
        if not vw.isOpened():
            print(f"[clip] FAILED to open writer for {path}")
            return
        for f in frames:
            vw.write(f)
        vw.release()
        secs = len(frames) / cfg["clips"]["fps"]
        print(f"[clip] saved {path} ({len(frames)} frames, {secs:.1f}s"
              + (", hit max length)" if truncated else ")"))
        
# ----------------------------------------------------------------------------
# Alerting
# ----------------------------------------------------------------------------


def fire_alert(frame, detections, zone_name, audio_uri=None):
    names = ", ".join(
        f"{cfg["classes"]["names"].get(c, c)} {p:.2f} (animal {a:.2f})"
        for c, p, _, a in detections
    )
    print(f"[ALERT] {time.strftime('%H:%M:%S')} - {names} in '{zone_name}'")

    Path(cfg["alert"]["snapshot_dir"]).mkdir(exist_ok=True)
    path = Path(cfg["alert"]["snapshot_dir"]) / f"{time.strftime('%Y%m%d-%H%M%S')}-{zone_name}.jpg"
    cv2.imwrite(str(path), frame)
    print(f"[ALERT] saved {path}")

    if audio_uri:
        try:
            print(f"[ALERT] Triggering Sonos at {cfg["audio"]["sonos_ip"]}...")
            sonos = SoCo(cfg["audio"]["sonos_ip"])
            original_volume = sonos.volume
            sonos.volume = cfg["audio"]["volume"]
            sonos.play_uri(audio_uri)
            time.sleep(cfg["audio"]["duration"])
            sonos.volume = original_volume
        except Exception as e:
            print(f"[ALERT] Error communicating with Sonos: {e}")


# ----------------------------------------------------------------------------
# Debug overlay
# ----------------------------------------------------------------------------

COLORS = {0: (60, 200, 255), 1: (255, 160, 60), 2: (200, 200, 200)}
clicked_points: list[tuple[int, int]] = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"[zone] point {len(clicked_points)}: ({x}, {y})")


ZONE_COLORS = [(0, 0, 220), (220, 0, 180), (0, 180, 220), (180, 120, 0)]


def draw_overlay(frame, detections, infer_ms, armed_streak, triggered, rois=None):
    out = frame.copy()
    h, w = out.shape[:2]

    if rois and any(r is not None for r in rois):
        mask = np.zeros((h, w), np.uint8)
        for r in rois:
            if r is None:
                continue
            cv2.rectangle(mask,
                          (int(r[0] * w), int(r[1] * h)),
                          (int(r[2] * w), int(r[3] * h)), 255, -1)
        dim = mask == 0
        out[dim] = (out[dim] * 0.45).astype(np.uint8)
        for r in rois:
            if r is None:
                continue
            cv2.rectangle(out,
                          (int(r[0] * w), int(r[1] * h)),
                          (int(r[2] * w), int(r[3] * h)), (0, 255, 255), 1)

    if cfg["zones"]:
        shade = out.copy()
        for i, (name, poly_pts) in enumerate(cfg["zones"].items()):
            if len(poly_pts) < 3:
                continue
            poly = np.array(poly_pts, dtype=np.int32)
            cv2.fillPoly(shade, [poly], ZONE_COLORS[i % len(ZONE_COLORS)])
        cv2.addWeighted(shade, 0.22, out, 0.78, 0, out)

        for i, (name, poly_pts) in enumerate(cfg["zones"].items()):
            if len(poly_pts) < 3:
                continue
            poly = np.array(poly_pts, dtype=np.int32)
            color = ZONE_COLORS[i % len(ZONE_COLORS)]
            cv2.polylines(out, [poly], True, color, 2)
            tx, ty = poly[:, 0].min(), poly[:, 1].min()
            cv2.putText(out, name, (int(tx) + 3, max(int(ty) - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    if clicked_points:
        for i, p in enumerate(clicked_points):
            cv2.circle(out, p, 4, (255, 0, 255), -1)
            cv2.putText(out, str(i), (p[0] + 6, p[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        if len(clicked_points) > 1:
            cv2.polylines(out, [np.array(clicked_points, np.int32)],
                          False, (255, 0, 255), 1)

    for cls, conf, box, acon in detections:
        if cls not in cfg["classes"]["draw"]:
            continue
        x1, y1, x2, y2 = box
        color = COLORS.get(cls, (180, 180, 180))
        ax, ay = anchor_point(box)
        zone = which_zone((ax, ay), cfg["zones"])
        inside = zone is not None

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{cfg["classes"]["names"].get(cls, cls)} {conf:.2f}"
        if cls in cfg["classes"]["target"]:
            label += f" | animal {acon:.2f}"
        if inside and zone != "frame":
            label += f" @{zone}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # anchor point - what the zone test actually uses
        cv2.circle(out, (ax, ay), 5, (0, 255, 0) if inside else (0, 0, 255), -1)
        cv2.circle(out, (ax, ay), 6, (0, 0, 0), 1)

    hud = f"{infer_ms:5.1f}ms  streak {armed_streak}/{cfg["history"]["hits"]}"
    cv2.rectangle(out, (0, 0), (230, 22), (0, 0, 0), -1)
    cv2.putText(out, hud, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 0) if triggered else (220, 220, 220), 1)
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def _mask_url(u: str) -> str:
    return re.sub(r"://([^:/@]+):([^@]+)@", r"://\1:***@", u)


def main():
    print(f"[init] loading {cfg["model"]["path"]}")
    det = Detector(cfg["model"]["path"], cfg["model"]["imgsz"])

    print(f"[init] url: {_mask_url(cfg["rtsp_url"])}")
    cam = FreshestFrame(cfg["rtsp_url"])
    
    local_ip = get_local_ip()
    print(f"[init] Starting audio server on {local_ip}:{cfg["audio"]["port"]}")
    audio_server = start_audio_server(cfg["audio"]["port"])
    audio_uri = f"http://{local_ip}:{cfg["audio"]["port"]}/{cfg["audio"]["file"]}"
    print(f"[init] Sonos deterrent URI configured as: {audio_uri}")

    while cam.read() is None:
        time.sleep(0.2)
    print("[init] first frame received")
    _f0 = cam.read()
    print(f"[init] connected: {_f0.shape[1]}x{_f0.shape[0]}")
    if _f0.shape[1] < 1280:
        print("[init] NOTE: low resolution - check the URL above is the "
              "main stream, and that nothing else holds the camera's "
              "main-stream slot.")

    for _zname, _zpoly in list(cfg["zones"].items()):
        if len(_zpoly) >= 3:
            fixed = order_polygon(_zpoly)
            if fixed != list(_zpoly):
                cfg["zones"][_zname] = fixed
                print(f"[zones] reordered '{_zname}' to remove self-intersection")

    check_zone_scale(cfg["zones"], _f0.shape)

    capture = None
    if cfg["dataset"]["enabled"]:
        capture = DatasetCapture(Path(cfg["dataset"]["dir"]), cfg["dataset"]["classes"])
        
    clips = ClipRecorder(Path(cfg["clips"]["dir"])) if cfg["clips"]["enabled"] else None

    latest_for_capture = {"frame": None, "dets": []}

    web = None
    if cfg["ui"]["web_port"]:
        def _do_capture(tag):
            f = latest_for_capture["frame"]
            if f is None or capture is None:
                return None
            return capture.maybe_capture(
                f, latest_for_capture["dets"], force=True, tag=tag)

        web = WebView(port=cfg["ui"]["web_port"], zones=cfg["zones"], quality=cfg["ui"]["web_quality"],
                      on_capture=_do_capture, order_points=order_polygon)

    if cfg["ui"]["show_window"]:
        cv2.namedWindow("catcam", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("catcam", on_mouse)

    history = deque(maxlen=cfg["history"]["len"])
    rr_index = 0
    last_rois = [None]
    last_alert: dict[str | None, float] = {}
    interval = 1.0 / cfg["model"]["infer_fps"]
    paused = False

    try:
        while True:
            t_loop = time.time()
            frame = cam.read()
            if frame is None:
                time.sleep(0.1)
                continue

            if not paused:
                t0 = time.time()

                if cfg["roi"]["mode"] == "zones":
                    rois = zone_rois(cfg["zones"], frame.shape)
                    current_roi_count = len(rois) if not cfg["roi"]["all_each_cycle"] else 1
                    
                    if cfg["roi"]["all_each_cycle"] or len(rois) == 1:
                        active = rois
                    else:
                        active = [rois[rr_index % len(rois)]]
                        rr_index += 1
                        
                    dets = merge_detections([det(frame, r) for _, r in active])
                    last_rois = [r for _, r in active]
                    
                elif cfg["roi"]["mode"] == "fixed":
                    current_roi_count = 1
                    dets = det(frame, cfg["roi"]["fixed"])
                    last_rois = [cfg["roi"]["fixed"]]
                    
                else:
                    current_roi_count = 1
                    dets = det(frame, None)
                    last_rois = [None]

                infer_ms = (time.time() - t0) * 1000

                dynamic_history_len = cfg["history"]["len"] * current_roi_count
                if history.maxlen != dynamic_history_len:
                    history = deque(history, maxlen=dynamic_history_len)
                    
                # person_zones = set()
                zone_hits = []
                hit_zone = None
                person_detected = False
                
                for d in dets:
                    if d[0] == 2:
                        person_detected = True
                        
                    if d[0] not in cfg["classes"]["target"]:
                        continue
                    z = which_zone(anchor_point(d[2]), cfg["zones"])
                    if z is not None:
                        zone_hits.append(d)
                        hit_zone = hit_zone or z

                if person_detected:
                    zone_hits = []
                    hit_zone = None
                
                history.append(bool(zone_hits))
                streak = sum(history)
                
                if capture is not None:
                    latest_for_capture["frame"] = frame
                    latest_for_capture["dets"] = dets
                    capture.maybe_capture(frame, dets)

                triggered = streak >= cfg["history"]["hits"] and bool(zone_hits)
                now = time.time()

                if triggered and clips is not None:
                    clips.trigger(hit_zone)
                if triggered and (now - last_alert.get(hit_zone, 0.0)) > cfg["alert"]["cooldown"]:
                    last_alert[hit_zone] = now
                    threading.Thread(
                        target=fire_alert,
                        args=(
                            frame.copy(), 
                            list(zone_hits), 
                            hit_zone, 
                            # audio_uri
                        ),
                        daemon=True,
                    ).start()
                    history.clear()

            view = draw_overlay(frame, dets, infer_ms, streak, triggered, last_rois)
            
            if clips is not None:
                clips.push(view if cfg["clips"]["overlay"] else frame)

            if web is not None:
                web.publish(view)
                web.set_status(
                    detections=[
                        {
                            "cls": cfg["classes"]["names"].get(c, str(c)),
                            "conf": round(p, 3),
                            "animal": round(a, 3),
                            "zone": which_zone(anchor_point(b), cfg["zones"]),
                            "target": c in cfg["classes"]["target"],
                            "box": list(b),
                        }
                        for c, p, b, a in dets
                    ],
                    infer_ms=round(infer_ms, 1),
                    streak=streak,
                    need=cfg["history"]["hits"],
                    triggered=triggered,
                    conf_thresh=cfg["model"]["conf_thresh"],
                )

            if cfg["ui"]["show_window"]:
                cv2.imshow("catcam", view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("s"):
                    Path(cfg["alert"]["snapshot_dir"]).mkdir(exist_ok=True)
                    p = Path(cfg["alert"]["snapshot_dir"]) / f"manual-{time.strftime('%Y%m%d-%H%M%S')}.jpg"
                    cv2.imwrite(str(p), view)
                    print(f"[snap] {p}")
                elif key == ord("n"):
                    if len(clicked_points) < 3:
                        print("[zone] need at least 3 points")
                    else:
                        name = input("zone name: ").strip() or f"zone{len(cfg["zones"]) + 1}"
                        cfg["zones"][name] = list(clicked_points)
                        clicked_points.clear()
                        print(f"[zone] stored '{name}' ({len(cfg["zones"][name])} pts)")
                elif key == ord("z"):
                    print("\nZONES = {")
                    for name, pts in cfg["zones"].items():
                        print(f"    {name!r}: {pts},")
                    if clicked_points:
                        print(f"    # unsaved: {clicked_points}")
                    print("}\n")
                elif key == ord("c"):
                    clicked_points.clear()
                    print("[zone] cleared pending points")
                elif key == ord("x"):
                    if cfg["zones"]:
                        last = list(cfg["zones"])[-1]
                        del cfg["zones"][last]
                        print(f"[zone] removed '{last}'")
                elif key == ord(" "):
                    paused = not paused

            sleep = interval - (time.time() - t_loop)
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        pass
    finally:
        if web is not None:
            web.stop()
            
        if clips is not None:
            clips.flush()
            
        cam.stop()
        if cfg["ui"]["show_window"]:
            cv2.destroyAllWindows()
        print("[exit] stopped")


if __name__ == "__main__":
    main()