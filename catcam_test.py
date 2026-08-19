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
Repeat for the next counter. Press 'z' and paste the output into ZONES.
"""

import os
import time
import threading
from collections import deque
from pathlib import Path

# Must be set before cv2 import touches the FFMPEG backend.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|reorder_queue_size;0"
)

import cv2
import numpy as np
import onnxruntime as ort

from webview import WebView

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

RTSP_URL = os.environ.get(
    "CATCAM_URL"
) + "stream1"

MODEL_PATH = os.environ.get("CATCAM_MODEL", "yolo11n.onnx")
IMGSZ = int(os.environ.get("CATCAM_IMGSZ", 640))

CONF_THRESH = 0.30
IOU_THRESH = 0.45
INFER_FPS = 5.0

# Region of interest, as fractions of the full frame: (x1, y1, x2, y2).
# Cropping to just the counters before inference is the single biggest win for
# distant cats - the animal fills far more of the model's input.
# None = use the whole frame.
ROI: tuple[float, float, float, float] | None = None

# COCO class ids we care about.
CLASS_NAMES = {0: "person", 15: "cat", 16: "dog", 21: "bear", 77: "teddy bear"}

# Classes merged into one "animal" decision. A cat split 0.28 cat / 0.26 dog
# fails both thresholds separately but passes comfortably when merged.
ANIMAL_CLASSES = [15, 16, 21, 77]

TARGET_CLASSES = set(ANIMAL_CLASSES)     # what triggers an alert
DRAW_CLASSES = {0, 15, 16, 21, 77}       # what gets drawn in the debug view

# Temporal filter: need N detections within the last M inference frames.
HISTORY_LEN = 5
HISTORY_HITS = 3

ALERT_COOLDOWN = 30.0          # seconds between alerts
SNAPSHOT_DIR = Path("snapshots")

# Active zones - a cat's anchor point inside ANY of these triggers.
# Points are (x, y) in ORIGINAL frame coordinates, not letterboxed.
# Empty list = whole frame active.
# Build these interactively: click the outline, press 'n' to name and store,
# repeat for the next counter, then press 'z' to print the whole block.
ZONES: dict[str, list[tuple[int, int]]] = {
    # "main_counter": [(120, 210), (300, 205), (310, 300), (110, 305)],
    # "island":       [(380, 190), (560, 195), (570, 280), (375, 275)],
}

SHOW_WINDOW = False        # local cv2 window; needs a display on the mini
WEB_PORT = 8080            # set to None to disable the MJPEG server
WEB_QUALITY = 70           # JPEG quality; lower = less bandwidth

# Intra-op threads. Cap this - oversubscription hurts on a 4-core box.
ORT_THREADS = 4

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------


class Detector:
    def __init__(self, model_path: str, imgsz: int = 640):
        so = ort.SessionOptions()
        so.intra_op_num_threads = ORT_THREADS
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sess = ort.InferenceSession(
            model_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.iname = self.sess.get_inputs()[0].name
        self.imgsz = imgsz

        # Two output layouts in the wild:
        #   (1, 84, 8400)  - v8/v11 style, raw grid predictions, needs NMS
        #   (1, 300, 6)    - v10/YOLO26 style, end-to-end, already decoded+sorted
        oshape = self.sess.get_outputs()[0].shape
        dims = [d if isinstance(d, int) else -1 for d in oshape]
        self.end2end = len(dims) == 3 and dims[2] == 6
        kind = "end-to-end (NMS-free)" if self.end2end else "grid + NMS"
        print(f"[model] {model_path} output {tuple(oshape)} -> {kind}")

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

        if self.end2end:
            # (300, 6) = [x1, y1, x2, y2, conf, class], already NMS'd and sorted.
            # No per-class score vector survives, so the animal score is just
            # the confidence when the predicted class is an animal.
            keep = out[:, 4] > CONF_THRESH
            if not keep.any():
                return []
            sel = out[keep]
            lx1, ly1, lx2, ly2 = sel[:, 0], sel[:, 1], sel[:, 2], sel[:, 3]
            conf = sel[:, 4]
            cls = sel[:, 5].astype(int)
            acon = np.where(np.isin(cls, ANIMAL_CLASSES), conf, 0.0)
            do_nms = False
        else:
            # (84, 8400) = 4 box coords + 80 class scores per grid cell.
            scores_all = out[4:]

            # Merge the animal classes into one score, so a cat split between
            # "cat" and "dog" isn't thresholded out of existence.
            animal_conf = scores_all[ANIMAL_CLASSES].max(axis=0)
            conf_all = np.maximum(animal_conf, scores_all.max(axis=0))

            keep = conf_all > CONF_THRESH
            if not keep.any():
                return []

            cx, cy, bw, bh = out[:4, keep]
            lx1, ly1 = cx - bw / 2, cy - bh / 2
            lx2, ly2 = cx + bw / 2, cy + bh / 2
            conf = conf_all[keep]
            acon = animal_conf[keep]
            cls = scores_all[:, keep].argmax(axis=0)
            do_nms = True

        # letterboxed crop space -> full frame space (shared by both formats)
        x1 = np.clip((lx1 - pad_x) / r, 0, cw - 1) + ox
        y1 = np.clip((ly1 - pad_y) / r, 0, ch - 1) + oy
        x2 = np.clip((lx2 - pad_x) / r, 0, cw - 1) + ox
        y2 = np.clip((ly2 - pad_y) / r, 0, ch - 1) + oy

        if do_nms:
            rects = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
            idxs = cv2.dnn.NMSBoxes(
                rects.tolist(), conf.tolist(), CONF_THRESH, IOU_THRESH
            )
            if len(idxs) == 0:
                return []
            idxs = np.array(idxs).flatten()
        else:
            idxs = np.arange(len(conf))

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


def anchor_point(box):
    """Bottom-center of the box - roughly where the animal contacts a surface."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, y2)


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


# ----------------------------------------------------------------------------
# Alerting
# ----------------------------------------------------------------------------


def fire_alert(frame, detections, zone_name):
    """Replace with the Sonos call. Runs on its own thread - keep it off the hot path."""
    names = ", ".join(
        f"{CLASS_NAMES.get(c, c)} {p:.2f} (animal {a:.2f})"
        for c, p, _, a in detections
    )
    print(f"[ALERT] {time.strftime('%H:%M:%S')} - {names} in '{zone_name}'")

    SNAPSHOT_DIR.mkdir(exist_ok=True)
    path = SNAPSHOT_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{zone_name}.jpg"
    cv2.imwrite(str(path), frame)
    print(f"[ALERT] saved {path}")

    # from soco import SoCo
    # SoCo("192.168.5.50").play_uri("http://.../get_off.mp3")


# ----------------------------------------------------------------------------
# Debug overlay
# ----------------------------------------------------------------------------

COLORS = {0: (200, 200, 200), 15: (60, 200, 255), 16: (255, 160, 60)}
clicked_points: list[tuple[int, int]] = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"[zone] point {len(clicked_points)}: ({x}, {y})")


ZONE_COLORS = [(0, 0, 220), (220, 0, 180), (0, 180, 220), (180, 120, 0)]


def draw_overlay(frame, detections, infer_ms, armed_streak, triggered):
    out = frame.copy()

    if ROI:
        h, w = out.shape[:2]
        rx1, ry1 = int(ROI[0] * w), int(ROI[1] * h)
        rx2, ry2 = int(ROI[2] * w), int(ROI[3] * h)
        dark = out.copy()
        cv2.rectangle(dark, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.rectangle(dark, (rx1, ry1), (rx2, ry2), (255, 255, 255), -1)
        mask = dark[:, :, 0] == 0
        out[mask] = (out[mask] * 0.45).astype(np.uint8)
        cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (0, 255, 255), 1)
        cv2.putText(out, f"ROI {rx2-rx1}x{ry2-ry1}", (rx1 + 3, ry1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    if ZONES:
        shade = out.copy()
        for i, (name, poly_pts) in enumerate(ZONES.items()):
            if len(poly_pts) < 3:
                continue
            poly = np.array(poly_pts, dtype=np.int32)
            cv2.fillPoly(shade, [poly], ZONE_COLORS[i % len(ZONE_COLORS)])
        cv2.addWeighted(shade, 0.22, out, 0.78, 0, out)

        for i, (name, poly_pts) in enumerate(ZONES.items()):
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
        if cls not in DRAW_CLASSES:
            continue
        x1, y1, x2, y2 = box
        color = COLORS.get(cls, (180, 180, 180))
        ax, ay = anchor_point(box)
        zone = which_zone((ax, ay), ZONES)
        inside = zone is not None

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES.get(cls, cls)} {conf:.2f}"
        if cls in TARGET_CLASSES:
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

    hud = f"{infer_ms:5.1f}ms  streak {armed_streak}/{HISTORY_HITS}"
    cv2.rectangle(out, (0, 0), (230, 22), (0, 0, 0), -1)
    cv2.putText(out, hud, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 0) if triggered else (220, 220, 220), 1)
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    print(f"[init] loading {MODEL_PATH}")
    det = Detector(MODEL_PATH, IMGSZ)

    print(f"[init] opening stream")
    cam = FreshestFrame(RTSP_URL)

    while cam.read() is None:
        time.sleep(0.2)
    print("[init] first frame received")

    web = None
    if WEB_PORT:
        web = WebView(port=WEB_PORT, zones=ZONES, quality=WEB_QUALITY)

    if SHOW_WINDOW:
        cv2.namedWindow("catcam", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("catcam", on_mouse)

    history = deque(maxlen=HISTORY_LEN)
    last_alert: dict[str | None, float] = {}
    interval = 1.0 / INFER_FPS
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
                dets = det(frame, ROI)
                infer_ms = (time.time() - t0) * 1000

                zone_hits = []
                hit_zone = None
                for d in dets:
                    if d[0] not in TARGET_CLASSES:
                        continue
                    z = which_zone(anchor_point(d[2]), ZONES)
                    if z is not None:
                        zone_hits.append(d)
                        hit_zone = hit_zone or z

                history.append(bool(zone_hits))
                streak = sum(history)

                triggered = streak >= HISTORY_HITS and bool(zone_hits)
                now = time.time()
                if triggered and (now - last_alert.get(hit_zone, 0.0)) > ALERT_COOLDOWN:
                    last_alert[hit_zone] = now
                    threading.Thread(
                        target=fire_alert,
                        args=(frame.copy(), list(zone_hits), hit_zone),
                        daemon=True,
                    ).start()
                    history.clear()

            view = draw_overlay(frame, dets, infer_ms, streak, triggered)

            if web is not None:
                web.publish(view)
                web.set_status(
                    detections=[
                        {
                            "cls": CLASS_NAMES.get(c, str(c)),
                            "conf": round(p, 3),
                            "animal": round(a, 3),
                            "zone": which_zone(anchor_point(b), ZONES),
                            "target": c in TARGET_CLASSES,
                            "box": list(b),
                        }
                        for c, p, b, a in dets
                    ],
                    infer_ms=round(infer_ms, 1),
                    streak=streak,
                    need=HISTORY_HITS,
                    triggered=triggered,
                    conf_thresh=CONF_THRESH,
                )

            if SHOW_WINDOW:
                cv2.imshow("catcam", view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("s"):
                    SNAPSHOT_DIR.mkdir(exist_ok=True)
                    p = SNAPSHOT_DIR / f"manual-{time.strftime('%Y%m%d-%H%M%S')}.jpg"
                    cv2.imwrite(str(p), view)
                    print(f"[snap] {p}")
                elif key == ord("n"):
                    if len(clicked_points) < 3:
                        print("[zone] need at least 3 points")
                    else:
                        name = input("zone name: ").strip() or f"zone{len(ZONES) + 1}"
                        ZONES[name] = list(clicked_points)
                        clicked_points.clear()
                        print(f"[zone] stored '{name}' ({len(ZONES[name])} pts)")
                elif key == ord("z"):
                    print("\nZONES = {")
                    for name, pts in ZONES.items():
                        print(f"    {name!r}: {pts},")
                    if clicked_points:
                        print(f"    # unsaved: {clicked_points}")
                    print("}\n")
                elif key == ord("c"):
                    clicked_points.clear()
                    print("[zone] cleared pending points")
                elif key == ord("x"):
                    if ZONES:
                        last = list(ZONES)[-1]
                        del ZONES[last]
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
        cam.stop()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()
        print("[exit] stopped")


if __name__ == "__main__":
    main()