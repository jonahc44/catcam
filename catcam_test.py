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
import time
import threading
from collections import deque
from pathlib import Path
import yaml

# Must be set before cv2 import touches the FFMPEG backend.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|reorder_queue_size;0"
)

import cv2
import numpy as np
import onnxruntime as ort

from webview import WebView

_cfg_path = Path(__file__).with_name("config.yaml")
if not _cfg_path.exists():
    _cfg_path.write_text(Path(__file__).with_name("config.example.yaml").read_text())
    print("[init] created config.yaml from config.example.yaml")
cfg = yaml.safe_load(_cfg_path.read_text())
cfg.setdefault("zones", {})
roi = cfg["roi"]["fixed"] if cfg["roi"]["mode"] == "fixed" else None

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
            keep = out[:, 4] > cfg["model"]["conf_thresh"]
            if not keep.any():
                return []
            sel = out[keep]
            lx1, ly1, lx2, ly2 = sel[:, 0], sel[:, 1], sel[:, 2], sel[:, 3]
            conf = sel[:, 4]
            cls = sel[:, 5].astype(int)
            acon = np.where(np.isin(cls, cfg["classes"]["animal"]), conf, 0.0)
            do_nms = False
        else:
            # (84, 8400) = 4 box coords + 80 class scores per grid cell.
            scores_all = out[4:]

            # Merge the animal classes into one score, so a cat split between
            # "cat" and "dog" isn't thresholded out of existence.
            animal_conf = scores_all[cfg["classes"]["animal"]].max(axis=0)
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
            do_nms = True

        # letterboxed crop space -> full frame space (shared by both formats)
        x1 = np.clip((lx1 - pad_x) / r, 0, cw - 1) + ox
        y1 = np.clip((ly1 - pad_y) / r, 0, ch - 1) + oy
        x2 = np.clip((lx2 - pad_x) / r, 0, cw - 1) + ox
        y2 = np.clip((ly2 - pad_y) / r, 0, ch - 1) + oy

        if do_nms:
            rects = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
            idxs = cv2.dnn.NMSBoxes(
                rects.tolist(), conf.tolist(), cfg["model"]["conf_thresh"], cfg["model"]["iou_thresh"]
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
        f"{cfg['classes']['names'].get(c, c)} {p:.2f} (animal {a:.2f})"
        for c, p, _, a in detections
    )
    print(f"[ALERT] {time.strftime('%H:%M:%S')} - {names} in '{zone_name}'")

    Path(cfg["alert"]["snapshot_dir"]).mkdir(exist_ok=True)
    path = Path(cfg["alert"]["snapshot_dir"]) / f"{time.strftime('%Y%m%d-%H%M%S')}-{zone_name}.jpg"
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

    if roi:
        h, w = out.shape[:2]
        rx1, ry1 = int(roi[0] * w), int(roi[1] * h)
        rx2, ry2 = int(roi[2] * w), int(roi[3] * h)
        dark = out.copy()
        cv2.rectangle(dark, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.rectangle(dark, (rx1, ry1), (rx2, ry2), (255, 255, 255), -1)
        mask = dark[:, :, 0] == 0
        out[mask] = (out[mask] * 0.45).astype(np.uint8)
        cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (0, 255, 255), 1)
        cv2.putText(out, f"ROI {rx2-rx1}x{ry2-ry1}", (rx1 + 3, ry1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

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
        label = f"{cfg['classes']['names'].get(cls, cls)} {conf:.2f}"
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

    hud = f"{infer_ms:5.1f}ms  streak {armed_streak}/{cfg['history']['hits']}"
    cv2.rectangle(out, (0, 0), (230, 22), (0, 0, 0), -1)
    cv2.putText(out, hud, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 0) if triggered else (220, 220, 220), 1)
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    print(f"[init] loading {cfg['model']['path']}")
    det = Detector(cfg["model"]["path"], cfg["model"]["imgsz"])

    print(f"[init] opening stream")
    cam = FreshestFrame(cfg["rtsp_url"])

    while cam.read() is None:
        time.sleep(0.2)
    print("[init] first frame received")

    web = None
    if cfg["ui"]["web_port"]:
        web = WebView(port=cfg["ui"]["web_port"], zones=cfg["zones"], quality=cfg["ui"]["web_quality"])

    if cfg["ui"]["show_window"]:
        cv2.namedWindow("catcam", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("catcam", on_mouse)

    history = deque(maxlen=cfg["history"]["len"])
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
                dets = det(frame, roi)
                infer_ms = (time.time() - t0) * 1000

                zone_hits = []
                hit_zone = None
                for d in dets:
                    if d[0] not in cfg["classes"]["target"]:
                        continue
                    z = which_zone(anchor_point(d[2]), cfg["zones"])
                    if z is not None:
                        zone_hits.append(d)
                        hit_zone = hit_zone or z

                history.append(bool(zone_hits))
                streak = sum(history)

                triggered = streak >= cfg["history"]["hits"] and bool(zone_hits)
                now = time.time()
                if triggered and (now - last_alert.get(hit_zone, 0.0)) > cfg["alert"]["cooldown"]:
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
                        name = input("zone name: ").strip() or f"zone{len(cfg['zones']) + 1}"
                        cfg["zones"][name] = list(clicked_points)
                        clicked_points.clear()
                        print(f"[zone] stored '{name}' ({len(cfg['zones'][name])} pts)")
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
        cam.stop()
        if cfg["ui"]["show_window"]:
            cv2.destroyAllWindows()
        print("[exit] stopped")


if __name__ == "__main__":
    main()