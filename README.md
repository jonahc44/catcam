# catcam

RTSP camera + YOLO cat detection. When a cat is in a configured zone, it can snapshot, record a clip, and play a deterrent on a Sonos speaker.

## Setup

Python 3.11+ recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the example config and edit it:

```bash
cp config.example.yaml config.yaml
```

`config.yaml` is gitignored. Set at least:

- `rtsp_url` — your camera stream
- `model.path` — ONNX weights (see below)
- `audio.sonos_ip` — if you want the deterrent

### Model

The scripts load an ONNX file. Export one from a YOLO checkpoint:

```bash
yolo export model=yolo11n.pt format=onnx imgsz=640 opset=12
```

Point `model.path` at the resulting `.onnx` file.

The example config uses stock COCO class ids (`person` 0, `cat` 15, `dog` 16). If you train a custom model, update `classes` and `dataset.coco_to_dataset` to match.

### Camera

Put the full RTSP URL in `rtsp_url`. Prefer the camera's main stream. To check the feed without running detection:

```bash
python test_stream.py
```

That writes `frame.jpg` if it can read frames.

## Run

```bash
python catcam.py
```

Open `http://<this-machine>:8080` for the live view. Draw zone polygons there, then paste the coordinates into `config.yaml` under `zones:` and restart. With empty zones, the whole frame is active.

Optional: put `deterrent.wav` next to the scripts (or change `audio.file`) so Sonos can play it.
