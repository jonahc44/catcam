import os, time, cv2, yaml
from pathlib import Path
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|reorder_queue_size;0"
)
_cfg_path = Path(__file__).with_name("config.yaml")
if not _cfg_path.exists():
    _cfg_path.write_text(Path(__file__).with_name("config.example.yaml").read_text())
    print("[init] created config.yaml from config.example.yaml")
cfg = yaml.safe_load(_cfg_path.read_text())
URL = cfg["rtsp_url"]
cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)
print("opened:", cap.isOpened())
t = time.time(); n = 0
while n < 60:
    ok, frame = cap.read()
    if not ok: print("read failed"); break
    n += 1
print(f"{n} frames, {frame.shape}, {n/(time.time()-t):.1f} fps")
cv2.imwrite("frame.jpg", frame)