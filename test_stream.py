import os, time, cv2
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|reorder_queue_size;0"
)
URL = "rtsp://camera:Op200143@192.168.5.207:554/stream2"
cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)
print("opened:", cap.isOpened())
t = time.time(); n = 0
while n < 60:
    ok, frame = cap.read()
    if not ok: print("read failed"); break
    n += 1
print(f"{n} frames, {frame.shape}, {n/(time.time()-t):.1f} fps")
cv2.imwrite("frame.jpg", frame)