import cv2
import threading
import time
import numpy as np

# Thread class for each camera
class CameraThread(threading.Thread):
    def __init__(self, cam_id, name="CameraThread"):
        super().__init__()
        self.cam_id = cam_id
        self.name = f"{name}-{cam_id}"
        self.cap = cv2.VideoCapture(cam_id)
        self.frame = None
        self.running = True

        # Optional: set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open camera {cam_id}")

    def run(self):
        print(f"[INFO] Starting camera {self.cam_id}")
        while self.running:
            ret, frame = self.cap.read()
            if not self.running:
                break
            if ret:
                self.frame = frame
            else:
                time.sleep(0.1)
        print(f"[INFO] Exiting camera {self.cam_id} thread")
        self.cap.release()

    def stop(self):
        print(f"[INFO] Stopping camera {self.cam_id}")
        self.running = False

def combine_frames(frames, layout="horizontal"):
    """Combine multiple frames into a single image."""
    valid_frames = [f for f in frames if f is not None]
    if not valid_frames:
        return None

    # Resize all frames to the same shape
    min_h = min(f.shape[0] for f in valid_frames)
    min_w = min(f.shape[1] for f in valid_frames)
    resized = [cv2.resize(f, (min_w, min_h)) for f in valid_frames]

    if layout == "horizontal":
        combined = cv2.hconcat(resized)
    elif layout == "vertical":
        combined = cv2.vconcat(resized)
    else:  # 2x2 grid example
        while len(resized) < 4:
            blank = np.zeros_like(resized[0])
            resized.append(blank)
        top = cv2.hconcat(resized[:2])
        bottom = cv2.hconcat(resized[2:])
        combined = cv2.vconcat([top, bottom])
    return combined