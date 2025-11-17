import threading
import time
import numpy as np
import cv2
from transformers import pipeline
from PIL import Image

class DepthThread(threading.Thread):
    def __init__(self, camera_threads, model_name="depth-anything/Depth-Anything-V2-Small-hf",
                 device="mps", batch_time=0.1):
        """
        camera_threads: list of CameraThread instances
        model_name: any HF depth-estimation model (Depth Anything v2 is supported)
        device: 'cpu', 'cuda', or 'mps' (Apple Metal)
        batch_time: seconds to wait between inference passes
        """
        super().__init__(daemon=True)
        self.camera_threads = camera_threads
        self.model_name = model_name
        self.device = device
        self.batch_time = batch_time
        self.running = True

        print(f"[INFO] Loading Depth Anything via Transformers: {model_name} on {device}")
        self.pipe = pipeline("depth-estimation", model=model_name, device=device)

    def stop(self):
        self.running = False

    def run(self):
        print("[INFO] Starting DepthThread (Transformers pipeline)")
        while self.running:
            for cam in self.camera_threads:
                frame = getattr(cam, "frame", None)
                if frame is None:
                    continue

                # Convert to PIL Image
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                # Run depth estimation
                try:
                    result = self.pipe(pil_img)
                except Exception as e:
                    print(f"[WARN] Depth inference failed: {e}")
                    continue

                # Extract depth map
                depth = np.array(result["depth"], dtype=np.float32)

                # Resize to match original frame
                depth_resized = cv2.resize(depth, (frame.shape[1], frame.shape[0]))

                # Normalize for display
                depth_vis = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX)
                depth_vis = depth_vis.astype(np.uint8)

                # Store results
                cam.depth_map = depth_resized
                cam.depth_vis = depth_vis

            time.sleep(self.batch_time)
        print("[INFO] Exiting DepthThread")