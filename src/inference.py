import threading
import time
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
    print("[WARN] ultralytics YOLO not available. Install ultralytics or adapt this file.")

class InferenceThread(threading.Thread):
    def __init__(self, camera_threads, model_path="yolov8n.pt", device=0,
                 imgsz=640, conf=0.35, batch_time=0.03):
        """
        camera_threads: list of CameraThread instances
        device: 0 for GPU, "cpu" for CPU
        imgsz: inference image size (square)
        batch_time: how long to sleep between inference cycles (s)
        """
        super().__init__(daemon=True)
        self.camera_threads = camera_threads
        self.model_path = model_path
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.batch_time = batch_time
        self.running = True

        if YOLO is None:
            raise RuntimeError("YOLO API not found. Install ultralytics or update inference code.")

        # Load single model instance
        print("[INFO] Loading YOLO model:", model_path)
        self.model = YOLO(model_path)  # loads to CPU by default
        try:
            # try to set device (ultralytics picks device in predict call but this helps)
            self.model.to(device)
        except Exception:
            pass

    def stop(self):
        self.running = False

    def run(self):
        print("[INFO] Starting InferenceThread")
        while self.running:
            frames = []
            cam_indices = []
            # Collect latest frames for each camera (skip None)
            for i, cam in enumerate(self.camera_threads):
                frame = getattr(cam, "frame", None)
                if frame is None:
                    continue
                # Optionally resize for inference to imgsz while keeping aspect ratio
                # Ultralytics can accept lists of arrays directly (it will resize internally),
                # but explicit resizing reduces memory / IO overhead.
                frames.append(frame)
                cam_indices.append(i)

            if frames:
                # Run batched inference on the list of frames
                # The ultralytics predict call accepts a list and handles batching
                try:
                    results = self.model.predict(frames,
                                                 imgsz=self.imgsz,
                                                 conf=self.conf,
                                                 device=self.device,
                                                 verbose=False,
                                                 save=False)
                except Exception as e:
                    print("[ERROR] Inference failed:", e)
                    results = []

                # results is a list-like with per-frame results
                # Store detection per camera thread in a simple structured format
                for idx, res in enumerate(results):
                    cam_idx = cam_indices[idx]
                    cam = self.camera_threads[cam_idx]
                    dets = []

                    # Ultralytics results: res.boxes.xyxy, res.boxes.conf, res.boxes.cls
                    try:
                        boxes = res.boxes.xyxy.cpu().numpy()  # Nx4
                        confs = res.boxes.conf.cpu().numpy()
                        clss = res.boxes.cls.cpu().numpy().astype(int)
                    except Exception:
                        # fallback: parse res.boxes if CPU tensors not available
                        boxes = []
                        confs = []
                        clss = []

                    for b, c, cl in zip(boxes, confs, clss):
                        x1, y1, x2, y2 = b
                        cx = float((x1 + x2) / 2.0)
                        cy = float((y1 + y2) / 2.0)
                        dets.append({
                            "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                            "center": [cx, cy],
                            "conf": float(c),
                            "cls": int(cl)
                        })

                    # Write detections atomically (simple assign; camera thread reads it)
                    cam.detections = dets

            # sleep a bit to avoid spinning; tuned to desired FPS / latency
            time.sleep(self.batch_time)

        print("[INFO] Exiting InferenceThread")