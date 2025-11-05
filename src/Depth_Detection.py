import os
import cv2
import glob
import numpy as np
import torch
import pandas as pd
from transformers import pipeline
from PIL import Image

# --- DepthEstimator class ---
class DepthEstimator:
    def __init__(self, model_size='small', device=None):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device
        if self.device == 'mps':
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            self.pipe_device = 'cpu'
        else:
            self.pipe_device = self.device
        model_map = {
            'small': 'depth-anything/Depth-Anything-V2-Small-hf',
            'base': 'depth-anything/Depth-Anything-V2-Base-hf',
            'large': 'depth-anything/Depth-Anything-V2-Large-hf'
        }
        model_name = model_map.get(model_size.lower(), model_map['small'])
        self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.pipe_device)

    def estimate_depth(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        depth_result = self.pipe(pil_image)
        depth_map = depth_result["depth"]
        if isinstance(depth_map, Image.Image):
            depth_map = np.array(depth_map)
        elif isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.cpu().numpy()
        # Normalize relative depth 0-1
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max > depth_min:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        return depth_map

    def colorize_depth(self, depth_map, cmap=cv2.COLORMAP_INFERNO):
        depth_map_uint8 = (depth_map * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_map_uint8, cmap)

    def get_depth_in_bbox(self, depth_map, bbox, method='median'):
        x1, y1, x2, y2 = [int(c) for c in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth_map.shape[1]-1, x2)
        y2 = min(depth_map.shape[0]-1, y2)
        region = depth_map[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0
        if method == 'median':
            return float(np.median(region))
        elif method == 'mean':
            return float(np.mean(region))
        elif method == 'min':
            return float(np.min(region))
        else:
            return float(np.median(region))

# --- Paths ---
YOLO_OUTPUT_DIR = r"camera_0/Outputs/YOLO_Results"          # YOLO output folder
DEPTH_OUTPUT_DIR = r"camera_0/Outputs/Depth_Results"        # Depth output folder
os.makedirs(DEPTH_OUTPUT_DIR, exist_ok=True)

IMAGE_DIR = YOLO_OUTPUT_DIR                                      # Depth input = YOLO images
YOLO_CSV = os.path.join(YOLO_OUTPUT_DIR, "detections.csv")       # YOLO detection CSV

# --- Initialize depth estimator ---
depth_estimator = DepthEstimator(model_size='small')

# --- Load YOLO detections ---
if not os.path.exists(YOLO_CSV):
    print(f"YOLO CSV not found: {YOLO_CSV}")
    exit(1)
detections_df = pd.read_csv(YOLO_CSV)

# --- Process images ---
image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
if not image_paths:
    print(f"No images found in {IMAGE_DIR}")
    exit(1)

for img_path in image_paths:
    base_name = os.path.basename(img_path)
    try:
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to read {img_path}")
            continue

        # Depth estimation
        depth_map = depth_estimator.estimate_depth(image)
        depth_colored = depth_estimator.colorize_depth(depth_map)

        # Alpha blend for overlay
        alpha = 0.3
        overlay = cv2.addWeighted(image, alpha, depth_colored, 1-alpha, 0)

        # Filter detections for this image
        img_detections = detections_df[detections_df['image'] == base_name]

        # --- Draw depth values on overlay and depth-only images ---
        overlay_with_text = overlay.copy()
        depth_only_overlay = image.copy()

        for _, row in img_detections.iterrows():
            x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
            cls_name = row['class_name']
            depth_val = depth_estimator.get_depth_in_bbox(depth_map, [x1, y1, x2, y2])
            display_depth = depth_val * 100  # scaled for readability
            depth_text = f"{cls_name}: {display_depth:.2f}"

            # Clamp text coordinates
            text_x, text_y = x1, max(15, y1-5)

            # Draw bounding boxes
            cv2.rectangle(overlay_with_text, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(depth_only_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add black rectangle behind text for readability
            (w, h), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay_with_text, (text_x, text_y - h - 2), (text_x + w, text_y + 2), (0, 0, 0), -1)
            cv2.rectangle(depth_only_overlay, (text_x, text_y - h - 2), (text_x + w, text_y + 2), (0, 0, 0), -1)

            # Put text
            cv2.putText(overlay_with_text, depth_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(depth_only_overlay, depth_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # --- Save images ---
        cv2.imwrite(os.path.join(DEPTH_OUTPUT_DIR, f"depth_colored_{base_name}"), depth_colored)
        cv2.imwrite(os.path.join(DEPTH_OUTPUT_DIR, f"overlay_{base_name}"), overlay_with_text)
        cv2.imwrite(os.path.join(DEPTH_OUTPUT_DIR, f"depth_only_{base_name}"), depth_only_overlay)

        print(f"Processed {base_name}")

    except Exception as e:
        print(f"Error processing {base_name}: {e}")

print(f"\n✅ All depth images saved in {DEPTH_OUTPUT_DIR}")
