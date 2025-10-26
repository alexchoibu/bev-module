import os
import cv2
import glob
import numpy as np
import torch
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
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max > depth_min:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        return depth_map

    def colorize_depth(self, depth_map, cmap=cv2.COLORMAP_INFERNO):
        depth_map_uint8 = (depth_map * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_map_uint8, cmap)

# --- Paths ---
IMAGE_DIR = r"camera_0\Outputs\yolo_results\bev_detection2"
Depth_Output = r"camera_0\Depth_Outputs"
OUTPUT_DIR = os.path.join(Depth_Output, "depth_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Initialize depth estimator ---
depth_estimator = DepthEstimator(model_size='small')

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
        depth_map = depth_estimator.estimate_depth(image)
        depth_colored = depth_estimator.colorize_depth(depth_map)
        alpha = 0.3
        overlay = cv2.addWeighted(image, alpha, depth_colored, 1 - alpha, 0)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"depth_{base_name}"), depth_colored)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"overlay_{base_name}"), overlay)
        print(f"Processed {base_name}")
    except Exception as e:
        print(f"Error processing {base_name}: {e}")

print(f"All depth maps and overlays saved in {OUTPUT_DIR}")
