import torch
import cv2
import os
import numpy as np
from torchvision import transforms
from depth_anything_v2.dpt import DepthAnythingV2
import matplotlib.pyplot as plt

# --- Paths ---
IMAGE_DIR = "camera_0\Outputs\yolo_results\bev_detection2"
Depth_Output = "camera_0\Depth_Outputs"
OUTPUT_DIR = os.path.join(Depth_Output, "depth_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load Model ---
model = DepthAnythingV2.from_pretrained("LiheYoung/Depth-Anything-V2-ViT-Small")
model.eval()

# --- Transform Setup ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((518, 518)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Process Each Image ---
for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_tensor = transform(rgb).unsqueeze(0)

    # --- Predict Depth ---
    with torch.no_grad():
        depth = model(input_tensor).squeeze().cpu().numpy()

    # --- Normalize Depth Map ---
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    depth_colored = (plt.cm.magma(depth_norm)[:, :, :3] * 255).astype(np.uint8)

    # --- Save Outputs ---
    depth_gray_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_name)[0]}_depth.png")
    depth_color_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_name)[0]}_depth_colored.png")

    cv2.imwrite(depth_gray_path, (depth_norm * 255).astype(np.uint8))
    cv2.imwrite(depth_color_path, cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR))

    print(f"✅ Depth map saved for {img_name}")

print("\nAll images processed! Depth maps saved in:", OUTPUT_DIR)
