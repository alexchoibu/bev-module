from ultralytics import YOLO
import pandas as pd
import os

# --- CONFIG ---
IMAGE_DIR = r"camera_0/Input_Images"
OUTPUT_DIR = r"camera_0/Outputs/YOLO_Results"
MODEL_NAME = "yolov8m.pt"

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD MODEL ---
model = YOLO(MODEL_NAME)

# --- RUN DETECTION ---
results = model.predict(
    source=IMAGE_DIR,
    save=True,
    project=OUTPUT_DIR,
    name=".",       
    exist_ok=True,  
    conf=0.25
)

# --- COLLECT DETECTIONS ---
all_detections = []
for r in results:
    img_name = os.path.basename(r.path)
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()

    for box, conf, cls in zip(boxes, confs, classes):
        x1, y1, x2, y2 = map(int, box)
        all_detections.append({
            "image": img_name,
            "class_id": int(cls),
            "class_name": model.names[int(cls)],
            "confidence": round(float(conf), 3),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2
        })

# --- SAVE TO CSV ---
df = pd.DataFrame(all_detections)
csv_path = os.path.join(OUTPUT_DIR, "detections.csv")
df.to_csv(csv_path, index=False)

print("\n✅ YOLO detection complete!")
print(f"Images saved to: {OUTPUT_DIR}")
print(f"Detections CSV saved to: {csv_path}")
