from ultralytics import YOLO
import pandas as pd
import os

# --- CONFIG ---
IMAGE_DIR = "camera_0\images"
Output_Folder = "camera_0\Outputs"
OUTPUT_DIR = os.path.join(Output_Folder, "yolo_results")
MODEL_NAME = "yolov8m.pt"  # you can also try yolov8s.pt or yolov8m.pt for more accuracy

# --- LOAD MODEL ---
model = YOLO(MODEL_NAME)

# --- RUN DETECTION ---
results = model.predict(
    source=IMAGE_DIR,
    save=True,
    project=OUTPUT_DIR,
    name="bev_detection",
    conf=0.25  # confidence threshold (you can adjust 0.25–0.5)
)

# --- COLLECT DETECTIONS ---
all_detections = []

for r in results:
    img_name = os.path.basename(r.path)
    boxes = r.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 in pixels
    confs = r.boxes.conf.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()

    for box, conf, cls in zip(boxes, confs, classes):
        x1, y1, x2, y2 = box
        all_detections.append({
            "image": img_name,
            "class_id": int(cls),
            "class_name": model.names[int(cls)],
            "confidence": round(float(conf), 3),
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2)
        })

# --- SAVE TO CSV ---
df = pd.DataFrame(all_detections)
csv_path = os.path.join(OUTPUT_DIR, "bev_detection", "detections.csv")
df.to_csv(csv_path, index=False)

print(f"\n✅ YOLO detection complete!")
print(f"Annotated images saved to: {os.path.join(OUTPUT_DIR, 'bev_detection', 'predict')}")
print(f"Detections CSV saved to: {csv_path}")
