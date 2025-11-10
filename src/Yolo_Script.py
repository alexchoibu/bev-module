from ultralytics import YOLO   
import pandas as pd            
import os                   

# --- CONFIGURATION ---
IMAGE_DIR = r"camera_0/Input_Images"           
OUTPUT_DIR = r"camera_0/Outputs/YOLO_Results" 
MODEL_NAME = "yolov8m.pt" # YOLOv8 medium model (you can try yolov8s.pt for smaller/faster)

# --- CREATE OUTPUT FOLDER ---
os.makedirs(OUTPUT_DIR, exist_ok=True)    

# --- LOAD YOLO MODEL ---
model = YOLO(MODEL_NAME)             

# --- RUN DETECTION ---
results = model.predict(
    source=IMAGE_DIR,      
    save=True,             # Save annotated images with bounding boxes
    project=OUTPUT_DIR,    
    name=".",              # Don't create extra nested folders; save images directly here
    exist_ok=True,         # Overwrite any existing files with the same name
    conf=0.25              # Minimum confidence threshold for detections (0.25 = 25%)
)

# --- COLLECT DETECTION RESULTS ---
all_detections = []  

for r in results:  # Loop through each image result
    img_name = os.path.basename(r.path)  # Get just the image file name
    boxes = r.boxes.xyxy.cpu().numpy()   # Bounding box coordinates: x1, y1, x2, y2
    confs = r.boxes.conf.cpu().numpy()   # Confidence score of each detection
    classes = r.boxes.cls.cpu().numpy()  # Class ID of each detected object

    # Loop through each detected object in the image
    for box, conf, cls in zip(boxes, confs, classes):
        x1, y1, x2, y2 = map(int, box)   # Convert box coordinates to integers
        all_detections.append({
            "image": img_name,                       # Image file name
            "class_id": int(cls),                    # Detected class ID
            "class_name": model.names[int(cls)],     # Detected class name
            "confidence": round(float(conf), 3),     # Confidence score (rounded)
            "x1": x1, "y1": y1, "x2": x2, "y2": y2   # Bounding box coordinates
        })

# --- SAVE DETECTIONS TO CSV ---
df = pd.DataFrame(all_detections)                      # Convert list of detections to a table (DataFrame)
csv_path = os.path.join(OUTPUT_DIR, "detections.csv")  
df.to_csv(csv_path, index=False)                       # Save CSV (without row numbers)

# --- PRINT SUMMARY ---
print("\n✅ YOLO detection complete!")                
print(f"Images saved to: {OUTPUT_DIR}")               
print(f"Detections CSV saved to: {csv_path}")        
