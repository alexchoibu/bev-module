import cv2
import os

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3000)

cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 40)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

curr_dir = os.getcwd()
img_num = len(os.listdir(curr_dir + "/calib_imgs"))
    
for i in range(10):
    ret, frame = cap.read()
    if not ret:
        print("Empty frame received")
    elif i >= 5:
        print("Frame successfully read")
        cv2.imshow('Input', frame)
        cv2.waitKey(0)
        cv2.imwrite("calib_imgs/" + str(img_num) + ".png", frame)
        cv2.destroyAllWindows()
        break
        
cap.release()