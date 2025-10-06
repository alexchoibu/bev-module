import cv2
from cv2_enumerate_cameras import enumerate_cameras

for camera_info in enumerate_cameras(cv2.CAP_AVFOUNDATION):
    print("Found camera:", camera_info.name, "at index", camera_info.index)

    cap = cv2.VideoCapture(camera_info.index) # Check the correct camera index

    # Set resolution to 12MP, assuming 4000x3000
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3000)

    # Set auto exposure
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 40)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")
        
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            print("Empty frame received")
        elif i >= 5:
            print("Camera", camera_info.index, "frame successfully read")
            cv2.imshow('Input', frame)
            cv2.waitKey(0)
            #cv2.imwrite("frame_" + str(i) + ".png", frame)
            cv2.destroyAllWindows()
            break

    cap.release()