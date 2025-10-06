import cv2

for i in range(3):
    cap = cv2.VideoCapture(i) # Check the correct camera index

    # Set resolution to 12MP, assuming 4000x3000
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3000)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")
        
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            print("Empty frame received")
        else:
            print("Frame " + str(i) + " successfully read")
            cv2.imshow('Input', frame)
            cv2.waitKey(0)
            cv2.imwrite("frame_" + str(i) + ".png", frame)
            cv2.destroyAllWindows()
            break

    cap.release()