import cv2
import threading
import time
import numpy as np
import os

# Thread class for each camera
class CameraThread(threading.Thread):
    def __init__(self, cam_id, name="CameraThread"):
        super().__init__()
        self.cam_id = cam_id
        self.name = f"{name}-{cam_id}"
        self.cap = cv2.VideoCapture(cam_id)
        self.frame = None
        self.running = True
        self.detections = []

        # Optional: set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open camera {cam_id}")

        # Load intrinsic camera parameters
        self.load_intrinsic()

        # Get extrinsic parameters
        self.get_extrinsic()

    def run(self):
        print(f"[INFO] Starting camera {self.cam_id}")
        while self.running:
            ret, frame = self.cap.read()
            if not self.running:
                break
            if ret:
                self.frame = frame
            else:
                time.sleep(0.1)
        print(f"[INFO] Exiting camera {self.cam_id} thread")
        self.cap.release()

    def stop(self):
        print(f"[INFO] Stopping camera {self.cam_id}")
        self.running = False

    def capture(self, calib=False):
        print(f"[INFO] Capturing frame from camera {self.cam_id}")

        if calib:
            image_dir = f"camera_{self.cam_id}/calib_imgs"
        else:
            image_dir = f"camera_{self.cam_id}/images"

        # List existing PNG files and extract numeric parts
        existing = []
        for f in os.listdir(image_dir):
            if f.endswith('.png') and f[:-4].isdigit():
                existing.append(int(f[:-4]))

        # Find the smallest available integer ≥ 0
        n = 0
        while n in existing:
            n += 1

        filename = f"{n}.png"
        path = os.path.join(image_dir, filename)

        success = cv2.imwrite(path, self.display_frame)

        if success:
            print(f"[INFO] Frame captured from camera {self.cam_id}")
        else:
            print(f"[ERROR] Failed to capture frame from camera {self.cam_id}")

    def load_intrinsic(self):
        """Load camera intrinsic parameters from YAML file."""
        fs = cv2.FileStorage(f"camera_{self.cam_id}/calibration.yaml", cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            print(f"[ERROR] Cannot open calibration file for camera {self.cam_id}")
            self.K = None
            self.dist_coeffs = None
            return

        self.K = fs.getNode("camera_matrix").mat()
        self.dist_coeffs = fs.getNode("dist_coeffs").mat()
        self.Knew = fs.getNode("Knew").mat()
        fs.release()

    def get_extrinsic(self, checkerboard_size=(7, 9), square_size=0.020):
        """
        Compute the extrinsic parameters (rotation and translation) of the camera
        using a checkerboard pattern placed on the floor.
        """
        # Check that intrinsic parameters are loaded
        if self.K is None or self.dist_coeffs is None:
            print(f"[ERROR] Intrinsic parameters not loaded for camera {self.cam_id}")
            return
        
        # Prepare 3D object points for checkerboard corners
        objp = np.zeros((checkerboard_size[0]*checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        objp *= square_size 

        # Capture frame to find corners
        ret, frame = self.cap.read()
        if not ret:
            print(f"[ERROR] Cannot read frame from camera {self.cam_id} for extrinsic calculation")
            return
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        if not ret:
            print(f"[ERROR] Cannot find checkerboard corners in camera {self.cam_id}")
            return
        
        # Refine corner locations
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        # Solve PnP to get rotation and translation vectors
        ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, self.K, self.dist_coeffs)
        if not ret:
            print(f"[ERROR] solvePnP failed for camera {self.cam_id}")
            return
        
        # Store extrinsic parameters
        self.R, _ = cv2.Rodrigues(rvecs)
        self.t = tvecs
        print(f"[INFO] Extrinsic parameters computed for camera {self.cam_id}")

        # Store a reference checkerboard point for later use
        self.checkerboard_img_point = corners2[0].ravel()
        self.checkerboard_world_point = objp[0].reshape(3)

        # Visualize detected corners
        #cv2.drawChessboardCorners(frame, checkerboard_size, corners2, ret)
        #cv2.imshow(f'Camera {self.cam_id} - Extrinsic', frame)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    def draw_detections(self, frame=None, class_names=None, color=(0,255,0)):
        if frame is None:
            frame = self.frame
        if frame is None:
            return None
        for d in getattr(self, "detections", []):
            x1,y1,x2,y2 = map(int, d["xyxy"])
            conf = d["conf"]
            cls = d["cls"]
            label = f"{cls}:{conf:.2f}" if class_names is None else f"{class_names.get(f"{cls}")}:{conf:.2f}"
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, max(20,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
        return frame
    
    def draw_depth(self, frame=None):
        if frame is None:
            frame = self.frame
        if frame is None or not hasattr(self, "depth_vis"):
            return None
        depth_colored = cv2.applyColorMap(self.depth_vis, cv2.COLORMAP_INFERNO)
        combined = cv2.addWeighted(frame, 0.3, depth_colored, 0.7, 0)
        return combined

def combine_frames(frames, layout="horizontal"):
    """Combine multiple frames into a single image."""
    valid_frames = [f for f in frames if f is not None]
    if not valid_frames:
        return None

    # Resize all frames to the same shape
    min_h = min(f.shape[0] for f in valid_frames)
    min_w = min(f.shape[1] for f in valid_frames)
    resized = [cv2.resize(f, (min_w, min_h)) for f in valid_frames]

    if layout == "horizontal":
        combined = cv2.hconcat(resized)
    elif layout == "vertical":
        combined = cv2.vconcat(resized)
    else:  # 2x2 grid example
        while len(resized) < 4:
            blank = np.zeros_like(resized[0])
            resized.append(blank)
        top = cv2.hconcat(resized[:2])
        bottom = cv2.hconcat(resized[2:])
        combined = cv2.vconcat([top, bottom])
    return combined