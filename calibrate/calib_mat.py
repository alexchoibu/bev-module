import numpy as np
import cv2 as cv
import glob
import argparse

parser = argparse.ArgumentParser(description='Camera Calibration using Chessboard Patterns with Fisheye Model')
parser.add_argument('--camid', type=int, default=0, help='Camera ID for saving calibration results')
args = parser.parse_args()

def combine_frames(frames, layout="horizontal"):
    valid_frames = [f for f in frames if f is not None]
    if not valid_frames:
        return None
    min_h = min(f.shape[0] for f in valid_frames)
    min_w = min(f.shape[1] for f in valid_frames)
    resized = [cv.resize(f, (min_w, min_h)) for f in valid_frames]
    return cv.hconcat(resized) if layout == "horizontal" else cv.vconcat(resized)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

# Prepare object points — note: pattern size = (columns, rows) of INNER CORNERS
# If your checkerboard has 8x10 squares, you should use (7,9)
pattern_size = (7, 9)  
square_size = 20.0  # mm

# object points in 3D
objp = np.zeros((np.prod(pattern_size), 1, 3), np.float32)
objp[:, 0, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image plane

images = glob.glob(f"camera_{args.camid}/calib_imgs/*.png")

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detect inner corners
    ret, corners = cv.findChessboardCorners(gray, pattern_size, None)

    if ret:
        # Refine corner locations
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        objpoints.append(objp)

        cv.drawChessboardCorners(img, pattern_size, corners2, ret)
        cv.imshow('Calib: ' + fname, img)
        cv.waitKey(0)

cv.destroyAllWindows()

# --- Correct fisheye calibration flags ---
flags = (cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
         cv.fisheye.CALIB_FIX_SKEW)

# --- Perform fisheye calibration ---
rms, K, D, rvecs, tvecs = cv.fisheye.calibrate(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None,
    flags=flags,
    criteria=criteria
)

print("RMS:", rms)
print("Camera matrix K:\n", K)
print("Distortion coefficients D:\n", D.ravel())

# --- Prepare undistortion maps ---
h, w = gray.shape[:2]
Knew = K.copy()
Knew[(0,1), (0,1)] = K[(0,1), (0,1)] * 0.8
map1, map2 = cv.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, (w,h), cv.CV_16SC2)

# --- Undistortion ---
for fname in images:
    img = cv.imread(fname)
    undistorted = cv.remap(img, map1, map2, cv.INTER_LINEAR)
    combined = combine_frames([img, undistorted])
    cv.imshow(fname, combined)
    cv.waitKey(0)
    wname = fname.split('/')[-1]
    cv.imwrite(f"camera_{args.camid}/calib_results/{wname}", undistorted)

cv.destroyAllWindows()

# --- Compute mean reprojection error ---
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error
print("Total reprojection error:", mean_error / len(objpoints))

# --- Save calibration results ---
fs = cv.FileStorage(f"camera_{args.camid}/calibration.yaml", cv.FILE_STORAGE_WRITE)
fs.write("camera_matrix", K)
fs.write("dist_coeffs", D)
fs.write("image_width", w)
fs.write("image_height", h)
try:
    fs.write("Knew", Knew)
except NameError:
    pass # Knew not defined, skip
fs.release()
print(f"Calibration results saved to camera_{args.camid}/calibration.yaml")