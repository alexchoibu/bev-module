import cv2
import numpy as np

FLOOR_SIZE = (8.0, 6.0)  # meters (width, height)
PIXELS_PER_METER = 200    # pixels per meter
Y_START_SCALE = 4.0  # extend floor patch beyond checkerboard

def create_bev(cam_thread):
    """
    Warp camera frame to BEV using extrinsics from checkerboard.
    Extends the floor patch beyond the checkerboard for a larger view.
    """
    if cam_thread.frame is None or not hasattr(cam_thread, "R") or not hasattr(cam_thread, "t"):
        return None

    # Undistort the frame
    frame_undistorted = cv2.fisheye.undistortImage(
        cam_thread.frame, cam_thread.K, cam_thread.dist_coeffs, Knew=cam_thread.Knew
    )

    # Define floor patch in world coordinates
    floor_width, floor_height = FLOOR_SIZE  # in meters
    y_start = -floor_height / Y_START_SCALE  # move floor patch back so BEV is more central
    output_w, output_h = int(floor_width*PIXELS_PER_METER), int(floor_height*PIXELS_PER_METER)
    floor_corners_world = np.array([
        [-floor_width/2, y_start, 0],
        [ floor_width/2, y_start, 0],
        [ floor_width/2, y_start + floor_height, 0],
        [-floor_width/2, y_start + floor_height, 0]
    ], dtype=np.float32)

    # Project corners into image
    rvec, _ = cv2.Rodrigues(cam_thread.R)
    img_corners, _ = cv2.projectPoints(floor_corners_world, rvec, cam_thread.t, cam_thread.Knew, np.zeros((4,1)))
    img_corners = img_corners.reshape(-1,2).astype(np.float32)

    # Define destination corners in BEV image
    dst_corners = np.array([
        [0, 0],
        [output_w-1, 0],
        [output_w-1, output_h-1],
        [0, output_h-1]
    ], dtype=np.float32)

    # Compute homography and warp
    H = cv2.getPerspectiveTransform(img_corners, dst_corners)
    bev_frame = cv2.warpPerspective(frame_undistorted, H, (output_w, output_h))

    return bev_frame, H, output_w, output_h

def project_point_to_bev(pt, H):
    """
    Project a single point from image coordinates to BEV coordinates using homography H.
    """
    pt_homog = np.array([pt[0], pt[1], 1.0])
    bev_pt_homog = H @ pt_homog
    bev_pt_homog /= bev_pt_homog[2]
    return (bev_pt_homog[0], bev_pt_homog[1])

def get_camera_apex_bev(cam_thread):
    """
    Returns the pixel coordinate in the BEV frame corresponding
    to the camera's real-world position (apex).
    
    Arguments:
        cam_thread: CameraThread instance with R, t
        ppm: pixels per meter
        floor_width: floor patch width in meters
        y_start: start offset of floor patch in meters
    """
    # Camera world position
    C_w = -cam_thread.R.T @ cam_thread.t  # shape (3,1)
    X, Y, Z = C_w.ravel()

    # Convert world X,Y → BEV pixels
    floor_width, floor_height = FLOOR_SIZE
    y_start = -floor_height / Y_START_SCALE
    bev_x = (X + floor_width / 2) * PIXELS_PER_METER
    bev_y = (Y - y_start) * PIXELS_PER_METER

    return (int(bev_x), int(bev_y))

def compute_fov_apex_world(cam_thread):
    """
    Compute the 3D world coordinates of the camera's FOV apex point.
    For this, we'll just use the camera optical center in world coordinates.
    """
    # Camera optical center in world coordinates
    C_w = -cam_thread.R.T @ cam_thread.t  # shape: (3,1)
    return C_w.reshape(3)  # returns [X, Y, Z] in world coords

# sample small patch mean to reduce noise
def sample_depth_mean(depth_map, px, py, k=1):
    h, w = depth_map.shape
    x0 = max(0, int(px)-k)
    x1 = min(w, int(px)+k+1)
    y0 = max(0, int(py)-k)
    y1 = min(h, int(py)+k+1)
    patch = depth_map[y0:y1, x0:x1]
    patch_valid = patch[np.isfinite(patch)]
    if patch_valid.size == 0:
        return float(patch.mean())
    return float(np.mean(patch_valid))

# fit polynomial mapping from depth_map value -> meters
def fit_depth_model(cam, degree=2, min_points=4):
    """
    Fit polynomial mapping D_meters = c0 + c1*z + c2*z^2 (degree default 2)
    using all checkerboard corners stored in cam.cb_img_points and cam.cb_world_points.

    Stores coefficients in cam.depth_poly as numpy array highest to lowest (np.polyfit order).
    Returns coeffs or None on failure.
    """
    if not hasattr(cam, "depth_map") or cam.depth_map is None:
        return None
    if not hasattr(cam, "cb_img_points") or cam.cb_img_points is None:
        return None
    
    img_pts = cam.cb_img_points  # Nx2
    world_pts = cam.cb_world_points  # Nx3

    # camera center in world coords
    C = (-cam.R.T @ cam.t).reshape(3)

    depth_vals = []
    true_dists = []
    for (px, py), world_pt in zip(img_pts, world_pts):
        px_i = int(round(px)); py_i = int(round(py))
        # clamp
        py_i = np.clip(py_i, 0, cam.depth_map.shape[0]-1)
        px_i = np.clip(px_i, 0, cam.depth_map.shape[1]-1)
        z_val = sample_depth_mean(cam.depth_map, px_i, py_i, k=1)
        if not np.isfinite(z_val) or z_val == 0:
            continue
        depth_vals.append(float(z_val))
        true_dists.append(float(np.linalg.norm(world_pt - C)))
    
    depth_vals = np.array(depth_vals, dtype=np.float64)
    true_dists = np.array(true_dists, dtype=np.float64)

    # Fit polynomial of requested degree
    coeffs = np.polyfit(depth_vals, true_dists, degree) # highest to lowest
    cam.depth_poly = coeffs
    print(f"[INFO] Fitted depth model (degree={degree}) coeffs:", coeffs)
    return coeffs

def predict_depth_from_model(cam, z):
    """
    Use cam.depth_poly (numpy array from np.polyfit highest to lowest) to predict meters.
    """
    if not hasattr(cam, "depth_poly") or cam.depth_poly is None:
        return None
    
    # polyval expects highest to lowest
    D = np.polyval(cam.depth_poly, z)
    return float(D)


# def bev_pixel_distance_to_m(cam, pixel_xy, checkerboard_img_point, checkerboard_world_point):
#     """
#     Convert a pixel in the camera image to approximate real-world distance (meters)
#     using Depth Anything relative depth map and a checkerboard reference point.

#     cam: CameraThread instance
#     pixel_xy: (x, y) pixel coordinates in camera image
#     checkerboard_img_point: (x, y) pixel coordinates of checkerboard point
#     checkerboard_world_point: 3D world coordinates of checkerboard point
#     """
#     # Compute real-world distance from camera to checkerboard reference
#     C = -cam.R.T @ cam.t  # Camera position
#     D_ref = np.linalg.norm(checkerboard_world_point - C.reshape(3))

#     # Get relative depth at checkerboard point
#     cb_x, cb_y = int(checkerboard_img_point[0]), int(checkerboard_img_point[1])
#     z_ref = cam.depth_map[cb_y, cb_x]

#     if z_ref == 0:
#         return None  # avoid divide-by-zero

#     # Invert depth so that larger depth map values correspond to closer objects
#     z_ref_inv = 1.0 / z_ref

#     # Compute scale factor to convert inverted depth map to meters
#     depth_scale = D_ref / z_ref_inv

#     # Sample depth at target pixel
#     px, py = int(pixel_xy[0]), int(pixel_xy[1])
#     z_obj = cam.depth_map[py, px]
#     if z_obj == 0:
#         return None

#     # Invert depth
#     z_obj_inv = 1.0 / z_obj

#     # Convert to meters
#     D_obj = z_obj_inv * depth_scale

#     return D_obj


def draw_fov_lines(bev_frame, cam_thread, detections, H, class_names=None):
    """
    Draw lines from the FOV intersection (apex) to the bottom-center of each detection.
    """
    if bev_frame is None or H is None:
        return
    
    # Ensure depth model fitted once
    fit_depth_model(cam_thread, degree=2, min_points=4)

    # Compute an apex in blank BEV space
    fov_apex_bev = get_camera_apex_bev(cam_thread)

    # Draw apex
    cv2.circle(bev_frame, fov_apex_bev, 5, (255,0,0), -1)
    cv2.putText(bev_frame, "Camera", (fov_apex_bev[0]+30, fov_apex_bev[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)

    # Draw each detection line
    for d in detections:
        x1, _, x2, y2 = d["xyxy"]
        bottom_center = np.array([(x1 + x2)/2.0, y2], dtype=np.float32)

        # Project detection to BEV
        bev_det_x, bev_det_y = project_point_to_bev(bottom_center, H)
        pt_bev = (int(bev_det_x), int(bev_det_y))

        # Compute real-world distance using depth map
        D_obj = None
        if getattr(cam_thread, "depth_map", None) is not None and getattr(cam_thread, "depth_poly", None) is not None:
            # sample depth (use small mean to reduce noise)
            img_px = int(np.clip(round(bottom_center[0]), 0, cam_thread.depth_map.shape[1]-1))
            img_py = int(np.clip(round(bottom_center[1]), 0, cam_thread.depth_map.shape[0]-1))
            z_obj = sample_depth_mean(cam_thread.depth_map, img_px, img_py, k=1)
            if np.isfinite(z_obj) and z_obj != 0:
                D_obj = predict_depth_from_model(cam_thread, z_obj)

        # Draw line from FOV apex to detection
        cv2.line(bev_frame, fov_apex_bev, pt_bev, (0, 255, 0), 2)

        # Draw detection point
        cv2.circle(bev_frame, pt_bev, 5, (0, 0, 255), -1)

        # Draw detection depth label
        cv2.putText(bev_frame, f"{class_names.get(f"{d["cls"]}")}:{D_obj:.2f}m", (pt_bev[0], pt_bev[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)