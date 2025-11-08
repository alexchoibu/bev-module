import cv2
import numpy as np

def create_bev(cam_thread, floor_size=(8.0, 6.0), pixels_per_meter=200):
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
    floor_width, floor_height = floor_size  # in meters
    y_start = -floor_height / 4  # move floor patch back so BEV is more central
    output_w, output_h = int(floor_width*pixels_per_meter), int(floor_height*pixels_per_meter)
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

    return bev_frame, H

def project_point_to_bev(pt, H):
    """
    Project a single point from image coordinates to BEV coordinates using homography H.
    """
    pt_homog = np.array([pt[0], pt[1], 1.0])
    bev_pt_homog = H @ pt_homog
    bev_pt_homog /= bev_pt_homog[2]
    return (bev_pt_homog[0], bev_pt_homog[1])