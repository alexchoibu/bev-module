import cv2
import numpy as np

def create_bev(frames):
    """
    Simple example: stack resized frames in a top-down layout.
    In practice, apply perspective transforms to each frame
    to generate a true bird's-eye view.
    """

    # Ensure all frames are valid
    frames = [f for f in frames if f is not None]
    if not frames:
        return None

    # Resize frames to same shape
    min_h = min(f.shape[0] for f in frames)
    min_w = min(f.shape[1] for f in frames)
    resized = [cv2.resize(f, (min_w, min_h)) for f in frames]

    # Simple combination: average the images (placeholder for real top-down transform)
    bev = np.mean(resized, axis=0).astype(np.uint8)
    return bev