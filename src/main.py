import cv2
from cv2_enumerate_cameras import enumerate_cameras
import numpy as np

import camera
import bev
import util
import inference
import depth

# Main function
def main():
    # Get current operating system platform
    cv_framework = util.get_cv_framework()
    if cv_framework is None:
        return

    # Identify each camera and create CameraThread instances for each
    threads = []
    for camera_info in enumerate_cameras(cv_framework):
        if camera_info.name.startswith("HD"):
            print("Found USB camera:", camera_info.name, "at index", camera_info.index)
            threads.append(camera.CameraThread(camera_info.index, camera_info.name))

    # Start all camera threads
    for t in threads:
        t.start()

    # Start YOLO inference
    inference_thread = inference.InferenceThread(threads, model_path="yolov8m.pt", device="cpu", imgsz=640, conf=0.35)
    inference_thread.start()

    # Start Depth Anything inference
    depth_thread = depth.DepthThread(threads, model_name="depth-anything/Depth-Anything-V2-Small-hf", device="mps", batch_time=0.3)
    depth_thread.start()

    try:
        while True:
            # draw detections onto camera frames
            for t in threads:
                if t.frame is not None:
                    # draw camera detections on the display frame copy
                    display_frame = t.frame.copy()
                    display_frame = t.draw_depth(display_frame)
                    t.draw_detections(display_frame, class_names=util.CLASSES)
                    # optionally save this modified frame to use in combine_frames
                    t.display_frame = display_frame

                while getattr(t, "depth_map", None) is None:
                    continue

            bev_frames = []
            for t in threads:
                bev_view, H, _, _ = bev.create_bev(t)
                if bev_view is not None and H is not None:
                    bev.draw_fov_lines(bev_view, t, t.detections, H, class_names=util.CLASSES)
                    bev_frames.append(bev_view)

            if bev_frames:
                birdseye_frame = np.mean(bev_frames, axis=0).astype(np.uint8) # Fix later to merge BEV frames
            else:
                birdseye_frame = np.zeros((1000, 1000, 3), dtype=np.uint8)

            all_frames = [getattr(t, "display_frame", t.frame) for t in threads] + [birdseye_frame]
            combined = camera.combine_frames(all_frames, layout="horizontal")  # "horizontal" "vertical" or "grid"
            if combined is not None:
                cv2.imshow("Multi-View + BEV", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') or key == ord('S'):  # Capture and save images
                [t.capture() for t in threads]
            elif key == ord('c') or key == ord('C'):  # Capture and save calibration images
                [t.capture(calib=True) for t in threads]
            elif key == ord('b') or key == ord('B'):  # Save BEV image
                if birdseye_frame is not None:
                    cv2.imwrite("bev/birdseye_view.png", birdseye_frame)
                    print("[INFO] Saved birdseye_view.png")
            elif key == ord('q'):  # Close display
                break
    except KeyboardInterrupt:
        pass
    finally:
        # Stop all threads cleanly
        inference_thread.stop()
        inference_thread.join()
        depth_thread.stop()
        depth_thread.join()
        for t in threads:
            t.stop()
        for t in threads:
            t.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()