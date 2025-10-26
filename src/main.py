import cv2
from cv2_enumerate_cameras import enumerate_cameras

import camera
import bev
import util

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

    try:
        while True:
            frames = [t.frame for t in threads]
            birdseye_frame = bev.create_bev(frames)
            all_frames = frames + [birdseye_frame]
            combined = camera.combine_frames(all_frames, layout="grid")  # "horizontal" "vertical" or "grid"
            if combined is not None:
                cv2.imshow("Multi-View + BEV", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') or key == ord('S'):  # Capture and save images
                [t.capture() for t in threads]
            elif key == ord('c') or key == ord('C'):  # Capture and save calibration images
                [t.capture(calib=True) for t in threads]
            elif key == ord('q'):  # Close display
                break
    except KeyboardInterrupt:
        pass
    finally:
        # Stop all threads cleanly
        for t in threads:
            t.stop()
        for t in threads:
            t.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()