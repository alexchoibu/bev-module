import cv2
from cv2_enumerate_cameras import enumerate_cameras

import camera
import bev

# Main function
def main():
    # Identify each camera and create CameraThread instances for each
    threads = []
    for camera_info in enumerate_cameras(cv2.CAP_AVFOUNDATION):
        print("Found camera:", camera_info.name, "at index", camera_info.index)
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

            if cv2.waitKey(1) & 0xFF == ord('q'):
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