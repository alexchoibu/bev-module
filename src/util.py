import cv2
import sys

def get_cv_framework():
    # Get current operating system platform
    if sys.platform.startswith('win'):
        return cv2.CAP_DSHOW
    elif sys.platform.startswith('linux'):
        return cv2.CAP_V4L2
    elif sys.platform.startswith('darwin'):
        return cv2.CAP_AVFOUNDATION
    else:
        print("[ERROR] Unsupported OS platform")
        return None