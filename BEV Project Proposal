# Bird’s Eye View (BEV) Camera Module Project Proposal

**Alexander Choi and Sohib Shafqat**  
**Fall 2025**

---

## Background / Previous Research

Recent advancements in computer vision and deep learning have significantly improved the capability of autonomous systems to perceive and understand their surroundings in three dimensions. Traditional single-camera perception systems are increasingly being replaced by multi-camera and surround-view setups, which enable richer spatial understanding and more accurate depth estimation — key requirements for autonomous navigation, driver assistance, and robotic perception.

**Singh (2023)** provides an extensive survey of surround-view, vision-based 3D detection methods tailored for autonomous driving applications. The paper highlights how integrating overlapping camera feeds enables full 360° environmental coverage, reducing blind spots and improving detection robustness in complex urban settings. Singh emphasizes that camera-based systems—compared to LiDAR or radar—offer cost-effective and scalable solutions, especially when coupled with geometric calibration and deep learning-based fusion techniques. The survey identifies key research challenges such as occlusion handling, multi-view feature fusion, and synchronization—topics that inform the architectural and algorithmic direction of our proposed system.

**Alaba and Ball (2023)** review deep learning-based image 3D object detection methods, categorizing them into monocular, stereo, and multi-view approaches. They emphasize the growing dominance of convolutional neural networks (CNNs) and transformer-based architectures for inferring depth and object geometry directly from 2D images. Their findings demonstrate that multi-view fusion and end-to-end learning pipelines can bridge the gap between RGB-based perception and real-world 3D spatial reasoning. This reinforces the feasibility of our approach to extract depth and spatial layout from synchronized camera streams without specialized 3D sensors.

**Ma et al. (2023)** provide a comprehensive survey of image-based 3D object detection frameworks across both monocular and multi-view paradigms. The authors compare geometric projection methods, depth map estimation, and voxel-based representations for reconstructing spatial scenes. A central takeaway is the increasing shift toward bird’s-eye-view (BEV) representations, which provide a unified top-down spatial layout that simplifies detection, tracking, and scene understanding.

---

## Project Mission

By integrating these principles, this project seeks to create a **3D Bird’s-Eye-View (BEV) Camera Module** that can enhance situational awareness in autonomous systems through real-time depth estimation and 3D scene reconstruction from multiple image streams.

---

## Target Users

The following table illustrates the target users that the BEV camera module will be applicable to:

| User | User Definition |
|------|-----------------|
| **Automotive Manufacturers** | Companies who will integrate a BEV system into vehicles (i.e. Tesla, Waymo, Cruise, etc.) |
| **Everyday Drivers** | Users who will use and receive information from a BEV system |

Additionally, the following table highlights user needs in the form of user stories that drive the design requirements for the BEV camera module:

| User | User Story |
|------|-------------|
| **Automotive Manufacturers** | As an AV manufacturer, I want an accurate 3D view of vehicles, pedestrians, and objects so that the vehicle plans safe maneuvers. |
| | As an AV manufacturer, I want an easily installable BEV camera system, so I can integrate BEV capabilities without a huge budget. |
| **Everyday Drivers** | As an everyday driver, I want to see a 360° view around my vehicle when driving, so that I can avoid hitting objects, pedestrians, or other vehicles. |

---

## Minimum Viable Product (MVP)

The **minimum viable product (MVP)** for this project will focus on demonstrating the fundamental capability of generating a 3D BEV representation from camera imagery, establishing the foundation for future multi-camera and real-time extensions. The MVP will not prioritize speed or system optimization; rather, it will validate the essential perception and transformation pipeline in a controlled environment.

At a minimum, the system will achieve the following:

### 1. Single-Camera Image Acquisition
- A camera module will capture static or recorded images of a known scene.  
- Calibration parameters (intrinsic and extrinsic) will be determined to map pixel coordinates to real-world spatial positions.

### 2. Bird’s-Eye-View Transformation
- Using geometric projection techniques, the captured image will be transformed into a top-down (bird’s-eye) view.  
- This transformation will serve as a spatial representation layer for object localization and visualization.

### 3. Depth and Object Estimation
- A deep learning or depth-from-mono model (e.g., **MiDaS** or **DPT**) will estimate relative depth across the scene.  
- Basic object detection (e.g., via pretrained **YOLO** or **DETR**) will be applied to identify and label objects within the transformed view.  
- Combined results will visualize the estimated object positions and distances in the BEV frame.

### 4. Visualization and Output
- The system will output a 2D top-down visualization showing detected objects and approximate depth contours.  
- Results can be static or frame-by-frame (non–real-time), with output displayed through a desktop interface or plotted image overlay.

---

## Success Criteria

The MVP will be considered successful if:

- A single camera input can be processed to produce a coherent BEV image.  
- Objects in the scene are detected and represented in appropriate positions in the BEV layout.  
- Approximate depth ordering (near vs. far) is visually distinguishable.

This MVP serves as a **proof of concept** that validates the feasibility of camera-based 3D environmental understanding using image transformation and perception models. Once functional, it provides a solid baseline for extending the system to **multi-camera fusion**, **synchronization**, and **real-time performance** in later development stages.

---

## References

- Singh, Apoorv. *"Surround-view vision-based 3d detection for autonomous driving: A survey."* Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.  
- Alaba, Simegnew Yihunie, and John E. Ball. *"Deep learning-based image 3-d object detection for autonomous driving."* IEEE Sensors Journal 23.4 (2023): 3378-3394.  
- Ma, Xinzhu, et al. *"3d object detection from images for autonomous driving: a survey."* IEEE Transactions on Pattern Analysis and Machine Intelligence 46.5 (2023): 3537-3556.
