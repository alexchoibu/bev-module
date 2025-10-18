# Final Project: BEV Camera Module

This contains the Sprint Plan for the course.

# 📅 Sprint Release Calendar

| **Sprint** | **Dates** | **Goal / Deliverable** |
|-------------|------------|-------------------------|
| **2** | Oct 6 - Oct 20 | - Select and setup camera system<br>- Acquire data and begin processing via code<br>- Implement baseline image processing |
| **3** | Oct 21 - Nov 3 | - Implement depth estimation<br>- Turn camera perception into top down view |
| **4** | Nov 4 - Nov 17 | - Turn 2D into 3D view<br>- Video demo of the live system |
| **5** | Nov 18 - Dec 03 | - Refine model/system for accuracy and efficiency<br>- Document all findings and prepare report<br>- Create poster presentation |


---

## 🚀 Sprint 2 – Foundations & Setup (2 weeks)

**Goal:** Establish basic infrastructure for camera input and proof-of-concept image handling.

**Person A**
- Research & select camera setup (stereo cameras, multi-camera rig, or depth camera alternative)
- Acquire / simulate test video feeds
- Build code to capture and store frames (OpenCV or similar)

**Person B**
- Research algorithms for 3D bird’s-eye view transformations (homography, structure from motion, stereo depth, etc.)
- Implement baseline image preprocessing (rectification, calibration pipeline)
- Document setup, dependencies, and architecture in a shared repo

**Deliverable**
- Camera feed successfully captured  
- Frames can be processed and saved  
- Initial documentation of approach and design  

---

## 🧠 Sprint 3 – Depth Estimation Prototype (2 weeks)

**Goal:** Get a minimal viable system that can estimate depth from stereo/multi-camera input.

**Person A**
- Calibrate camera(s) and implement stereo rectification  
- Implement stereo depth estimation (block matching, Semi-Global Matching, or a lightweight deep-learning model)  
- Test on small datasets to verify disparity map quality  

**Person B**
- Start developing bird’s-eye projection (transform camera perspective to top-down)  
- Explore open-source libraries (e.g., Open3D, COLMAP, PyTorch/TF-based depth models) for generating point clouds  
- Benchmark latency to see if real-time streaming is feasible  

**Deliverable**
- Prototype producing disparity maps  
- Early experiments of top-down (bird’s-eye) projection from depth map  
- Notes on latency, bottlenecks, and feasibility of real-time pipeline  

---

## ⚙️ Sprint 4 – Integration & Real-Time Pipeline (2 weeks)

**Goal:** Fuse depth estimation and bird’s-eye view into a live-streaming system.

**Person A**
- Implement real-time streaming loop (capture → process → display)  
- Optimize depth computation (GPU acceleration, model pruning, or more efficient algorithm)  
- Ensure ~10–20 FPS pipeline on available hardware  

**Person B**
- Integrate 3D rendering of bird’s-eye view (OpenGL/Unity/Unreal/Three.js for visualization)  
- Add overlays (e.g., color-coded depth, object contours)  
- Validate against sample scenarios (objects at different depths/distances)  

**Deliverable**
- End-to-end working pipeline: live camera input → depth estimation → 3D bird’s-eye rendering  
- Preliminary video demo of the live system  

---

## 🏁 Sprint 5 – Polish, Validation & Presentation (2 weeks)

**Goal:** Refine system, validate performance, and prepare final deliverables (poster + demo).

**Person A**
- Evaluate accuracy: test with real-world objects of known distances  
- Conduct latency & performance profiling  
- Write up results (tables, charts, sample outputs)  

**Person B**
- Polish visualization for clarity (labels, color maps, smoother rendering)  
- Design & create poster presentation (motivation, method, results, visuals)  
- Prepare a live demo setup (scripted workflow for presentation)  

**Deliverable**
- Final prototype (demo-ready)  
- Evaluation results (depth accuracy, FPS, limitations)  
- Poster presentation (research-style, with visuals, diagrams, pipeline flow)  


