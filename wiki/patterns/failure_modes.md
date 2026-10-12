# Edge Vision Model Failure Modes and Mitigation Strategies

This document establishes the systematic failure mode analysis for EdgeSight on edge devices (CPU/NPU).

---

## 1. Failure Taxonomy & Root Causes

### Case 1: Low-Light Underexposure (Noise Floor Domination)
- **Symptom:** Detection recall drops significantly below 0.35 confidence threshold under <50 lux illumination.
- **Root Cause:** CMOS sensor amplifies analog gain resulting in high spatial Gaussian and salt-and-pepper noise, degrading edge gradients extracted by convolutional backbones.
- **Mitigation:** Integrate lightweight adaptive histogram equalization (CLAHE) or gamma correction in `capture.py` before inference; retrain or fine-tune with low-light augmentation.

### Case 2: Motion Blur from High-Velocity Dynamics
- **Symptom:** Bounding boxes flicker or vanish during rapid hand gestures or camera motion.
- **Root Cause:** Long exposure time per frame smears high-frequency spatial boundaries across multiple pixels, causing non-maximum suppression (NMS) confidence collapse.
- **Mitigation:** Implement temporal smoothing and state persistence (e.g., exponential moving average of bounding boxes across a 3-frame buffer or Kalman filter tracking).

### Case 3: Partial Boundary Occlusion
- **Symptom:** Severe truncation when objects or hands intersect the edge of the camera sensor frame.
- **Root Cause:** Standard anchor priors expect complete feature representations; truncated features fail the class confidence threshold.
- **Mitigation:** Apply boundary-aware penalty scaling or re-identification tracking that maintains object identity when an active bounding box exits the frame border.

### Case 4: Extreme Scale Variation (Sub-Pixel or Over-Saturated Objects)
- **Symptom:** Detection failure when target objects are farther than 3 meters (occupying <16x16 pixels) or excessively close (<15cm, occupying >80% of the field of view).
- **Root Cause:** Multi-scale feature pyramid (FPN/PAN) in lightweight YOLOv8n has limited receptive field resolution for tiny sub-pixel anchors.
- **Mitigation:** Calibrate operational working distance (0.4m to 1.5m); add high-resolution crop inference (SAHI - Slicing Aided Hyper Inference) if tiny object detection is mandated.

### Case 5: Background Clutter and Specular Glare False Positives
- **Symptom:** Occasional phantom detections on glossy furniture, monitors, or complex textile textures.
- **Root Cause:** High-frequency textures mimic edges of trained object categories (e.g., cups, phones, keyboards).
- **Mitigation:** Class-specific confidence filtering (e.g., `conf >= 0.45` for ambiguous classes) and multi-frame temporal voting (requiring an object to appear in at least 2 consecutive frames before trigger).

---

## 2. On-Device Inspection Protocol
When running the interactive webcam stream:
1. Press `s` to capture the current frame and append structured detection telemetry to `raw/snapshots/snapshots.jsonl`.
2. Inspect the saved snapshot in `raw/snapshots/` against the taxonomy above.
3. Record the failure category in this document for compounding architectural improvements.
