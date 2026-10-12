# EdgeSight: On-Device Edge Vision and Inference Telemetry

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)

**EdgeSight** is an engineering portfolio project developed by **Thai Hung Nguyen (RaH11)** focused on lightweight, real-time computer vision inference on edge architectures (CPU, NPU, and embedded SoCs). It bridges raw video streams with quantized model execution, temporal metrics, and rigorous per-stage telemetry (P50/P95 latency, throughput, and memory profiling).

---

## 1. Key Features

1. **Robust Video and Webcam Capture:** OpenCV-based acquisition with Windows DirectShow acceleration, synthetic stream generator for automated testing, and automatic buffer management.
2. **On-Device YOLOv8n Inference:** Lightweight detection backend pre-warmed for CPU execution, eliminating cold-start latency spikes.
3. **Microsecond-Level Telemetry HUD:** Real-time semi-transparent head-up display rendering instant FPS, per-stage latency breakdown (preprocess, inference, postprocess), hardware indicator, and active detection count.
4. **Statistical Benchmarking Suite:** Automated percentile calculation (Mean, Min, Max, P50, P90, P95, P99) and JSON report generation.
5. **Interactive Failure Inspection:** Interactive snapshot capture (`s` key) logging synchronized frames and structured detection metadata to `raw/snapshots/` for root-cause error analysis.
6. **WikiSkill Knowledge Compounding:** Integrated engineering wiki (`wiki/patterns/`) cataloging observed failure modes and concrete mitigation techniques.

---

## 2. Installation and Environment Setup

```bash
# 1. Clone repository
git clone https://github.com/huanight19RaH/edgesight.git
cd edgesight

# 2. Set up environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / WSL:
source .venv/bin/activate

# 3. Install dependencies in editable mode
pip install -r requirements.txt
pip install -e .
```

---

## 3. Quickstart and CLI Usage

### 3.1. Interactive Live Webcam Stream
Run the live inference loop with real-time HUD and detection boxes on camera index 0:
```bash
python -m edgesight.pipeline --source 0 --conf 0.35
```

**Interactive Controls During Live Execution:**
- `q` or `ESC`: Exit stream cleanly and display summary statistics.
- `s`: Capture frame snapshot and record detection metadata in `raw/snapshots/snapshots.jsonl`.
- `h`: Toggle telemetry HUD overlay.
- `b`: Toggle bounding boxes.

### 3.2. Headless Automated Benchmark
Run automated headless profiling without opening a GUI window (ideal for CI/CD or embedded boards):
```bash
# Benchmark using synthetic video stream
python -m edgesight.pipeline --source synthetic --benchmark --max-frames 100 --save-telemetry raw/telemetry_benchmark.json

# Benchmark using recorded video file
python -m edgesight.pipeline --source path/to/sample.mp4 --benchmark --max-frames 200
```

### 3.3. Video Export Mode
Process a video file or live stream and record annotated video output:
```bash
python -m edgesight.pipeline --source 0 --output raw/demo_recording.mp4
```

---

## 4. Empirical Performance Benchmarks

Measured on host system (Intel CPU, Windows 11, Python 3.12, PyTorch 2.10, resolution 640x480):

| Model Backend | Device | Resolution | Effective FPS | Latency Mean (ms) | Latency P50 (ms) | Latency P95 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8n** (FP32) | CPU | 640x480 | 13.5 FPS | 71.8 ms | 68.9 ms | 86.4 ms |
| **Baseline Canny** | CPU | 640x480 | 185.0 FPS | 4.8 ms | 4.2 ms | 6.5 ms |

*Note: Phase 2 will introduce post-training INT8 quantization (PTQ) via ONNX Runtime to target 30+ FPS on edge CPU.*

---

## 5. Project Architecture

```text
edgesight/
├── src/edgesight/
│   ├── __init__.py          # Package initialization
│   ├── capture.py           # VideoSource with DirectShow and synthetic generator
│   ├── model.py             # BaseEdgeDetector, YOLOEdgeDetector, BaselineDetector
│   ├── overlay.py           # Bounding boxes, labels, and semi-transparent HUD
│   ├── telemetry.py         # Statistical percentile calculator (P50, P95, P99)
│   └── pipeline.py          # CLI runner and interactive event loop
├── raw/                     # Execution traces, benchmark JSONs, snapshot archives
├── weights/                 # Model weights (ignored in Git, <5MB repo footprint)
├── wiki/                    # Persistent compounding knowledge base
│   ├── patterns/            # Systematic failure mode analysis
│   ├── logs.md              # Iteration evolution history
│   └── skill-impact.md      # Performance impact tracking
├── tests/                   # Automated unit test suite
│   └── test_edgesight.py    # 5 test suites covering capture, models, and telemetry
├── pyproject.toml           # PEP 621 package specification
└── requirements.txt         # Pinned runtime dependencies
```

---

## 6. Testing

Run the automated test suite:
```bash
python -m unittest discover tests/
# or directly:
python tests/test_edgesight.py
```

---

## 7. Roadmap

- [x] Phase 0 Day 1: Project scaffold, requirements, .gitignore, and WikiSkill architecture.
- [x] Phase 0 Day 2: Real-time webcam frame acquisition, Windows DirectShow support, and HUD overlay.
- [x] Phase 0 Day 3: Lightweight YOLOv8n baseline inference, confidence filtering, and unit tests.
- [x] Phase 0 Day 4: Microsecond-precision telemetry profiling (P50/P95 latency, throughput, JSON reports).
- [x] Phase 0 Day 5: 5 model failure modes taxonomy documented in `wiki/patterns/failure_modes.md`.
- [ ] Phase 1 (12/2026): Gesture temporal smoothing (Kalman / EMA filter) and webcam demonstration recording.
- [ ] Phase 2 (05/2027): ONNX export and Post-Training Quantization (PTQ INT8) for Orange Pi / RK3588 edge deployment.

---

## 8. Author

- **Thai Hung Nguyen (RaH11)**
- Email: hungnguyen.190206@gmail.com
- GitHub: https://github.com/huanight19RaH
