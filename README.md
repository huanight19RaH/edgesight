# EdgeSight: On-Device Edge Vision & Gesture Recognition

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)

**EdgeSight** is an engineering portfolio project focused on lightweight, real-time computer vision inference on edge devices (CPU/NPU/Edge SoC). It bridges raw video streams with quantized models, temporal smoothing, and rigorous telemetry (latency P50/P95, FPS, memory).

---

## 🚀 Key Features

1. **Lightweight Input Capture:** Resilient OpenCV-based capture from webcam (`/dev/video0` or camera index 0) or pre-recorded video files.
2. **Pretrained Baseline Inference:** End-to-end detection and classification using lightweight backbones.
3. **Telemetry & Benchmark Suite:** Microsecond-precision per-stage latency tracking (Preprocess, Inference, Postprocess, Total).
4. **WikiSkill Knowledge Evolution:** Structured persistent wiki (`wiki/`) capturing error patterns, failure modes, and mitigation strategies.
5. **Multi-Machine Synchronization:** Compact Git repository (<5MB) designed for effortless synchronization across home and work stations.

---

## 📦 Installation & Quickstart

```bash
# 1. Clone repository
git clone <repo-url>
cd edgesight

# 2. Create virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/WSL:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .

# 4. Run webcam demo
python -m edgesight.pipeline --source 0

# 5. Run video benchmark
python -m edgesight.pipeline --source path/to/sample.mp4 --benchmark
```

---

## 📁 Project Structure (WikiSkill 3-Tier Architecture)

```text
edgesight/
├── src/edgesight/           # Production source code
│   ├── capture.py           # Frame acquisition with buffer management
│   ├── model.py             # Pretrained model loader & inference wrappers
│   └── pipeline.py          # End-to-end execution pipeline & CLI
├── raw/                     # [Layer 1] Immutable execution traces & benchmark logs
├── wiki/                    # [Layer 2] Compounding knowledge base
│   ├── patterns/            # Root-cause analysis of failure cases
│   ├── logs.md              # Iteration evolution history
│   └── skill-impact.md      # Performance progression tracking
├── skills/                  # [Layer 3] Procedural reusable agent skills
│   └── camera-benchmark/   # Standardized benchmarking workflows
└── tests/                   # Automated pytest suite
```

---

## 📊 Roadmap (Q4 2026 – Q1 2027)

- [x] **Day 1:** Project scaffold, requirements, .gitignore, and WikiSkill architecture.
- [ ] **Day 2:** Real-time webcam frame acquisition & FPS overlay.
- [ ] **Day 3:** Lightweight PyTorch baseline inference & confidence filtering.
- [ ] **Day 4:** Telemetry profiling (P50/P95 latency, throughput, RAM).
- [ ] **Day 5:** 10 model failure cases inspection & error analysis in `wiki/patterns/`.
- [ ] **Phase 2 (12/2026):** Post-training quantization (PTQ INT8) and ONNX Runtime deployment.
