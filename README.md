# EdgeSight: On-Device Edge Vision and Inference Telemetry

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8n-On--Device_Inference-00FFFF?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-EDBB00?style=flat-square)](LICENSE)

**EdgeSight** is an edge computer vision engineering system developed by **Thai Hung Nguyen** focused on lightweight, real-time computer vision inference on edge architectures (CPU, NPU, and embedded SoCs).

It bridges live video streams with quantized model execution, temporal metrics, and microsecond-precision per-stage telemetry (P50/P90/P95 latency, throughput, and memory profiling).

---

## 1. Key Engineering Highlights

1. **Robust Video and Webcam Capture**: OpenCV acquisition with DirectShow hardware acceleration on Windows, synthetic stream generator for deterministic headless testing, and automatic buffer flush.
2. **On-Device YOLOv8n Inference**: Lightweight detection backend pre-warmed for CPU execution, eliminating cold-start latency spikes.
3. **Microsecond-Precision Telemetry HUD**: Real-time semi-transparent head-up display rendering instant FPS, per-stage latency breakdown (preprocess, inference, postprocess), hardware indicator, and active detection count.
4. **Statistical Benchmarking Suite**: Automated percentile calculation (Mean, Min, Max, P50, P90, P95, P99) and structured JSON telemetry export.
5. **Interactive Failure Inspection**: Interactive snapshot capture logging synchronized frames and detection metadata to `raw/snapshots/` for root-cause analysis.
6. **Compounding Knowledge Base**: Integrated engineering wiki (`wiki/patterns/`) cataloging observed failure modes and concrete mitigation techniques.

---

## 2. Empirical Performance Benchmarks

Measured on host system (Intel CPU, Windows 11, Python 3.12, PyTorch, resolution 640x480):

| Model Backend | Device | Resolution | Effective FPS | Latency Mean | Latency P50 | Latency P95 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8n** (FP32) | CPU | 640x480 | **13.5 FPS** | 71.8 ms | 68.9 ms | 86.4 ms |
| **Baseline Canny** | CPU | 640x480 | **185.0 FPS** | 4.8 ms | 4.2 ms | 6.5 ms |

---

## 3. System Architecture

```text
edgesight/
├── src/edgesight/
│   ├── __init__.py          # Package initialization
│   ├── capture.py           # VideoSource with DirectShow and synthetic generator
│   ├── model.py             # BaseEdgeDetector, YOLOEdgeDetector, BaselineDetector
│   ├── overlay.py           # Bounding boxes, labels, and semi-transparent HUD
│   ├── telemetry.py         # Statistical percentile calculator (P50, P95, P99)
│   └── pipeline.py          # Core pipeline and interactive event loop
├── raw/                     # Execution traces, benchmark JSONs, snapshot archives
├── weights/                 # Model weights (lightweight repo footprint)
├── wiki/                    # Knowledge base and systematic failure mode analysis
├── tests/                   # Automated unit test suite
├── pyproject.toml           # PEP 621 package specification
└── requirements.txt         # Runtime dependencies
```

---

<details>
<summary><b>Tiếng Việt (Bản tóm tắt dự án)</b></summary>

<br>

### Tổng quan Dự án

**EdgeSight** là dự án kỹ thuật thị giác máy tính biên (Edge Computer Vision) do **Nguyễn Thái Hưng** phát triển, tập trung vào bài toán tối ưu hóa suy luận thời gian thực cho các thiết bị phần cứng giới hạn tài nguyên (Edge CPU, NPU, vi điều khiển SoC).

### Các tính năng kỹ thuật nổi bật

1. **Thu nhận luồng video tối ưu**: Hỗ trợ tăng tốc phần cứng DirectShow trên Windows và bộ sinh luồng video giả lập (Synthetic Stream) phục vụ kiểm thử tự động không cần camera vật lý.
2. **Suy luận On-Device với YOLOv8n**: Cơ chế Pre-warm mô hình giúp loại bỏ hiện tượng giật/trễ khung hình ban đầu (cold-start latency).
3. **Màn hình HUD hiển thị Telemetry thời gian thực**: Đo lường và trực quan hóa tức thì FPS, chi tiết độ trễ từng giai đoạn (tiền xử lý, suy luận mạng nơ-ron, hậu xử lý NMS).
4. **Bộ đo lường thống kê chuyên sâu**: Tự động tính toán các phân vị độ trễ (P50, P90, P95, P99) và xuất báo cáo JSON chuẩn hóa phục vụ phân tích hiệu năng.
5. **Cơ chế ghi nhận lỗi tương tác**: Chụp ảnh khung hình lỗi và siêu dữ liệu bounding box để phân tích nguyên nhân gốc rễ (Root-cause analysis).

### Kết quả đo kiểm thực tế (CPU Intel, 640x480)

- **YOLOv8n (FP32)**: Tốc độ 13.5 FPS, độ trễ trung vị P50: 68.9 ms, P95: 86.4 ms.
- **Baseline Canny**: Tốc độ 185 FPS, độ trễ P50: 4.2 ms.

</details>

---

## Author & Contact

- **Author**: Nguyen Thai Hung (Thai Hung Nguyen)
- **Role**: AI Engineering Undergraduate • FPT University Da Nang
- **Email**: [hungnguyen.190206@gmail.com](mailto:hungnguyen.190206@gmail.com)
- **GitHub**: [@huanight19RaH](https://github.com/huanight19RaH)
