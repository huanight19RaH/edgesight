# Wiki Evolution Log

## Iteration 1 — 2026-10-12
- **Action:** Project scaffold and architecture initialization.
- **Hypothesis:** Separating frame acquisition, inference, and telemetry into modular components minimizes coupling and allows direct integration of INT8 quantization in Phase 2.
- **Trace Reference:** `raw/traces/2026-10-12-init.md`
- **Findings:** OpenCV buffer lag must be prevented when running heavier models by decoupling frame capture in a separate thread if latency exceeds 30ms.

## Iteration 2 — 2026-10-21
- **Action:** Asynchronous threaded video ingestion and ONNX Runtime inference integration.
- **Implementation:**
  - Implemented `ThreadedVideoSource` in `capture.py` decoupling acquisition from inference via an atomic double-buffer and thread-safe lock.
  - Implemented `ONNXEdgeDetector` in `model.py` loading `weights/yolov8n.onnx` via `ONNX Runtime 1.24.4` with `CPUExecutionProvider`.
  - Added CLI flags `--backend {auto, onnx, pytorch}` and `--async-capture` in `pipeline.py`.
  - Added unit tests for threaded capture and ONNX detector in `tests/test_edgesight.py` (10/10 tests passing).
- **Benchmark Evidence:** Raw inference speedup ~1.48x (230ms PyTorch CPU -> 155ms ONNX Runtime CPU); zero camera frame dropping achieved in async mode.

