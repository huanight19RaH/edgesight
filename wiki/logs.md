# Wiki Evolution Log

## Iteration 1 — 2026-10-12
- **Action:** Project scaffold and architecture initialization.
- **Hypothesis:** Separating frame acquisition, inference, and telemetry into modular components minimizes coupling and allows direct integration of INT8 quantization in Phase 2.
- **Trace Reference:** `raw/traces/2026-10-12-init.md`
- **Findings:** OpenCV buffer lag must be prevented when running heavier models by decoupling frame capture in a separate thread if latency exceeds 30ms.
