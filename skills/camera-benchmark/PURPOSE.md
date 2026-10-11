# Purpose: Camera Benchmark Skill

- **Motivation:** Naive `time.time()` measurements wrap camera capture, decode, and model execution together, obscuring edge bottlenecks.
- **Goal:** Standardize stage-by-stage telemetry (capture, preprocess, inference, postprocess) so quantization speedups (FP32 vs INT8) can be faithfully measured.
- **Origin Wiki Pattern:** `wiki/logs.md` Iteration 1.
