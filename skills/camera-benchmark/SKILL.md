---
name: camera-benchmark
description: Standardized procedure to benchmark on-device camera inference latency and throughput.
---

# Camera Benchmark Procedure

1. Verify camera hardware is available or prepare a fixed evaluation video sequence (e.g. `data/benchmark_sample.mp4`).
2. Run benchmark in headless mode for 100 consecutive frames:
   ```bash
   python -m edgesight.pipeline --source data/benchmark_sample.mp4 --benchmark --max-frames 100
   ```
3. Record P50, P95 latency (ms), average FPS, and peak RAM consumption.
4. If inference latency exceeds 40ms, inspect OpenCV frame queue and record bottleneck in `wiki/patterns/`.
