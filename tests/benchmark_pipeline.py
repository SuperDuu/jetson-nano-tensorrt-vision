"""
Benchmark: Old CPU pipeline vs New GPU pipeline.
Compares preprocessing time, inference time, and total FPS.

Usage (on Jetson Nano):
    python3 tests/benchmark_pipeline.py --engine models/yolo_sp.engine
    python3 tests/benchmark_pipeline.py --engine models/yolo_sp.engine --frames 200
"""

import sys
import os
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def benchmark_cpu_pipeline(engine_path, imgsz, num_warmup, num_frames):
    """Benchmark original CPU preprocessing + TRTEngine."""
    from core.trt_engine import TRTEngine
    from core.utils import letterbox

    print("\n--- CPU Pipeline Benchmark ---")
    engine = TRTEngine(engine_path)

    # Generate test frame
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(num_warmup):
        canvas, s, (px, py) = letterbox(frame, (imgsz, imgsz))
        inp = canvas.transpose((2, 0, 1)).reshape((1, 3, imgsz, imgsz)).astype(np.float32) / 255.0
        engine.predict(inp)

    # Benchmark
    preprocess_times = []
    inference_times = []
    total_times = []

    for i in range(num_frames):
        t0 = time.time()
        canvas, s, (px, py) = letterbox(frame, (imgsz, imgsz))
        inp = canvas.transpose((2, 0, 1)).reshape((1, 3, imgsz, imgsz)).astype(np.float32) / 255.0
        t1 = time.time()
        engine.predict(inp)
        t2 = time.time()

        preprocess_times.append(t1 - t0)
        inference_times.append(t2 - t1)
        total_times.append(t2 - t0)

    return {
        'preprocess_ms': np.mean(preprocess_times) * 1000,
        'inference_ms': np.mean(inference_times) * 1000,
        'total_ms': np.mean(total_times) * 1000,
        'fps': 1000.0 / (np.mean(total_times) * 1000),
        'preprocess_std': np.std(preprocess_times) * 1000,
        'inference_std': np.std(inference_times) * 1000,
    }


def benchmark_gpu_pipeline(engine_path, imgsz, num_warmup, num_frames):
    """Benchmark GPU preprocessing + TRTEngineV2."""
    import pycuda.driver as cuda
    import pycuda.autoinit
    from core.trt_engine_v2 import TRTEngineV2
    from core.gpu_preprocess import GPUPreprocessor

    print("\n--- GPU Pipeline Benchmark ---")
    engine = TRTEngineV2(engine_path)
    preprocessor = GPUPreprocessor(imgsz, engine.stream)

    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(num_warmup):
        d_ptr, s, px, py = preprocessor(frame)
        engine.predict_from_device(d_ptr)

    # Benchmark
    preprocess_times = []
    inference_times = []
    total_times = []

    for i in range(num_frames):
        t0 = time.time()
        d_ptr, s, px, py = preprocessor(frame)
        engine.stream.synchronize()  # sync preprocess for timing
        t1 = time.time()
        engine.predict_from_device(d_ptr)
        t2 = time.time()

        preprocess_times.append(t1 - t0)
        inference_times.append(t2 - t1)
        total_times.append(t2 - t0)

    return {
        'preprocess_ms': np.mean(preprocess_times) * 1000,
        'inference_ms': np.mean(inference_times) * 1000,
        'total_ms': np.mean(total_times) * 1000,
        'fps': 1000.0 / (np.mean(total_times) * 1000),
        'preprocess_std': np.std(preprocess_times) * 1000,
        'inference_std': np.std(inference_times) * 1000,
    }


def benchmark_gpu_doublebuffer(engine_path, imgsz, num_warmup, num_frames):
    """Benchmark GPU pipeline with double-buffering (async)."""
    import pycuda.driver as cuda
    import pycuda.autoinit
    from core.trt_engine_v2 import TRTEngineV2
    from core.gpu_preprocess import GPUPreprocessor

    print("\n--- GPU Double-Buffer Benchmark ---")
    engine = TRTEngineV2(engine_path)
    preprocessor = GPUPreprocessor(imgsz, engine.stream)

    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(num_warmup):
        d_ptr, s, px, py = preprocessor(frame)
        engine.predict_from_device(d_ptr)

    # Simulate double-buffer: measure total per-frame time
    frame_times = []
    prev_output = None

    for i in range(num_frames):
        t0 = time.time()

        # Launch async GPU work
        d_ptr, s, px, py = preprocessor(frame)
        engine.infer_async(d_ptr)

        # Simulate CPU post-processing of previous frame (~1ms work)
        if prev_output is not None:
            _ = np.squeeze(prev_output[0])
            # Simulate NMS-like CPU work
            time.sleep(0.0005)

        # Sync
        prev_output = engine.sync_output()
        t1 = time.time()

        frame_times.append(t1 - t0)

    return {
        'total_ms': np.mean(frame_times) * 1000,
        'fps': 1000.0 / (np.mean(frame_times) * 1000),
        'total_std': np.std(frame_times) * 1000,
    }


def print_results(name, results):
    print("\n  [{}]".format(name))
    for key, val in sorted(results.items()):
        print("    {:<20s}: {:.2f}".format(key, val))


def main():
    parser = argparse.ArgumentParser(description="Benchmark CPU vs GPU pipeline")
    parser.add_argument('--engine', required=True, help='Path to .engine file')
    parser.add_argument('--imgsz', type=int, default=512, help='Input size')
    parser.add_argument('--warmup', type=int, default=50, help='Warmup frames')
    parser.add_argument('--frames', type=int, default=300, help='Benchmark frames')
    args = parser.parse_args()

    print("=" * 60)
    print("Pipeline Benchmark: {} (imgsz={})".format(args.engine, args.imgsz))
    print("Warmup: {} frames, Benchmark: {} frames".format(args.warmup, args.frames))
    print("=" * 60)

    cpu = benchmark_cpu_pipeline(args.engine, args.imgsz, args.warmup, args.frames)
    gpu = benchmark_gpu_pipeline(args.engine, args.imgsz, args.warmup, args.frames)
    dbuf = benchmark_gpu_doublebuffer(args.engine, args.imgsz, args.warmup, args.frames)

    print_results("CPU Pipeline", cpu)
    print_results("GPU Pipeline", gpu)
    print_results("GPU Double-Buffer", dbuf)

    print("\n" + "=" * 60)
    print("SPEEDUP SUMMARY:")
    print("  Preprocess: {:.1f}x faster (CPU {:.1f}ms -> GPU {:.1f}ms)".format(
        cpu['preprocess_ms'] / max(gpu['preprocess_ms'], 0.01),
        cpu['preprocess_ms'], gpu['preprocess_ms'],
    ))
    print("  Total FPS:  CPU {:.1f} -> GPU {:.1f} -> DoubleBuffer {:.1f}".format(
        cpu['fps'], gpu['fps'], dbuf['fps'],
    ))
    print("=" * 60)


if __name__ == "__main__":
    main()
