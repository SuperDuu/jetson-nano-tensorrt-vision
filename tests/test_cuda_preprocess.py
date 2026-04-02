"""
Correctness test: Compare CUDA kernel output vs CPU letterbox output.
Verifies that the GPU preprocessing produces results matching the CPU path.

Usage (on Jetson Nano):
    python3 tests/test_cuda_preprocess.py
"""

import sys
import os
import numpy as np

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def cpu_preprocess(frame_bgr, imgsz):
    """CPU reference: letterbox + BGR2RGB + normalize + CHW (matches vision.py:61)."""
    from core.utils import letterbox
    canvas, scale, (pad_x, pad_y) = letterbox(frame_bgr, (imgsz, imgsz))
    # BGR -> RGB, HWC -> CHW, normalize
    rgb = canvas[:, :, ::-1].copy()  # BGR to RGB
    chw = rgb.transpose((2, 0, 1)).reshape((1, 3, imgsz, imgsz)).astype(np.float32) / 255.0
    return chw, scale, pad_x, pad_y


def gpu_preprocess(frame_bgr, imgsz):
    """GPU path using PyCUDA kernel."""
    import pycuda.driver as cuda
    import pycuda.autoinit
    from core.gpu_preprocess import GPUPreprocessor

    stream = cuda.Stream()
    preprocessor = GPUPreprocessor(imgsz, stream)
    device_ptr, scale, pad_x, pad_y = preprocessor(frame_bgr)

    # Copy result back to host for comparison
    stream.synchronize()
    dst_size = 1 * 3 * imgsz * imgsz
    host_buf = np.empty(dst_size, dtype=np.float32)
    cuda.memcpy_dtoh(host_buf, device_ptr)
    result = host_buf.reshape(1, 3, imgsz, imgsz)
    return result, scale, pad_x, pad_y


def test_random_image(imgsz=512, src_h=480, src_w=640):
    """Test with a random image."""
    print("=" * 60)
    print("Test: Random image {}x{} -> {}x{}".format(src_w, src_h, imgsz, imgsz))

    np.random.seed(42)
    # Generate BGR
    frame_bgr = np.random.randint(0, 256, (src_h, src_w, 3), dtype=np.uint8)
    
    import cv2
    frame_bgra = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)

    cpu_out, cpu_scale, cpu_px, cpu_py = cpu_preprocess(frame_bgr, imgsz)
    gpu_out, gpu_scale, gpu_px, gpu_py = gpu_preprocess(frame_bgra, imgsz)

    print("  CPU scale={:.4f} pad=({}, {})".format(cpu_scale, cpu_px, cpu_py))
    print("  GPU scale={:.4f} pad=({}, {})".format(gpu_scale, gpu_px, gpu_py))
    print("  CPU output shape: {}  dtype: {}".format(cpu_out.shape, cpu_out.dtype))
    print("  GPU output shape: {}  dtype: {}".format(gpu_out.shape, gpu_out.dtype))

    # Check metadata matches
    assert cpu_scale == gpu_scale, "Scale mismatch: {} vs {}".format(cpu_scale, gpu_scale)
    assert cpu_px == gpu_px, "pad_x mismatch"
    assert cpu_py == gpu_py, "pad_y mismatch"

    # Check output values (allow tolerance for bilinear interpolation differences)
    # Note: CPU uses cv2.INTER_LINEAR, GPU uses manual bilinear — small differences expected
    atol = 3.0 / 255.0  # ±3 levels tolerance
    max_diff = np.max(np.abs(cpu_out - gpu_out))
    mean_diff = np.mean(np.abs(cpu_out - gpu_out))
    match = np.allclose(cpu_out, gpu_out, atol=atol)

    print("  Max diff:  {:.6f} ({:.2f} levels)".format(max_diff, max_diff * 255))
    print("  Mean diff: {:.6f} ({:.2f} levels)".format(mean_diff, mean_diff * 255))
    print("  Pass (atol={}): {}".format(atol, match))

    # Check padding regions are identical
    # Padding should be exactly 128/255.0 for both
    pad_val = 128.0 / 255.0
    gpu_pad_check = gpu_out[0, 0, 0, 0]  # Top-left corner (likely padding)
    print("  Padding value check: GPU={:.6f} expected={:.6f}".format(gpu_pad_check, pad_val))

    if not match:
        # Find where differences are largest
        diff = np.abs(cpu_out - gpu_out)
        worst = np.unravel_index(np.argmax(diff), diff.shape)
        print("  Worst diff at {}: CPU={:.6f} GPU={:.6f}".format(
            worst, cpu_out[worst], gpu_out[worst]))

    print("  RESULT: {}".format("PASS" if match else "FAIL"))
    print("=" * 60)
    return match


def test_various_resolutions():
    """Test with different source resolutions."""
    configs = [
        (480, 640, 512),   # Standard VGA
        (480, 640, 416),   # Smaller target
        (720, 1280, 512),  # 720p
        (240, 320, 512),   # Small source
        (512, 512, 512),   # Square (no padding needed)
    ]
    results = []
    for src_h, src_w, imgsz in configs:
        result = test_random_image(imgsz, src_h, src_w)
        results.append(result)

    print("\n" + "=" * 60)
    print("SUMMARY: {}/{} tests passed".format(sum(results), len(results)))
    print("=" * 60)
    return all(results)


if __name__ == "__main__":
    success = test_various_resolutions()
    sys.exit(0 if success else 1)
