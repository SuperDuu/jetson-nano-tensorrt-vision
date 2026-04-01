"""
GPU Preprocessor wrapper for the CUDA letterbox kernel.
Manages device memory allocation and kernel launch.

Compatible with Python 3.6+ / Jetson Nano.
"""

import numpy as np
import pycuda.driver as cuda
import logging

from .cuda_preprocess import compile_preprocess_kernel

logger = logging.getLogger(__name__)

# Max source resolution supported (pre-allocate buffer)
MAX_SRC_H = 720
MAX_SRC_W = 1280
BACKGROUND_VALUE = 128


class GPUPreprocessor(object):
    """
    GPU-accelerated letterbox preprocessor.

    Uploads a BGR frame to GPU, runs a CUDA kernel that performs
    letterbox resize + BGR2RGB + normalize + HWC->CHW in one pass,
    and returns a device pointer to the output tensor.
    """

    def __init__(self, imgsz, stream, max_src_h=MAX_SRC_H, max_src_w=MAX_SRC_W):
        """
        Args:
            imgsz: Target square size (e.g. 512).
            stream: PyCUDA Stream to use (shared with TRT engine).
            max_src_h: Maximum source frame height for buffer pre-allocation.
            max_src_w: Maximum source frame width for buffer pre-allocation.
        """
        self.imgsz = imgsz
        self.stream = stream
        self.logger = logging.getLogger("{}.GPUPreprocessor".format(__name__))

        # Compile CUDA kernel
        self.logger.info("Compiling CUDA preprocess kernel...")
        self.kernel = compile_preprocess_kernel()
        self.logger.info("CUDA preprocess kernel compiled.")

        # Pre-allocate device buffers
        src_nbytes = max_src_h * max_src_w * 3  # uint8 BGR
        dst_nbytes = 1 * 3 * imgsz * imgsz * 4  # float32 CHW

        self.d_src = cuda.mem_alloc(src_nbytes)
        self.d_dst = cuda.mem_alloc(dst_nbytes)
        self.dst_nbytes = dst_nbytes

        # Pre-allocate pinned host buffer for fast upload
        self.h_src = cuda.pagelocked_empty(src_nbytes, np.uint8)

        # Kernel launch config
        self.block = (16, 16, 1)
        self.grid = (
            (imgsz + self.block[0] - 1) // self.block[0],
            (imgsz + self.block[1] - 1) // self.block[1],
        )

        self.logger.info(
            "GPUPreprocessor ready: imgsz=%d, grid=%s, block=%s",
            imgsz, self.grid, self.block
        )

    def _compute_letterbox_params(self, src_h, src_w):
        """Compute letterbox scale and padding (matches core/utils.py letterbox)."""
        r = min(float(self.imgsz) / src_h, float(self.imgsz) / src_w)
        new_w = int(round(src_w * r))
        new_h = int(round(src_h * r))
        pad_w = (self.imgsz - new_w) / 2.0
        pad_h = (self.imgsz - new_h) / 2.0
        pad_left = int(round(pad_w - 0.1))
        pad_top = int(round(pad_h - 0.1))
        return r, pad_left, pad_top

    def __call__(self, frame_bgr):
        """
        Preprocess a BGR frame on GPU.

        Args:
            frame_bgr: numpy array (H, W, 3) uint8 BGR image.

        Returns:
            (device_ptr, scale, pad_left, pad_top) where device_ptr points
            to the preprocessed float32 tensor [1, 3, imgsz, imgsz] on GPU.
        """
        src_h, src_w = frame_bgr.shape[:2]
        scale, pad_left, pad_top = self._compute_letterbox_params(src_h, src_w)
        pad_val = np.float32(BACKGROUND_VALUE / 255.0)

        # Upload source frame to GPU (async)
        nbytes = src_h * src_w * 3
        self.h_src[:nbytes] = frame_bgr.ravel()
        cuda.memcpy_htod_async(self.d_src, self.h_src[:nbytes], self.stream)

        # Launch kernel
        self.kernel(
            self.d_src, self.d_dst,
            np.int32(src_h), np.int32(src_w),
            np.int32(self.imgsz), np.int32(self.imgsz),
            np.float32(scale),
            np.int32(pad_left), np.int32(pad_top),
            pad_val,
            block=self.block, grid=self.grid, stream=self.stream,
        )

        return self.d_dst, scale, pad_left, pad_top
