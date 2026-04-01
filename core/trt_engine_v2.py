"""
TensorRT Engine V2 with zero-copy GPU preprocessing support.
Extends TRTEngine with async inference from device pointers.

Compatible with Python 3.6+ / Jetson Nano.
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import logging

from .trt_engine import TRTEngine

logger = logging.getLogger(__name__)


class TRTEngineV2(TRTEngine):
    """
    Extended TRT engine supporting:
    - predict_from_device(): inference from a GPU device pointer (skip htod copy)
    - infer_async() / sync_output(): split async inference for double-buffering
    """

    def __init__(self, engine_path, max_batch_size=1):
        super(TRTEngineV2, self).__init__(engine_path, max_batch_size)
        self._input_nbytes = self.inputs[0]['host'].nbytes
        self.logger = logging.getLogger("{}.TRTEngineV2".format(__name__))
        self.logger.info("TRTEngineV2 initialized (input buffer: %d bytes)", self._input_nbytes)

    def predict_from_device(self, input_device_ptr):
        """
        Run inference from a device pointer (GPU -> GPU, no htod copy).

        Args:
            input_device_ptr: PyCUDA DeviceAllocation pointing to input tensor on GPU.

        Returns:
            List of numpy arrays (inference outputs).
        """
        with self._lock:
            self.cuda_ctx.push()
            try:
                # Device-to-device copy (much faster than host-to-device)
                cuda.memcpy_dtod_async(
                    self.inputs[0]['device'],
                    input_device_ptr,
                    self._input_nbytes,
                    self.stream,
                )

                # Execute inference
                self.context.execute_async_v2(
                    bindings=self.bindings,
                    stream_handle=self.stream.handle,
                )

                # Copy outputs device -> host
                for out in self.outputs:
                    cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

                self.stream.synchronize()

                return [
                    out['host'].copy().reshape(shape)
                    for out, shape in zip(self.outputs, self.output_shapes)
                ]
            finally:
                self.cuda_ctx.pop()

    def infer_async(self, input_device_ptr):
        """
        Launch async inference from device pointer. Does NOT synchronize.
        Call sync_output() after CPU work to collect results.

        Must be called from the same thread as sync_output().
        """
        self.cuda_ctx.push()

        # D2D copy input
        cuda.memcpy_dtod_async(
            self.inputs[0]['device'],
            input_device_ptr,
            self._input_nbytes,
            self.stream,
        )

        # Execute async
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle,
        )

        # Copy outputs D->H async
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

    def sync_output(self):
        """
        Synchronize the CUDA stream and return inference results.
        Must be called after infer_async().

        Returns:
            List of numpy arrays (inference outputs).
        """
        self.stream.synchronize()
        self.cuda_ctx.pop()

        return [
            out['host'].copy().reshape(shape)
            for out, shape in zip(self.outputs, self.output_shapes)
        ]
