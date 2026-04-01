"""
RobotVision V2 with GPU-accelerated preprocessing.
Uses GPUPreprocessor + TRTEngineV2 for zero-copy inference pipeline.

Compatible with Python 3.6+ / Jetson Nano.
"""

import cv2
import numpy as np
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional

from .vision import (
    RobotVision, DetectedObject,
    DEFAULT_CLASS_ID, DEFAULT_INPUT_SIZE, DEFAULT_CONF_THRESHOLD,
)
from .trt_engine_v2 import TRTEngineV2
from .gpu_preprocess import GPUPreprocessor

logger = logging.getLogger(__name__)


class RobotVisionV2(RobotVision):
    """
    GPU-accelerated vision: preprocessing on GPU, zero-copy inference.

    Inherits post-processing, EMA smoothing, and Kalman filter from RobotVision.
    Adds:
        - predict_gpu(): full GPU pipeline (preprocess + infer + postprocess)
        - launch_inference() / collect_and_postprocess(): async double-buffer API
    """

    def __init__(self, model_path, imgsz=DEFAULT_INPUT_SIZE,
                 class_id=DEFAULT_CLASS_ID, device="GPU"):
        # Skip RobotVision.__init__() to avoid creating old TRTEngine.
        # Manually initialize shared state.
        self.class_id = class_id
        self.device = device
        self.logger = logging.getLogger("{}.RobotVisionV2".format(__name__))
        self.last_boxes = {}
        self.alpha = 0.7
        self.imgsz = imgsz

        # Resolve engine path
        engine_path = model_path if model_path.endswith('.engine') else "{}/best.engine".format(model_path)
        if not Path(engine_path).exists():
            raise FileNotFoundError("TensorRT engine not found: {}".format(engine_path))

        # Create V2 engine and GPU preprocessor (share the same CUDA stream)
        self.model = TRTEngineV2(engine_path)
        self.preprocessor = GPUPreprocessor(imgsz, self.model.stream)

        # Initialize Kalman filter (inherited method)
        self._init_kalman_filter()

        # Async state
        self._async_meta = None
        self.logger.info("RobotVisionV2 ready: imgsz=%d, engine=%s", imgsz, engine_path)

    # ──── Synchronous GPU Pipeline ────────────────────────────────

    def predict_gpu(self, frame, conf_threshold=DEFAULT_CONF_THRESHOLD):
        """
        Full GPU pipeline: preprocess(GPU) -> infer(GPU) -> postprocess(CPU).

        Args:
            frame: BGR numpy array.
            conf_threshold: Detection confidence threshold.

        Returns:
            List of DetectedObject.
        """
        if frame is None:
            return []

        device_ptr, scale, pad_x, pad_y = self.preprocessor(frame)
        outputs = self.model.predict_from_device(device_ptr)
        return self._postprocess_raw(outputs, conf_threshold, scale, pad_x, pad_y)

    # ──── Async Double-Buffer API ─────────────────────────────────

    def launch_inference(self, frame):
        """
        Launch async GPU inference (preprocess + infer, no sync).

        Args:
            frame: BGR numpy array.

        Returns:
            Metadata tuple (scale, pad_x, pad_y) for later postprocessing,
            or None if frame is invalid.
        """
        if frame is None:
            self._async_meta = None
            return None

        device_ptr, scale, pad_x, pad_y = self.preprocessor(frame)
        self.model.infer_async(device_ptr)
        meta = (scale, pad_x, pad_y)
        self._async_meta = meta
        return meta

    def collect_raw_output(self):
        """
        Synchronize GPU and return raw TRT output tensors.

        Returns:
            List of numpy arrays (raw TRT outputs).
        """
        return self.model.sync_output()

    def postprocess_raw(self, raw_outputs, conf_threshold, scale, pad_x, pad_y):
        """
        Post-process raw TRT outputs into DetectedObject list.
        Public wrapper for use in double-buffer loops.
        """
        return self._postprocess_raw(raw_outputs, conf_threshold, scale, pad_x, pad_y)

    # ──── Internal ────────────────────────────────────────────────

    def _postprocess_raw(self, raw_outputs, conf_threshold, scale, pad_x, pad_y):
        """Detect output format and dispatch to appropriate postprocessor."""
        predictions = np.squeeze(raw_outputs[0])

        if predictions.ndim < 2:
            self.logger.warning("Unexpected TRT output shape: %s", predictions.shape)
            return []

        is_yolo26 = (
            predictions.ndim == 2
            and predictions.shape[1] == 6
            and predictions.shape[0] < 400
        )

        if is_yolo26:
            return self._postprocess_yolo26(predictions, conf_threshold, scale, pad_x, pad_y)
        else:
            return self._postprocess_yolov8(predictions, conf_threshold, scale, pad_x, pad_y)
