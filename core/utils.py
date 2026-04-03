"""
Utility functions for RBC2026 Robocon Vision System.

This module contains shared utility functions used across the project.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

# Constants
BACKGROUND_VALUE = 128
CNN_INPUT_SIZE = 64


def preprocess_roi_for_cnn(roi: np.ndarray, input_size: int = CNN_INPUT_SIZE) -> Optional[np.ndarray]:
    """
    Preprocess ROI for CNN classification.
    Optimized for ARM processors: uses squashing resize to match training pipeline.
    """
    if roi is None or roi.size == 0:
        return None
    
    try:
        # 1. Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGRA2GRAY) if roi.shape[2] == 4 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        # 2. Resize directly to (input_size, input_size) to MATCH SQUASHED training images
        # tf.image_dataset_from_directory and tf.image.resize squash images by default.
        h, w = gray.shape[:2]
        scale = input_size / max(h, w)
        
        # Use INTER_CUBIC for upscaling, INTER_AREA for downscaling
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(gray, (input_size, input_size), interpolation=interp)
        
        # 3. Convert to float32 and Standardize
        canvas_float = resized.astype(np.float32)
        
        # Standardize (Zero-Mean, Unit-Variance) - Optimized with cv2.meanStdDev
        mean_val, std_val = cv2.meanStdDev(canvas_float)
        mean = mean_val[0][0]
        stddev = std_val[0][0]
        
        num_pixels = input_size * input_size
        adjusted_stddev = max(stddev, 1.0 / np.sqrt(num_pixels))
        
        standardized = (canvas_float - mean) / adjusted_stddev
        
        # Reshape to (1, input_size, input_size, 1)
        return standardized.reshape(1, input_size, input_size, 1)
    
    except Exception as e:
        logger.error(f"Error preprocessing ROI: {e}")
        return None


def validate_and_clamp_bbox(x1: int, y1: int, x2: int, y2: int, frame_shape: Tuple[int, int, int]) -> Optional[Tuple[int, int, int, int]]:
    """
    Validate and Clamp Bounding Box before cropping ROI.
    Prevents memory errors or crashes from invalid YOLO indices.
    """
    h_frame, w_frame = frame_shape[:2]
    
    x1_c = max(0, min(x1, w_frame - 1))
    y1_c = max(0, min(y1, h_frame - 1))
    x2_c = max(0, min(x2, w_frame))
    y2_c = max(0, min(y2, h_frame))
    
    if x2_c <= x1_c or y2_c <= y1_c:
        return None
        
    return (x1_c, y1_c, x2_c, y2_c)


def validate_roi_bounds(roi: np.ndarray, frame_shape: Tuple[int, int, int]) -> bool:
    """
    Validate ROI bounds are within frame dimensions.
    
    Args:
        roi: ROI array to validate
        frame_shape: Frame shape tuple (height, width, channels)
    
    Returns:
        True if ROI is valid, False otherwise
    """
    if roi is None or roi.size == 0:
        return False
    
    h_frame, w_frame = frame_shape[:2]
    h_roi, w_roi = roi.shape[:2]
    
    return 0 < h_roi <= h_frame and 0 < w_roi <= w_frame


def calculate_fps(current_time: float, previous_time: float) -> float:
    """
    Calculate FPS from time difference.
    
    Args:
        current_time: Current timestamp
        previous_time: Previous timestamp
    
    Returns:
        FPS value (0 if time difference is invalid)
    """
    time_diff = current_time - previous_time
    if time_diff > 0:
        return 1.0 / time_diff
    return 0.0


def clamp_value(value: int, min_val: int, max_val: int) -> int:
    """
    Clamp value between min and max.
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
    
    Returns:
        Clamped value
    """
    return max(min_val, min(value, max_val))


def letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (512, 512), color: Tuple[int, int, int] = (BACKGROUND_VALUE, BACKGROUND_VALUE, BACKGROUND_VALUE)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image to a 32-pixel-multiple rectangle while preserving aspect ratio using padding.
    
    Args:
        img: Input image
        new_shape: Target shape (height, width)
        color: Padding color (BGR)
        
    Returns:
        - Resized and padded image
        - Resize ratio
        - (padding_w, padding_h)
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, r, (left, top)
