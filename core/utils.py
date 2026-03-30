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
    Preprocess ROI (Region of Interest) for CNN classification.
    
    Converts ROI to grayscale, resizes with aspect ratio preservation,
    and pads to fixed size with gray background.
    
    Args:
        roi: Input ROI image (BGR format)
        input_size: Target input size (default: 64)
    
    Returns:
        Preprocessed array of shape (1, input_size, input_size, 1) normalized to [0, 1],
        or None if ROI is invalid
    """
    if roi is None or roi.size == 0:
        return None
    
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate aspect ratio preserving resize
        h, w = gray.shape[:2]
        scale = input_size / max(h, w)
        nw, nh = int(w * scale), int(h * scale)
        
        # Resize with high-quality interpolation
        resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
        
        # Create canvas with gray background
        canvas = np.full((input_size, input_size), BACKGROUND_VALUE, dtype=np.uint8)
        
        # Center the resized image on canvas
        y_offset = (input_size - nh) // 2
        x_offset = (input_size - nw) // 2
        canvas[y_offset:y_offset+nh, x_offset:x_offset+nw] = resized
        
        # Reshape and normalize to [0, 1]
        return canvas.reshape(1, input_size, input_size, 1).astype(np.float32) / 255.0
    
    except Exception as e:
        logger.error(f"Error preprocessing ROI: {e}")
        return None


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
