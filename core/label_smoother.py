"""
Label smoothing module for RBC2026 Robocon Vision System.

This module provides temporal smoothing for labels to reduce noise.
"""

import numpy as np
import logging
from collections import Counter
from typing import Tuple, Dict, List, Optional

logger = logging.getLogger(__name__)


class LabelSmoother:
    """
    Smooths labels over time using sliding window.
    
    Reduces flickering and noise by maintaining history
    and returning most common label with averaged confidence.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize label smoother.
        
        Args:
            window_size: Size of sliding window for smoothing (default: 5)
        """
        self.window_size = window_size
        self.history: Dict[str, List[Tuple[str, float]]] = {}
        self.logger = logging.getLogger(f"{__name__}.LabelSmoother")
    
    def smooth(self, box_id: str, label: str, conf: float) -> Tuple[str, float]:
        """
        Smooth label prediction using sliding window.
        
        Args:
            box_id: Unique identifier for the bounding box
            label: Raw predicted label
            conf: Raw confidence score
        
        Returns:
            Tuple of (smoothed_label, averaged_confidence)
        """
        try:
            # Initialize history for new box_id
            if box_id not in self.history:
                self.history[box_id] = []
            
            # Add new prediction to history
            self.history[box_id].append((label, conf))
            
            # Maintain window size
            if len(self.history[box_id]) > self.window_size:
                self.history[box_id].pop(0)
            
            # Get most common label in window
            labels = [x[0] for x in self.history[box_id]]
            if not labels:
                return label, conf
            
            most_common_label, count = Counter(labels).most_common(1)[0]
            
            # Calculate average confidence for most common label
            confidences = [x[1] for x in self.history[box_id] if x[0] == most_common_label]
            avg_conf = np.mean(confidences) if confidences else conf
            
            return most_common_label, float(avg_conf)
        
        except Exception as e:
            self.logger.error(f"Error smoothing label: {e}")
            return label, conf
    
    def clear(self, box_id: Optional[str] = None) -> None:
        """
        Clear history for specific box_id or all boxes.
        
        Args:
            box_id: Box ID to clear, or None to clear all
        """
        if box_id is None:
            self.history.clear()
        elif box_id in self.history:
            del self.history[box_id]
    
    def generate_box_id(self, x1: int, y1: int, grid_size: int = 40) -> str:
        """
        Generate box ID from coordinates using grid-based hashing.
        
        Args:
            x1: Top-left x coordinate
            y1: Top-left y coordinate
            grid_size: Grid size for hashing (default: 40)
        
        Returns:
            Box ID string
        """
        return f"{x1//grid_size}_{y1//grid_size}"
