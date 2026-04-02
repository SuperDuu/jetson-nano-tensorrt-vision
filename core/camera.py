"""
Camera streaming module for RBC2026 Robocon Vision System.

This module provides threaded camera capture for low-latency video streaming.
"""

import cv2
import threading
import logging
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class CameraStream:
    """
    Threaded camera stream for real-time video capture.
    
    Uses separate thread to continuously read frames, reducing latency.
    """
    
    def __init__(self, src: int = 0, buffer_size: int = 1, width: Optional[int] = None, height: Optional[int] = None):
        """
        Initialize camera stream.
        
        Args:
            src: Camera device ID (default: 0)
            buffer_size: Buffer size for VideoCapture (default: 1 for low latency)
            width: Desired camera capture width
            height: Desired camera capture height
        """
        self.src = src
        self.buffer_size = buffer_size
        self.width = width
        self.height = height
        self.logger = logging.getLogger(f"{__name__}.CameraStream")
        
        self.cap: Optional[cv2.VideoCapture] = None
        self._frame_buf = deque(maxlen=1)  # Only keep the latest frame
        self.stopped = False
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        
        self._init_camera()
    
    def _init_camera(self) -> None:
        """Initialize camera capture."""
        try:
            self.cap = cv2.VideoCapture(self.src)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.src}")
            
            # Set buffer size for low latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            
            # Set resolution if provided
            if self.width:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Read initial frame
            success, frame = self.cap.read()
            if not success:
                raise RuntimeError(f"Failed to read initial frame from camera {self.src}")
            self._frame_buf.append(frame)
            
            self.logger.info(f"Camera {self.src} initialized successfully")
        
        except Exception as e:
            self.logger.error(f"Error initializing camera: {e}")
            raise
    
    def start(self) -> 'CameraStream':
        """
        Start background thread for frame capture.
        
        Returns:
            Self for method chaining
        """
        if self.stopped:
            raise RuntimeError("Cannot start stopped camera stream")
        
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self
    
    def update(self) -> None:
        """Background thread function to continuously read frames."""
        while not self.stopped:
            try:
                if self.cap is None or not self.cap.isOpened():
                    self.stopped = True
                    break
                
                success, frame = self.cap.read()
                if success:
                    with self.lock:
                        self._frame_buf.append(frame)
                    self.new_frame_event.set()
                else:
                    self.logger.warning("Failed to read frame from camera")
                    self.stopped = True
                    break
            
            except Exception as e:
                self.logger.error(f"Error reading frame: {e}")
                self.stopped = True
                break
    
    def read(self) -> Optional[cv2.VideoCapture]:
        """
        Get latest frame from camera.
        
        Returns:
            Latest frame or None if not available
        """
        with self.lock:
            return self._frame_buf[-1] if self._frame_buf else None
    
    def read_latest(self, wait: bool = False, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Get the latest frame, optionally waiting for a truly new frame.
        
        Args:
            wait: If True, blocks until a new frame is captured.
            timeout: Maximum time to wait for a frame (seconds).
        Returns:
            Latest frame, or None if timed out/not available.
        """
        if wait:
            if not self.new_frame_event.wait(timeout):
                return None
                
        self.new_frame_event.clear()
        with self.lock:
            return self._frame_buf[-1] if self._frame_buf else None
    
    def stop(self) -> None:
        """Stop camera stream and release resources."""
        self.logger.info("Stopping camera stream...")
        self.stopped = True
        
        if self.cap is not None:
            try:
                self.cap.release()
                self.logger.info("Camera released")
            except Exception as e:
                self.logger.error(f"Error releasing camera: {e}")
