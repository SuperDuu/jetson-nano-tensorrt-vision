"""
GStreamer-based camera stream for Jetson Nano.
Drop-in replacement for CameraStream with same API.

Uses nvarguscamerasrc (CSI) or v4l2src (USB) + hardware ISP.
Compatible with Python 3.6+ / Jetson Nano L4T r32.7.1.

Requires: python3-gi, gstreamer1.0-plugins-good/bad
  apt-get install -y python3-gi gstreamer1.0-tools \\
      gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
"""

import cv2
import threading
import logging
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


def _build_gst_pipeline(src, width, height, framerate=30, camera_type="auto"):
    """
    Build GStreamer pipeline string for OpenCV VideoCapture.

    Args:
        src: Device ID (int) or device path (str like '/dev/video0').
        width: Capture width.
        height: Capture height.
        framerate: Target framerate.
        camera_type: 'csi', 'usb', or 'auto'.

    Returns:
        GStreamer pipeline string.
    """
    if camera_type == "auto":
        # CSI cameras on Jetson use nvarguscamerasrc (no /dev/videoN path)
        if isinstance(src, int) and src == 0:
            camera_type = "csi"
        else:
            camera_type = "usb"

    if camera_type == "csi":
        # CSI camera via Jetson ISP
        pipeline = (
            "nvarguscamerasrc sensor-id={src} ! "
            "video/x-raw(memory:NVMM), width=(int){w}, height=(int){h}, "
            "format=(string)NV12, framerate=(fraction){fps}/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=(int){w}, height=(int){h}, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! "
            "appsink drop=true max-buffers=1"
        ).format(src=src, w=width, h=height, fps=framerate)
    else:
        # USB camera via v4l2
        dev = src if isinstance(src, str) else "/dev/video{}".format(src)
        pipeline = (
            "v4l2src device={dev} ! "
            "video/x-raw, width=(int){w}, height=(int){h} ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! "
            "appsink drop=true max-buffers=1"
        ).format(dev=dev, w=width, h=height)

    return pipeline


class GstCameraStream(object):
    """
    GStreamer camera stream for Jetson Nano.
    Same API as core.camera.CameraStream: start(), read(), read_latest(), stop().
    """

    def __init__(self, src=0, width=640, height=480, framerate=30,
                 camera_type="auto", fallback_opencv=True):
        """
        Args:
            src: Camera device ID or path.
            width: Capture width.
            height: Capture height.
            framerate: Target framerate.
            camera_type: 'csi', 'usb', or 'auto'.
            fallback_opencv: If GStreamer fails, fall back to cv2.VideoCapture.
        """
        self.src = src
        self.width = width
        self.height = height
        self.framerate = framerate
        self.camera_type = camera_type
        self.fallback_opencv = fallback_opencv
        self.logger = logging.getLogger("{}.GstCameraStream".format(__name__))

        self.cap = None  # type: Optional[cv2.VideoCapture]
        self._frame_buf = deque(maxlen=1)
        self.stopped = False
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self._using_gst = False

        self._init_camera()

    def _init_camera(self):
        """Initialize camera with GStreamer pipeline, fallback to raw OpenCV."""
        # Try GStreamer pipeline first
        pipeline = _build_gst_pipeline(
            self.src, self.width, self.height,
            self.framerate, self.camera_type
        )
        self.logger.info("Trying GStreamer pipeline: %s", pipeline)

        try:
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if self.cap.isOpened():
                success, frame = self.cap.read()
                if success:
                    self._frame_buf.append(frame)
                    self._using_gst = True
                    self.logger.info(
                        "GStreamer camera initialized: %dx%d",
                        frame.shape[1], frame.shape[0]
                    )
                    return
                else:
                    self.cap.release()
        except Exception as e:
            self.logger.warning("GStreamer init failed: %s", e)

        # Fallback to raw OpenCV
        if self.fallback_opencv:
            self.logger.info("Falling back to OpenCV VideoCapture(%s)", self.src)
            self.cap = cv2.VideoCapture(self.src)
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open camera {}".format(self.src))
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self.width:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            success, frame = self.cap.read()
            if not success:
                raise RuntimeError("Failed to read initial frame")
            self._frame_buf.append(frame)
            self._using_gst = False
            self.logger.info("OpenCV fallback camera initialized: %s", frame.shape)
        else:
            raise RuntimeError("GStreamer camera failed and fallback disabled")

    def start(self):
        """Start background frame capture thread. Returns self."""
        if self.stopped:
            raise RuntimeError("Cannot start stopped camera stream")
        threading.Thread(target=self._update_loop, daemon=True).start()
        return self

    def _update_loop(self):
        """Background thread: continuously read frames."""
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
                    self.logger.warning("Failed to read frame")
                    self.stopped = True
                    break
            except Exception as e:
                self.logger.error("Error reading frame: %s", e)
                self.stopped = True
                break

    def read(self):
        """Get latest frame."""
        with self.lock:
            return self._frame_buf[-1] if self._frame_buf else None

    def read_latest(self):
        """Get latest frame and clear new_frame_event."""
        self.new_frame_event.clear()
        with self.lock:
            return self._frame_buf[-1] if self._frame_buf else None

    def stop(self):
        """Stop camera and release resources."""
        self.logger.info("Stopping GstCameraStream...")
        self.stopped = True
        if self.cap is not None:
            try:
                self.cap.release()
                self.logger.info("Camera released (gst=%s)", self._using_gst)
            except Exception as e:
                self.logger.error("Error releasing camera: %s", e)
