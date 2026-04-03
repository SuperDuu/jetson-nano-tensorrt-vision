import cv2
import logging

logger = logging.getLogger(__name__)

class UDPStreamer:
    """
    Hardware-accelerated H.264 UDP Video Streamer for Jetson Nano.
    Uses NVENC (nvv4l2h264enc) to offload video compression from CPU.
    """
    def __init__(self, host="127.0.0.1", port=5000, width=640, height=480, fps=30, bitrate=4000000):
        self.host = host
        self.port = port
        self.width = width
        self.height = height
        self.fps = fps
        
        # GStreamer Pipeline:
        # appsrc (BGR) -> videoconvert (NV12 CPU buffer) -> nvvidconv (NV12 NVMM buffer) -> 
        # nvv4l2h264enc (Hardware Encode) -> RTP Payload -> UDP Sink
        # Note: videoconvert is inevitable here because OpenCV appsrc strictly outputs BGR (User Space),
        # but the heavy MJPEG/H264 encoding is entirely shifted to the hardware NVENC block!
        self.pipeline = (
            f"appsrc ! video/x-raw, format=BGR ! "
            f"videoconvert ! video/x-raw, format=I420 ! "
            f"nvvidconv ! video/x-raw(memory:NVMM), format=NV12 ! "
            f"nvv4l2h264enc insert-sps-pps=true bitrate={bitrate} ! "
            f"h264parse ! rtph264pay config-interval=1 ! "
            f"udpsink host={host} port={port} qos=false max-lateness=-1"
        )
        
        self.writer = cv2.VideoWriter(
            self.pipeline,
            cv2.CAP_GSTREAMER,
            0,
            float(self.fps),
            (self.width, self.height)
        )
        
        if not self.writer.isOpened():
            logger.error("Failed to open UDP Streamer GStreamer Pipeline")
            raise RuntimeError("GStreamer hardware encoding pipeline failed to build.")
            
        logger.info(f"UDP Streamer initialized on {host}:{port} with H.264 Hardware Encoding")

    def send_frame(self, frame):
        """
        Sends an OpenCV BGR frame to the UDP stream.
        Must be the exact size specified at initialization.
        """
        if frame is None:
            return
            
        # Ensure dimensions match
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
            
        self.writer.write(frame)

    def close(self):
        if self.writer.isOpened():
            self.writer.release()
            logger.info("UDP Streamer shutdown gracefully.")
