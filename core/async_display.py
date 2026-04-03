"""
Async display process for offloading cv2.imshow to a separate OS process.
Frees the main inference loop from display blocking.

Compatible with Python 3.6+ / Jetson Nano.
"""

import threading
import queue
import cv2
import numpy as np
import logging
import time
from core.utils import letterbox

try:
    from core.udp_streamer import UDPStreamer
except ImportError:
    UDPStreamer = None

logger = logging.getLogger(__name__)


class DisplayThread(object):
    """
    Runs cv2.imshow + overlay drawing in a separate thread.
    Zero-IPC copy overhead. Safe for OpenCV on Python 3.6+ using waitKey releasing GIL.
    """

    def __init__(self, window_name="RBC2026 V2", max_queue_size=2, headless=False, udp_stream=False):
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._key_queue = queue.Queue(maxsize=10)
        self._stop_event = threading.Event()
        self.window_name = window_name
        self.headless = headless
        self.udp_stream = udp_stream
        
        self.streamer = None
        if self.udp_stream and UDPStreamer is not None:
             # Fixed Laptop IP for Wifi/Ethernet Modem Setup
            self.streamer = UDPStreamer(host="192.168.2.1")
            
        self._thread = threading.Thread(
            target=self._display_loop,
            daemon=True,
        )

    def start(self):
        """Start the display thread."""
        self._thread.start()
        return self

    def send_frame(self, frame, target_point=None, status="", label="",
                   error_x=0, fps=0.0, extra_dets=None, state=0, force_square=False):
        """
        Send a frame + overlay data to the display thread (non-blocking).
        Drops old frames if queue is full.

        Args:
            frame: BGR numpy array to display.
            target_point: (x, y) tuple or None.
            status: Status string ("LOCKED", "SEARCHING", "LOST").
            label: Current label string.
            error_x: Error X value for HUD.
            fps: Current FPS for HUD.
            extra_dets: List of ([x1,y1,x2,y2], label, score) for debug boxes.
            state: Current algorithm state (1 or 2).
            force_square: Whether to letterbox to 512x512.
        """
        payload = {
            "frame": frame.copy() if frame is not None else None, # Thread-safety copy 1 pass
            "target": target_point,
            "status": status,
            "label": label,
            "err_x": error_x,
            "fps": fps,
            "dets": extra_dets,
            "state": state,
            "force_square": force_square,
        }

        # Non-blocking: drop oldest if full
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except Exception:
                pass
        try:
            self._queue.put_nowait(payload)
        except Exception:
            pass

    def get_key(self):
        """Get the latest key typed in the display window."""
        try:
            return self._key_queue.get_nowait()
        except Exception:
            return None

    def _display_loop(self):
        """Method running in child thread."""
        while not self._stop_event.is_set():
            try:
                payload = self._queue.get(timeout=0.5)
            except Exception:
                continue

            frame = payload.get("frame")
            if frame is None:
                break
                
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
            force_square = payload.get("force_square", False)
            scale_d = 1.0
            pw, ph = 0, 0
            if force_square:
                frame, scale_d, (pw, ph) = letterbox(frame, (512, 512))

            h, w = frame.shape[:2]
            target = payload.get("target")
            status = payload.get("status", "")
            label = payload.get("label", "")
            err_x = payload.get("err_x", 0)
            fps = payload.get("fps", 0.0)
            dets = payload.get("dets")
            state = payload.get("state", 0)

            color = (0, 255, 0) if status in ("LOCKED", "SEARCHING") else (0, 0, 255)
            sc_x = w // 2

            # Draw debug detection boxes
            if dets:
                for box, b_label, b_score in dets:
                    bx1, by1, bx2, by2 = box
                    dbx1, dby1 = int(bx1 * scale_d + pw), int(by1 * scale_d + ph)
                    dbx2, dby2 = int(bx2 * scale_d + pw), int(by2 * scale_d + ph)
                    cv2.rectangle(frame, (dbx1, dby1), (dbx2, dby2), (0, 255, 0), 1)
                    cv2.putText(frame, "{} {:.2f}".format(b_label, b_score),
                                (dbx1, dby1 - 5), 0, 0.4, (0, 255, 0), 1)

            # Draw target tracker
            if target is not None:
                tx, ty = target
                dtx, dty = int(tx * scale_d + pw), int(ty * scale_d + ph)
                
                cv2.line(frame, (sc_x, h), (dtx, dty), color, 2)
                cv2.circle(frame, (dtx, dty), 10, color, -1)

            # HUD header
            cv2.rectangle(frame, (0, 0), (w, 35), (40, 40, 40), -1)
            hud = "MODE:{} | {} | {} | EX:{}".format(state, status, label, err_x)
            cv2.putText(frame, hud, (10, 25), 0, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, "FPS:{:.1f}".format(fps), (10, 55), 0, 0.6, (200, 200, 200), 2)

            if self.streamer:
                self.streamer.send_frame(frame)

            if not self.headless:
                cv2.imshow(self.window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self._stop_event.set()
                    break
                elif key in (ord('0'), ord('1'), ord('2')):
                    try:
                        self._key_queue.put_nowait(chr(key))
                    except Exception:
                        pass

        if not self.headless:
            cv2.destroyAllWindows()
        if self.streamer:
            self.streamer.close()

    def is_running(self):
        """Check if display thread is alive."""
        return self._thread.is_alive() and not self._stop_event.is_set()

    def stop(self):
        """Stop the display thread."""
        logger.info("Stopping DisplayThread...")
        self._stop_event.set()
        # Send None to unblock get()
        try:
            self._queue.put_nowait({"frame": None})
        except Exception:
            pass
        self._thread.join(timeout=2.0)
        logger.info("DisplayThread stopped.")
