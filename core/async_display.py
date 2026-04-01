"""
Async display process for offloading cv2.imshow to a separate OS process.
Frees the main inference loop from display blocking.

Compatible with Python 3.6+ / Jetson Nano.
"""

import multiprocessing as mp
import cv2
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


class DisplayProcess(object):
    """
    Runs cv2.imshow + overlay drawing in a separate process.

    Usage:
        display = DisplayProcess("RBC2026")
        display.start()
        ...
        display.send_frame(frame, overlays)
        ...
        display.stop()
    """

    def __init__(self, window_name="RBC2026 V2", max_queue_size=2):
        self._queue = mp.Queue(maxsize=max_queue_size)
        self._stop_event = mp.Event()
        self._process = mp.Process(
            target=DisplayProcess._display_loop,
            args=(self._queue, self._stop_event, window_name),
            daemon=True,
        )

    def start(self):
        """Start the display process."""
        self._process.start()
        return self

    def send_frame(self, frame, target_point=None, status="", label="",
                   error_x=0, fps=0.0, extra_dets=None):
        """
        Send a frame + overlay data to the display process (non-blocking).
        Drops old frames if queue is full.

        Args:
            frame: BGR numpy array to display.
            target_point: (x, y) tuple or None.
            status: Status string ("LOCKED", "SEARCHING", "LOST").
            label: Current label string.
            error_x: Error X value for HUD.
            fps: Current FPS for HUD.
            extra_dets: List of ([x1,y1,x2,y2], label, score) for debug boxes.
        """
        payload = {
            "frame": frame,
            "target": target_point,
            "status": status,
            "label": label,
            "err_x": error_x,
            "fps": fps,
            "dets": extra_dets,
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

    @staticmethod
    def _display_loop(queue, stop_event, window_name):
        """Static method running in child process."""
        while not stop_event.is_set():
            try:
                payload = queue.get(timeout=0.5)
            except Exception:
                continue

            frame = payload["frame"]
            if frame is None:
                break

            h, w = frame.shape[:2]
            target = payload["target"]
            status = payload["status"]
            label = payload["label"]
            err_x = payload["err_x"]
            fps = payload["fps"]
            dets = payload["dets"]

            color = (0, 255, 0) if status in ("LOCKED", "SEARCHING") else (0, 0, 255)
            sc_x = w // 2

            # Draw debug detection boxes
            if dets:
                for box, b_label, b_score in dets:
                    bx1, by1, bx2, by2 = box
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 1)
                    cv2.putText(frame, "{} {:.2f}".format(b_label, b_score),
                                (bx1, by1 - 5), 0, 0.4, (0, 255, 0), 1)

            # Draw target tracker
            if target is not None:
                cv2.line(frame, (sc_x, h), target, color, 2)
                cv2.circle(frame, target, 10, color, -1)

            # HUD header
            cv2.rectangle(frame, (0, 0), (w, 35), (40, 40, 40), -1)
            hud = "{} | {} | EX:{}".format(status, label, err_x)
            cv2.putText(frame, hud, (10, 25), 0, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, "FPS:{:.1f}".format(fps),
                        (w - 120, 25), 0, 0.6, (0, 255, 0), 2)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stop_event.set()
                break

        cv2.destroyAllWindows()

    def is_running(self):
        """Check if display process is alive."""
        return self._process.is_alive() and not self._stop_event.is_set()

    def stop(self):
        """Stop the display process."""
        logger.info("Stopping DisplayProcess...")
        self._stop_event.set()
        # Send None to unblock get()
        try:
            self._queue.put_nowait({"frame": None})
        except Exception:
            pass
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
        logger.info("DisplayProcess stopped.")
