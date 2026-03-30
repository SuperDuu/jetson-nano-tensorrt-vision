import cv2
import time
import json
import logging
import numpy as np
import ctypes
import threading
from pathlib import Path
from typing import Optional, Tuple, List

from .vision import RobotVision
from .connection import UARTManager
from .camera import CameraStream
from .label_smoother import LabelSmoother
from .utils import preprocess_roi_for_cnn, letterbox
from .config_manager import ConfigManager
from .trt_engine import TRTEngine

class RoboconSystem:
    def __init__(self, config_path: str = "config.yaml"):
        try:
            self.winmm = ctypes.WinDLL('winmm')
            self.winmm.timeBeginPeriod(1)
        except: pass

        self._setup_logging()
        self.logger = logging.getLogger("RBC2026")
        
        # Core's ConfigManager now handles project root and relative paths
        self.config = ConfigManager(config_path)
        
        self.target_mode = self.config.get("classification.target_types", ["REAL"])[0].upper()
        self.frame_idx = 0
        self.loss_counter = 0
        self.max_loss_frames = self.config.get("detection.max_loss_frames", 7)
        
        self.latest_target_point = None
        self.latest_label = "NONE"
        self.latest_error_x = 0
        self.status_text = "SEARCHING"
        self.last_target_x = None 
        self.target_switch_threshold = 80
        
        self.is_headless = self.config.get("display.headless", False)
        self.target_fps = self.config.get("display.fps", 30)
        self.frame_duration = 1.0 / self.target_fps
        
        self.display_fps = 0.0
        self.last_fps_update_time = time.time()
        self.frame_count_since_update = 0

        # ASYNC INFERENCE STATE
        self.inference_lock = threading.Lock()
        self.latest_inference_data = (None, "NONE")
        self.inference_running = True
        self.is_async = self.config.get("system.async", True)

        try:
            self._init_models()
            self._init_hardware()
            self._init_tracking()

            if self.is_async:
                self.inf_thread = threading.Thread(target=self._inference_loop, daemon=True)
                self.inf_thread.start()

            self.logger.info(f"--- SYSTEM READY | ASYNC: {self.is_async} | MODE: {self.target_mode} ---")
        except Exception as e:
            self.logger.error(f"Init Failed: {e}")
            raise

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    def _init_models(self): 
        # YOLO - RobotVision now handles its own TRTEngine
        yolo_path = self.config.get_path("paths.models.yolo_engine")
        if not yolo_path:
             yolo_path = self.config.get_path("paths.models.yolo_xml") # Fallback to check path
             yolo_path = str(Path(yolo_path).with_suffix('.engine'))
             
        self.vision = RobotVision(yolo_path, device="GPU")
        
        # CNN (Optional based on config)
        self.use_cnn = "cnn_engine" in self.config.get("paths.models", {})
        if self.use_cnn:
            cnn_path = self.config.get_path("paths.models.cnn_engine")
            self.compiled_cnn = TRTEngine(cnn_path)
            
            labels_path = self.config.get_path("paths.models.labels_json")
            with open(labels_path, 'r') as f:
                self.labels_cnn = {int(v): k for k, v in json.load(f).items()}

    def _init_hardware(self):
        cam_id = self.config.get("hardware.camera.device_id", 0)
        self.camera = CameraStream(src=cam_id, buffer_size=1).start()
        self.uart = UARTManager(port=self.config.get("hardware.uart.port", "COM10"), 
                                baudrate=self.config.get("hardware.uart.baudrate", 115200))

    def _init_tracking(self):
        self.smoother = LabelSmoother(window_size=self.config.get("detection.label_smoothing.window_size", 7))
        self.conf_yolo = self.config.get("models.yolo.conf_threshold", 0.6)
        self.conf_cnn = self.config.get("models.cnn.conf_threshold", 0.5)

    def _inference_loop(self):
        imgsz = self.config.get("models.yolo.input_size", 512)
        while self.inference_running and not self.camera.stopped:
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.01); continue
            
            res = self._process_frame(frame, imgsz)
            with self.inference_lock:
                self.latest_inference_data = res
            time.sleep(0.001)

    def _process_frame(self, frame, imgsz):
        detections = self.vision.predict(frame, conf_threshold=self.conf_yolo, imgsz=imgsz)
        if not detections: return None, "NONE"
        
        if not self.use_cnn:
            # Pick leftmost, if tied prefer uppermost (lower Y)
            det = min(detections, key=lambda d: ((d.xyxy[0][0] + d.xyxy[0][2]) / 2 // 60, (d.xyxy[0][1] + d.xyxy[0][3]) / 2))
            cx, cy = int((det.xyxy[0][0] + det.xyxy[0][2]) // 2), int((det.xyxy[0][1] + det.xyxy[0][3]) // 2)
            return (cx, cy), "OBJECT"

        # Selective Leftmost with CNN
        h_f, w_f = frame.shape[:2]
        sorted_dets = sorted(detections, key=lambda d: ((d.xyxy[0][0] + d.xyxy[0][2]) / 2 // 60, (d.xyxy[0][1] + d.xyxy[0][3]) / 2))
        for det in sorted_dets:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            roi = frame[max(0, y1):min(h_f, y2), max(0, x1):min(w_f, x2)]
            input_data = preprocess_roi_for_cnn(roi)
            if input_data is None: continue
            
            cnn_res = self.compiled_cnn.predict(input_data)[0]
            idx = np.argmax(cnn_res[0])
            label_raw = self.labels_cnn.get(idx, "UNK").upper()
            
            if label_raw.startswith(self.target_mode) and cnn_res[0][idx] >= self.conf_cnn:
                if self.last_target_x is None or abs(cx - self.last_target_x) > self.target_switch_threshold:
                    self._reset_kalman_at_pos(cx, cy)
                self.last_target_x = cx
                smoothed_label, _ = self.smoother.smooth("target", label_raw, cnn_res[0][idx])
                return (cx, cy), smoothed_label
        return None, "NONE"

    def _reset_kalman_at_pos(self, x, y):
        self.vision._init_kalman_filter()
        self.vision.kalman.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.vision.kalman.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.vision.kalman_initialized = True
        self.smoother.history.clear()

    def run(self):
        try:
            while not self.camera.stopped:
                loop_start = time.time()
                frame = self.camera.read()
                if frame is None: continue
                
                h_f, w_f = frame.shape[:2]
                imgsz = self.config.get("models.yolo.input_size", 512)
                force_square = self.config.get("display.force_square", True)

                # 1. AI STEP
                if self.is_async:
                    new_ai_pt, label = (None, "NONE")
                    with self.inference_lock:
                        if self.latest_inference_data:
                            new_ai_pt, label = self.latest_inference_data
                            self.latest_inference_data = None
                else:
                    new_ai_pt, label = self._process_frame(frame, imgsz)

                # 2. TRACKING STEP
                if new_ai_pt:
                    tx, ty = self.vision.update_kalman(new_ai_pt[0], new_ai_pt[1])
                    self.latest_label = label
                    self.loss_counter, self.status_text = 0, "LOCKED"
                else:
                    self.loss_counter += 1
                    tx, ty = self.vision.update_kalman()
                    
                    status_is_searching = (self.status_text == "SEARCHING")
                    if status_is_searching:
                        # If we never found anything, don't let Kalman drift from center
                        tx, ty = (w_f // 2, h_f // 2)
                    
                    if self.loss_counter >= self.max_loss_frames:
                        self.status_text, self.latest_label, self.last_target_x = "LOST", "NONE", None
                        tx, ty = (w_f // 2, h_f // 2)

                # 3. DISPLAY & UART
                if force_square:
                    _, display_scale, (pad_w, pad_h) = letterbox(frame, (imgsz, imgsz))
                    screen_center_x = imgsz // 2
                    dtx, dty = int(tx * display_scale + pad_w), int(ty * display_scale + pad_h)
                    target_point = (dtx, dty)
                    curr_wh = imgsz
                else:
                    screen_center_x = w_f // 2
                    target_point = (tx, ty)
                    curr_wh = (w_f, h_f)

                self.latest_error_x = int(target_point[0] - screen_center_x) if self.status_text == "LOCKED" else 999
                self.uart.send_error(self.latest_error_x)

                # 4. UI Rendering (Omitted for brevity in shared core if headless, but kept for local running)
                if not self.is_headless:
                    df = frame if not force_square else letterbox(frame, (imgsz, imgsz))[0]
                    color = (0, 255, 0) if self.status_text == "LOCKED" else (0, 0, 255)
                    
                    # Target Point and Line
                    curr_h = imgsz if force_square else h_f
                    curr_w = imgsz if force_square else w_f
                    
                    cv2.line(df, (screen_center_x, curr_h), target_point, color, 2)
                    cv2.circle(df, target_point, 10, color, -1)
                    
                    # Status Header
                    cv2.rectangle(df, (0, 0), (curr_w, 35), (40, 40, 40), -1)
                    cv2.putText(df, f"{self.status_text} | {self.latest_label} | EX:{self.latest_error_x}", (10, 25), 0, 0.6, (255, 255, 255), 1)
                    cv2.putText(df, f"FPS: {self.display_fps:.1f}", (curr_w - 100, 25), 0, 0.6, (0, 255, 0), 2)
                    
                    cv2.imshow("RBC2026 Core", df)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break

                self.frame_count_since_update += 1
                if time.time() - self.last_fps_update_time >= 0.5:
                    self.display_fps = self.frame_count_since_update / (time.time() - self.last_fps_update_time)
                    print(f"FPS: {self.display_fps:.1f} | {self.status_text} | ErrX: {self.latest_error_x}", end='\r')
                    self.frame_count_since_update, self.last_fps_update_time = 0, time.time()

                wait = self.frame_duration - (time.time() - loop_start)
                if wait > 0: time.sleep(wait)
        finally:
            self.cleanup()

    def cleanup(self):
        self.inference_running = False
        if hasattr(self, 'inf_thread'): self.inf_thread.join(timeout=1.0)
        self.camera.stop(); self.uart.stop(); cv2.destroyAllWindows()
        try: self.winmm.timeEndPeriod(1)
        except: pass
