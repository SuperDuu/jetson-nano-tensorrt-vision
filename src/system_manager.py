import cv2
import numpy as np
import threading
import time
import logging
import json
import serial
import sys
import os
from pathlib import Path
import yaml

# Add project root to sys.path for core imports
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # project root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Import from core
from core.config_manager import ConfigManager
from core.vision import RobotVision
from core.camera import CameraStream
from core.label_smoother import LabelSmoother
from core.utils import letterbox, preprocess_roi_for_cnn
from core.trt_engine import TRTEngine

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("SystemManager")

class SystemManager:
    def __init__(self, config_path="global_config.yaml"):
        print(f"--- INIT START: {config_path} ---", flush=True)
        # The core ConfigManager handles relative paths automatically
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        self.load_mode = self.config['system'].get('load_mode', 3)
        self.state = self.config['system']['initial_state']
        self.force_square = self.config['system']['force_square']
        self.headless = self.config['system']['headless']
        # FPS capping removed — run at max throughput
        
        # Model references (None until loaded)
        self.v1_vision = None
        self.v2_vision = None
        self.v2_cnn = None
        self.v2_labels = {}
        self.v2_smoother = None
        
        # Async Inference State
        self.inference_lock = threading.Lock()
        self.latest_inference_data = ((None, "NONE"), [])
        self.inference_running = True
        
        # Filtering & Locking State
        self.locked_target_id = None
        self.label_history = []  # Buffer for classification stability
        self.history_len = 5     # Shorter buffer for faster response
        self.min_majority = 2    # Require at least 2 identical labels
        
        # UART & Control Smoothing
        self.last_uart_time = 0
        self.uart_interval = 1.0 / 30.0 # 30Hz limit
        self.last_filtered_pt = None    # Last FILTERED point for stability
        
        # Target selection: bin X then prefer top (lower Y)
        self.x_bin_size = 60            # Objects within 60px X are considered "same column"
        
        try:
            print("  Initializing hardware...", flush=True)
            self._init_hardware()
            print("  Initializing models...", flush=True)
            self._init_models()
            
            # Start Background Inference Thread
            print("  Starting inference thread...", flush=True)
            self.inf_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self.inf_thread.start()
            
            logger.info(f"SystemManager initialized. Initial State: {self.state}")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def _init_hardware(self):
        self.use_test_image = self.config['system'].get('test_image', False)
        if self.use_test_image:
            self.test_img_path = self.config['system']['test_image_path']
            abs_img_path = self.config_manager.resolve_path(self.test_img_path) if hasattr(self.config_manager, 'resolve_path') else self.test_img_path
            self.test_frame = cv2.imread(abs_img_path)
            if self.test_frame is None:
                print(f"FAILED TO LOAD TEST IMAGE: {abs_img_path}", flush=True)
                self.use_test_image = False
            else:
                print(f"TEST IMAGE LOADED: {abs_img_path} {self.test_frame.shape}", flush=True)
        
        if not self.use_test_image:
            cam_cfg = self.config['hardware']['camera']
            cam_id = cam_cfg['device_id']
            cam_w = cam_cfg.get('width', 640)
            cam_h = cam_cfg.get('height', 480)
            self.camera = CameraStream(src=cam_id, width=cam_w, height=cam_h).start()
        else:
            self.camera = None
        
        try:
            self.serial_port = serial.Serial(
                port=self.config['hardware']['serial']['port'],
                baudrate=self.config['hardware']['serial']['baudrate'],
                timeout=0.1
            )
        except Exception as e:
            self.serial_port = None

    def _init_models(self):
        print("\n" + "═"*50, flush=True)
        print(f"  [SYSTEM MANAGER] LOAD MODE: {self.load_mode}", flush=True)
        
        # Mode 2 or 3: Load V1 (SpearHead)
        if self.load_mode in [2, 3]:
            v1_cfg = self.config['v1_model']
            v1_engine = v1_cfg['yolo_engine']
            print(f"  Loading V1 Engine: {v1_engine} on {v1_cfg['device']}...", flush=True)
            self.v1_vision = RobotVision(v1_engine, device=v1_cfg['device'])
            print(f"  - Model V1 Load (SpearHead): SUCCESS", flush=True)
        else:
            print(f"  - Model V1 (SpearHead): SKIPPED (load_mode={self.load_mode})", flush=True)
        
        # Mode 1 or 3: Load V2 (KFS)
        if self.load_mode in [1, 3]:
            v2_cfg = self.config['v2_model']
            v2_yolo_engine = v2_cfg['yolo_engine']
            print(f"  Loading V2 YOLO: {v2_yolo_engine} on {v2_cfg['yolo_device']}...", flush=True)
            self.v2_vision = RobotVision(v2_yolo_engine, device=v2_cfg['yolo_device'])
            
            v2_cnn_engine = v2_cfg['cnn_engine']
            print(f"  Loading V2 CNN: {v2_cnn_engine} on {v2_cfg['cnn_device']}...", flush=True)
            self.v2_cnn = TRTEngine(v2_cnn_engine)
            
            v2_labels = v2_cfg['labels_json']
            with open(v2_labels, 'r') as f:
                self.v2_labels = {int(v): k for k, v in json.load(f).items()}
            print(f"  - Model V2 Load (KFS): SUCCESS", flush=True)
            self.v2_smoother = LabelSmoother(window_size=7)
        else:
            print(f"  - Model V2 (KFS): SKIPPED (load_mode={self.load_mode})", flush=True)
        
        # Lock state for single-mode runs
        if self.load_mode == 1:
            self.state = 2  # Force KFS state
            print(f"  State locked to 2 (KFS) for load_mode=1", flush=True)
        elif self.load_mode == 2:
            self.state = 1  # Force SpearHead state
            print(f"  State locked to 1 (SpearHead) for load_mode=2", flush=True)
        
        print("" + "═"*50 + "\n", flush=True)

    def _get_valid_candidates(self, dets, frame_w, frame_h):
        """
        Tier 1: Geometric Filtering.
        Filters out obvious noise and returns valid candidates sorted from Left to Right.
        """
        if not dets:
            return []
            
        candidates = []
        for d in dets:
            x1, y1, x2, y2 = map(int, d.xyxy[0])
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)
            
            # 1. Physical Filtering: Target must be in a realistic area
            # Avoid extreme screen edges for robustness
            if cy < frame_h * 0.05 or cy > frame_h * 0.95: continue
            # Target should not be too tiny (noise) or too massive
            # Reduced min_area to 0.1% (300 pixels on VGA)
            if area < (frame_w * frame_h * 0.001) or area > (frame_w * frame_h * 0.9): continue
            
            # Store (cx, cy, det) for left+top sorting
            candidates.append((cx, cy, d))
            
        if not candidates:
            return []
            
        # Sort: bin X (left priority), then Y ascending (top priority within same column)
        bin_sz = self.x_bin_size
        candidates.sort(key=lambda c: (c[0] // bin_sz, c[1]))
        return [c[2] for c in candidates]

    def _warmup_idle_engines(self):
        """Send dummy tensor through idle engines to keep GPU memory hot."""
        if self.load_mode != 3:
            return
        try:
            if self.state != 1 and self.v1_vision:
                # V1 is idle, warm it up
                dummy = np.zeros((1, 3, 512, 512), dtype=np.float32)
                self.v1_vision.model.predict(dummy)
            if self.state != 2 and self.v2_cnn:
                # V2 CNN is idle, warm it up
                cnn_sz = self.config['v2_model']['cnn_input_size']
                dummy_cnn = np.zeros((1, 3, cnn_sz, cnn_sz), dtype=np.float32)
                self.v2_cnn.predict(dummy_cnn)
        except Exception as e:
            logger.debug(f"Warm-up pass: {e}")

    def _inference_loop(self):
        """Background thread for continuous AI processing."""
        inf_last_t = time.time()
        inf_frames = 0
        warmup_counter = 0
        self.inference_fps = 0.0
        
        while self.inference_running:
            if self.use_test_image:
                frame = self.test_frame.copy()
            elif self.camera and not self.camera.stopped:
                # Use read_latest() to always get the newest frame, dropping stale ones
                frame = self.camera.read_latest()
            else:
                frame = None
            
            if frame is None:
                time.sleep(0.01); continue
            
            # --- Inference Logic ---
            res = (None, "NONE")
            all_dets = []
            
            h_in, w_in = frame.shape[:2]
            
            if self.state == 1 and self.v1_vision: # V1 SpearHead
                dets = self.v1_vision.predict(frame, conf_threshold=self.config['v1_model']['conf_threshold'])
                candidates = self._get_valid_candidates(dets, w_in, h_in)
                if candidates:
                    target_det = candidates[0] # Leftmost
                    cx, cy = int((target_det.xyxy[0][0] + target_det.xyxy[0][2]) // 2), int((target_det.xyxy[0][1] + target_det.xyxy[0][3]) // 2)
                    res = (cx, cy), "TARGET"
                    
                    if self.use_test_image:
                        for d in dets:
                            x1, y1, x2, y2 = map(int, d.xyxy[0])
                            label = "TARGET" if d == target_det else "CANDIDATE"
                            all_dets.append(([x1, y1, x2, y2], label, d.conf))
                
            elif self.state == 2 and self.v2_vision: # V2 KFS
                dets = self.v2_vision.predict(frame, conf_threshold=self.config['v2_model']['conf_threshold_yolo'])
                valid_candidates = self._get_valid_candidates(dets, w_in, h_in)
                
                target_found = False
                for target_det in valid_candidates:
                    if target_found: break
                    
                    x1, y1, x2, y2 = map(int, target_det.xyxy[0])
                    roi = frame[max(0, y1):min(h_in, y2), max(0, x1):min(w_in, x2)]
                    input_data = preprocess_roi_for_cnn(roi, input_size=self.config['v2_model']['cnn_input_size'])
                    
                    if input_data is not None:
                        # TRTEngine.predict returns a list of outputs
                        cnn_outputs = self.v2_cnn.predict(input_data)
                        cnn_res = np.squeeze(cnn_outputs[0])
                        idx = np.argmax(cnn_res)
                        label_raw = self.v2_labels.get(idx, "UNK").upper()
                        score = float(cnn_res[idx])
                        
                        # Flexible target matching (handles REAL_xx, R1, etc)
                        target_names = [t.upper() for t in self.config['v2_model']['target_types']]
                        is_target = any(label_raw.startswith(t) or label_raw == t for t in target_names)
                        
                        # Special case: label history should only update for the 'best target candidate' found
                        # To simplify, we smooth the FINAL target label in the loop.
                        
                        if is_target and score >= self.config['v2_model']['conf_threshold_cnn']:
                            # Update label history for classification stability
                            self.label_history.append(label_raw)
                            if len(self.label_history) > self.history_len:
                                self.label_history.pop(0)
                            
                            most_common = max(set(self.label_history), key=self.label_history.count)
                            votes = self.label_history.count(most_common)
                            
                            if votes >= self.min_majority:
                                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                                label, _ = self.v2_smoother.smooth("target", most_common, score)
                                res = (cx, cy), label
                                target_found = True
                            else:
                                logger.debug(f"Target {most_common} found but needs more frames ({votes}/{self.min_majority})")
                        elif is_target:
                            logger.debug(f"Target {label_raw} rejected by CNN threshold: {score:.2f} < {self.config['v2_model']['conf_threshold_cnn']}")
                        
                        if self.use_test_image:
                            all_dets.append(([x1, y1, x2, y2], label_raw, score))
                
                if not target_found:
                    # Optional: fill all_dets with remaining YOLO boxes if in test mode
                    if self.use_test_image:
                        for d in dets:
                            if not any(np.array_equal(d.xyxy[0], c.xyxy[0]) for c in valid_candidates):
                                x1v, y1v, x2v, y2v = map(int, d.xyxy[0])
                                all_dets.append(([x1v, y1v, x2v, y2v], "YOLO", float(d.conf)))
                    
                    # Improved logic: only pop history if NO valid candidates are present
                    if len(valid_candidates) == 0 and len(self.label_history) > 0:
                        self.label_history.pop(0)

            with self.inference_lock:
                self.latest_inference_data = (res, all_dets)
            
            inf_frames += 1
            warmup_counter += 1
            curr_t = time.time()
            if curr_t - inf_last_t >= 0.5:
                self.inference_fps = inf_frames / (curr_t - inf_last_t)
                inf_frames, inf_last_t = 0, curr_t
            
            # Periodically warm up idle engines (every ~30 cycles)
            if warmup_counter >= 30:
                self._warmup_idle_engines()
                warmup_counter = 0
                
            # No sleep — run inference at max throughput

    def run(self):
        print(f"--- RUN LOOP START | HEADLESS: {self.headless} ---", flush=True)
        self.last_fps_update_time = time.time()
        self.frame_count_since_update = 0
        self.display_fps = 0.0

        self.loss_counter = 0
        self.max_loss_frames = self.config['detection']['max_loss_frames']
        self.current_label = "NONE"
        self.current_status = "SEARCHING"
        # inference_skip removed — process every frame
        self.all_dets = []

        try:
            while self.inference_running:
                loop_start = time.time()
                if self.use_test_image:
                    frame = self.test_frame.copy()
                elif self.camera and not self.camera.stopped:
                    frame = self.camera.read()
                else:
                    break
                    
                if frame is None: continue
                
                h_orig, w_orig = frame.shape[:2]

                # Consume latest AI results
                has_new_inference = False
                new_pt, a_label = None, "NONE"
                with self.inference_lock:
                    if self.latest_inference_data:
                        (new_pt, a_label), self.all_dets = self.latest_inference_data
                        self.latest_inference_data = None
                        has_new_inference = True
                
                # Kalman Tracking & Smoothing
                current_vision = None
                if self.state == 1:
                    current_vision = self.v1_vision
                elif self.state == 2:
                    current_vision = self.v2_vision
                
                if current_vision:
                    if has_new_inference:
                        if new_pt:
                            # Always feed measurement to Kalman (no hysteresis blocking)
                            tx, ty = current_vision.update_kalman(new_pt[0], new_pt[1])
                            
                            # Log target lock
                            if not self.last_filtered_pt:
                                logger.info(f"Target LOCKED at: ({tx}, {ty}) with label '{a_label}'")
                            
                            self.last_filtered_pt = (tx, ty)
                            self.current_status = "LOCKED"
                            if a_label != "NONE" or self.current_label == "NONE":
                                self.current_label = a_label
                            self.loss_counter = 0
                        else:
                            self.loss_counter += 1
                            if self.loss_counter < self.max_loss_frames and self.last_filtered_pt:
                                # Coast with Kalman prediction for a few frames
                                tx, ty = current_vision.update_kalman()
                            else:
                                tx, ty = w_orig // 2, h_orig // 2
                                if self.loss_counter == self.max_loss_frames:
                                    logger.info("Target LOST - returning to center")
                    else:
                        # AI is busy, stay at the last stable position
                        if self.last_filtered_pt:
                            tx, ty = self.last_filtered_pt
                        else:
                            tx, ty = w_orig // 2, h_orig // 2
                else:
                    tx, ty = w_orig // 2, h_orig // 2
                    
                # Status Hysteresis (Stay LOCKED if we just missed a few AI frames)
                if self.loss_counter <= 1:
                    self.current_status = "LOCKED"
                elif self.loss_counter < self.max_loss_frames:
                    self.current_status = "SEARCHING"
                else:
                    self.current_status = "LOST"
                
                if self.current_status == "LOST":
                    self.current_label = "NONE"
                    tx, ty = w_orig // 2, h_orig // 2
                
                status = self.current_status
                label = self.current_label
                
                # Display Mapping
                if self.force_square:
                    imgsz = 512
                    _, scale, (pw, ph) = letterbox(frame, (imgsz, imgsz))
                    dtx, dty = int(tx * scale + pw), int(ty * scale + ph)
                    sc_x = imgsz // 2
                    target_point = (dtx, dty)
                    curr_h, curr_w = imgsz, imgsz
                else:
                    sc_x = w_orig // 2
                    target_point = (tx, ty)
                    curr_h, curr_w = h_orig, w_orig
                    scale = 1.0; pw = ph = 0
                
                # Error Calculation & Serial (UART Capping)
                err_x = int(target_point[0] - sc_x) if status in ["LOCKED", "SEARCHING"] else 999
                
                curr_t = time.time()
                if self.serial_port and (curr_t - self.last_uart_time >= self.uart_interval):
                    self.serial_port.write(f"{err_x}\n".encode())
                    self.last_uart_time = curr_t
                    
                    # UART Read — always flush buffer to prevent serial overflow
                    # Only apply state changes in load_mode 3
                    if self.serial_port.in_waiting > 0:
                        try:
                            cmd_data = self.serial_port.read(self.serial_port.in_waiting).decode('utf-8', errors='ignore')
                            if self.load_mode == 3:
                                for char in reversed(cmd_data):
                                    if char in ['0', '1', '2']:
                                        new_st = int(char)
                                        if new_st != self.state:
                                            self.state = new_st
                                            logger.info(f"UART Mode Sync: {self.state}")
                                        break
                        except Exception as e:
                            pass # Silently handle decode errors

                
                # FPS Calculation
                self.frame_count_since_update += 1
                curr_t = time.time()
                if curr_t - self.last_fps_update_time >= 0.5:
                    self.display_fps = self.frame_count_since_update / (curr_t - self.last_fps_update_time)
                    self.frame_count_since_update, self.last_fps_update_time = 0, curr_t

                # UI
                if not self.headless:
                    df = frame if not self.force_square else letterbox(frame, (512, 512))[0]
                    # Color Unification: Both LOCKED and SEARCHING are Green (0, 255, 0)
                    color = (0, 255, 0) if status in ["LOCKED", "SEARCHING"] else (0, 0, 255)
                    
                    # Draw All Boxes in Test Mode
                    if self.use_test_image and self.all_dets:
                        for box, b_label, b_score in self.all_dets:
                            # Map box to display coords
                            bx1, by1, bx2, by2 = box
                            dbx1, dby1 = int(bx1 * scale + pw), int(by1 * scale + ph)
                            dbx2, dby2 = int(bx2 * scale + pw), int(by2 * scale + ph)
                            
                            cv2.rectangle(df, (dbx1, dby1), (dbx2, dby2), (0, 255, 0), 1)
                            cv2.putText(df, f"{b_label} {b_score:.2f}", (dbx1, dby1 - 5), 0, 0.4, (0, 255, 0), 1)

                    # Target Point and Line
                    cv2.line(df, (sc_x, curr_h), target_point, color, 2)
                    cv2.circle(df, target_point, 10, color, -1)
                    
                    # Status Header
                    cv2.rectangle(df, (0, 0), (curr_w, 35), (40, 40, 40), -1)
                    cv2.putText(df, f"MODE:{self.state} | {status} | {label} | EX:{err_x}", (10, 25), 0, 0.6, (255, 255, 255), 1)
                    fps_txt = f"SYS:{self.display_fps:.1f} | AI:{getattr(self, 'inference_fps', 0):.1f}"
                    cv2.putText(df, fps_txt, (10, 55), 0, 0.6, (200, 200, 200), 2)
                    
                    cv2.imshow("SystemManager", df)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'): break
                    elif key in [ord('1'), ord('2'), ord('0')]:
                        self.state = int(chr(key))
                        logger.info(f"Switched state to: {self.state}")
                
                # FPS capping removed — run at max throughput
        finally:
            self.cleanup()

    def cleanup(self):
        self.inference_running = False
        if hasattr(self, 'inf_thread'): self.inf_thread.join(timeout=1.0)
        if self.camera: self.camera.stop()
        if self.serial_port: self.serial_port.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    SystemManager().run()
