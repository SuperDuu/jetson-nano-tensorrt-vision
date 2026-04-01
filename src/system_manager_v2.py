"""
System Manager V2 — GPU-Accelerated Pipeline for Jetson Nano.

Integrates:
  - GStreamer camera (hardware ISP)
  - GPU preprocessing (PyCUDA kernel)
  - TensorRT V2 engine (zero-copy inference)
  - Double-buffer async: GPU infers frame N while CPU post-processes frame N-1
  - Multiprocessing display (cv2.imshow offloaded)

Compatible with Python 3.6+ / Jetson Nano L4T r32.7.1.
"""

import cv2
import numpy as np
import time
import logging
import json
import serial
import sys
import os
from pathlib import Path

# Project root setup
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.config_manager import ConfigManager
from core.vision_v2 import RobotVisionV2
from core.gst_camera import GstCameraStream
from core.label_smoother import LabelSmoother
from core.utils import letterbox, preprocess_roi_for_cnn
from core.trt_engine_v2 import TRTEngineV2
from core.async_display import DisplayProcess

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("SystemManagerV2")


class SystemManagerV2(object):
    """
    GPU-optimized system manager with double-buffer pipeline.

    Main loop flow:
        1. Read frame
        2. Launch GPU inference (async: preprocess + TRT + d2h copy)
        3. While GPU works → CPU post-processes PREVIOUS frame
        4. Sync GPU → store raw output for next iteration
        5. Send display frame to DisplayProcess (non-blocking)
    """

    def __init__(self, config_path="global_config.yaml"):
        print("--- INIT V2 START: {} ---".format(config_path), flush=True)

        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config

        self.load_mode = self.config['system'].get('load_mode', 3)
        self.state = self.config['system']['initial_state']
        self.force_square = self.config['system']['force_square']
        self.headless = self.config['system']['headless']

        # Model references
        self.v1_vision = None  # type: RobotVisionV2
        self.v2_vision = None  # type: RobotVisionV2
        self.v2_cnn = None     # type: TRTEngineV2
        self.v2_labels = {}
        self.v2_smoother = None

        # Main loop control
        self.running = True

        # Filtering & Locking
        self.locked_target_id = None
        self.label_history = []
        self.history_len = 5
        self.min_majority = 2
        self.x_bin_size = 60

        # UART
        self.last_uart_time = 0
        self.uart_interval = 1.0 / 30.0
        self.last_filtered_pt = None

        try:
            print("  Initializing hardware (GStreamer)...", flush=True)
            self._init_hardware()
            print("  Initializing models (GPU V2)...", flush=True)
            self._init_models()

            # Display process (separate OS process)
            self.display = None
            if not self.headless:
                self.display = DisplayProcess("RBC2026 V2").start()
                print("  DisplayProcess started.", flush=True)

            print("  V2 Ready (double-buffer mode).", flush=True)
            logger.info("SystemManagerV2 initialized. State: %d", self.state)
        except Exception as e:
            logger.error("V2 Init Failed: %s", e)
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def _init_hardware(self):
        self.use_test_image = self.config['system'].get('test_image', False)
        if self.use_test_image:
            img_path = self.config['system']['test_image_path']
            self.test_frame = cv2.imread(img_path)
            if self.test_frame is None:
                print("FAILED TO LOAD TEST IMAGE: {}".format(img_path), flush=True)
                self.use_test_image = False
            else:
                print("TEST IMAGE: {} {}".format(img_path, self.test_frame.shape), flush=True)

        if not self.use_test_image:
            cam_cfg = self.config['hardware']['camera']
            cam_id = cam_cfg['device_id']
            cam_w = cam_cfg.get('width', 640)
            cam_h = cam_cfg.get('height', 480)
            cam_type = cam_cfg.get('type', 'auto')
            self.camera = GstCameraStream(
                src=cam_id, width=cam_w, height=cam_h, camera_type=cam_type
            ).start()
        else:
            self.camera = None

        try:
            self.serial_port = serial.Serial(
                port=self.config['hardware']['serial']['port'],
                baudrate=self.config['hardware']['serial']['baudrate'],
                timeout=0.1,
            )
        except Exception:
            self.serial_port = None

    def _init_models(self):
        print("\n" + "=" * 50, flush=True)
        print("  [V2] LOAD MODE: {}".format(self.load_mode), flush=True)

        imgsz = 512  # default

        # V1 (SpearHead)
        if self.load_mode in (2, 3):
            v1_cfg = self.config['v1_model']
            v1_engine = v1_cfg['yolo_engine']
            imgsz = v1_cfg.get('input_size', 512)
            print("  Loading V1 Engine (GPU V2): {}".format(v1_engine), flush=True)
            self.v1_vision = RobotVisionV2(v1_engine, imgsz=imgsz, device="GPU")
            print("  - V1 (SpearHead): SUCCESS", flush=True)

        # V2 (KFS)
        if self.load_mode in (1, 3):
            v2_cfg = self.config['v2_model']
            v2_yolo = v2_cfg['yolo_engine']
            imgsz = v2_cfg.get('yolo_input_size', 512)
            print("  Loading V2 YOLO (GPU V2): {}".format(v2_yolo), flush=True)
            self.v2_vision = RobotVisionV2(v2_yolo, imgsz=imgsz, device="GPU")

            v2_cnn = v2_cfg['cnn_engine']
            print("  Loading V2 CNN: {}".format(v2_cnn), flush=True)
            self.v2_cnn = TRTEngineV2(v2_cnn)

            with open(v2_cfg['labels_json'], 'r') as f:
                self.v2_labels = {int(v): k for k, v in json.load(f).items()}
            self.v2_smoother = LabelSmoother(window_size=7)
            print("  - V2 (KFS): SUCCESS", flush=True)

        # Lock state for single-mode
        if self.load_mode == 1:
            self.state = 2
        elif self.load_mode == 2:
            self.state = 1

        print("=" * 50 + "\n", flush=True)

    def _get_current_vision(self):
        """Get the active RobotVisionV2 for current state."""
        if self.state == 1:
            return self.v1_vision
        elif self.state == 2:
            return self.v2_vision
        return None

    def _get_valid_candidates(self, dets, frame_w, frame_h):
        """Tier 1: Geometric filtering. Returns sorted candidates (left-to-right)."""
        if not dets:
            return []
        candidates = []
        for d in dets:
            x1, y1, x2, y2 = map(int, d.xyxy[0])
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            area = (x2 - x1) * (y2 - y1)
            if cy < frame_h * 0.05 or cy > frame_h * 0.95:
                continue
            if area < (frame_w * frame_h * 0.001) or area > (frame_w * frame_h * 0.9):
                continue
            candidates.append((cx, cy, d))
        if not candidates:
            return []
        bin_sz = self.x_bin_size
        candidates.sort(key=lambda c: (c[0] // bin_sz, c[1]))
        return [c[2] for c in candidates]

    def _postprocess_detections(self, raw_outputs, vision, frame_hw, conf_threshold):
        """
        CPU post-processing: decode TRT output → DetectedObject list.
        Uses vision's inherited postprocess logic.
        """
        if raw_outputs is None:
            return []
        scale = raw_outputs['scale']
        pad_x = raw_outputs['pad_x']
        pad_y = raw_outputs['pad_y']
        return vision.postprocess_raw(
            raw_outputs['tensors'], conf_threshold, scale, pad_x, pad_y
        )

    def _apply_tracking(self, dets, frame, vision, h_orig, w_orig):
        """
        Run target selection, CNN classification (V2), and return (point, label).
        """
        if self.state == 1 and vision:
            candidates = self._get_valid_candidates(dets, w_orig, h_orig)
            if candidates:
                det = candidates[0]
                cx = int((det.xyxy[0][0] + det.xyxy[0][2]) // 2)
                cy = int((det.xyxy[0][1] + det.xyxy[0][3]) // 2)
                return (cx, cy), "TARGET"

        elif self.state == 2 and vision:
            candidates = self._get_valid_candidates(dets, w_orig, h_orig)
            for det in candidates:
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                roi = frame[max(0, y1):min(h_orig, y2), max(0, x1):min(w_orig, x2)]
                cnn_input_size = self.config['v2_model']['cnn_input_size']
                input_data = preprocess_roi_for_cnn(roi, input_size=cnn_input_size)
                if input_data is None:
                    continue

                cnn_out = self.v2_cnn.predict(input_data)
                cnn_res = np.squeeze(cnn_out[0])
                idx = np.argmax(cnn_res)
                label_raw = self.v2_labels.get(idx, "UNK").upper()
                score = float(cnn_res[idx])

                target_names = [t.upper() for t in self.config['v2_model']['target_types']]
                is_target = any(label_raw.startswith(t) or label_raw == t for t in target_names)

                if is_target and score >= self.config['v2_model']['conf_threshold_cnn']:
                    self.label_history.append(label_raw)
                    if len(self.label_history) > self.history_len:
                        self.label_history.pop(0)

                    most_common = max(set(self.label_history), key=self.label_history.count)
                    votes = self.label_history.count(most_common)

                    if votes >= self.min_majority:
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        label, _ = self.v2_smoother.smooth("target", most_common, score)
                        return (cx, cy), label

            # No target found — decay history
            if len(self.label_history) > 0:
                self.label_history.pop(0)

        return None, "NONE"

    def run(self):
        """
        Double-buffer main loop:
          GPU processes frame N while CPU post-processes frame N-1.
        """
        print("--- V2 RUN LOOP START | HEADLESS: {} ---".format(self.headless), flush=True)

        last_fps_time = time.time()
        frame_count = 0
        display_fps = 0.0

        loss_counter = 0
        max_loss = self.config['detection']['max_loss_frames']
        current_label = "NONE"
        current_status = "SEARCHING"

        # Double-buffer state
        prev_raw = None   # Previous frame's raw TRT output + metadata
        prev_frame = None  # Previous frame (for CNN ROI)
        prev_vision = None  # Vision object used for previous frame

        try:
            while self.running:
                # ── 1. Read Frame ─────────────────────────────────
                if self.use_test_image:
                    frame = self.test_frame.copy()
                elif self.camera and not self.camera.stopped:
                    frame = self.camera.read_latest()
                else:
                    break

                if frame is None:
                    continue

                h_orig, w_orig = frame.shape[:2]
                vision = self._get_current_vision()

                # ── 2. Launch ASYNC GPU inference for current frame ─
                if vision:
                    conf = self._get_conf_threshold()
                    meta = vision.launch_inference(frame)
                else:
                    meta = None

                # ── 3. CPU: Post-process PREVIOUS frame (overlapped) ──
                if prev_raw is not None and prev_vision is not None:
                    dets = prev_vision.postprocess_raw(
                        prev_raw['tensors'],
                        self._get_conf_threshold_for_state(prev_raw['state']),
                        prev_raw['scale'], prev_raw['pad_x'], prev_raw['pad_y'],
                    )
                    new_pt, a_label = self._apply_tracking(
                        dets, prev_frame, prev_vision,
                        prev_raw['h'], prev_raw['w'],
                    )

                    # Kalman update
                    if new_pt:
                        tx, ty = prev_vision.update_kalman(new_pt[0], new_pt[1])
                        if not self.last_filtered_pt:
                            logger.info("Target LOCKED at (%d, %d) label='%s'", tx, ty, a_label)
                        self.last_filtered_pt = (tx, ty)
                        current_status = "LOCKED"
                        if a_label != "NONE" or current_label == "NONE":
                            current_label = a_label
                        loss_counter = 0
                    else:
                        loss_counter += 1
                        if loss_counter < max_loss and self.last_filtered_pt:
                            tx, ty = prev_vision.update_kalman()
                        else:
                            tx, ty = w_orig // 2, h_orig // 2
                            if loss_counter == max_loss:
                                logger.info("Target LOST - returning to center")

                    # Status
                    if loss_counter <= 1:
                        current_status = "LOCKED"
                    elif loss_counter < max_loss:
                        current_status = "SEARCHING"
                    else:
                        current_status = "LOST"
                        current_label = "NONE"
                        tx, ty = w_orig // 2, h_orig // 2

                    # Display mapping
                    if self.force_square:
                        imgsz = 512
                        _, scale_d, (pw, ph) = letterbox(prev_frame, (imgsz, imgsz))
                        dtx, dty = int(tx * scale_d + pw), int(ty * scale_d + ph)
                        sc_x = imgsz // 2
                        target_point = (dtx, dty)
                    else:
                        sc_x = w_orig // 2
                        target_point = (tx, ty)

                    err_x = int(target_point[0] - sc_x) if current_status in ("LOCKED", "SEARCHING") else 999

                    # UART
                    curr_t = time.time()
                    if self.serial_port and (curr_t - self.last_uart_time >= self.uart_interval):
                        self.serial_port.write("{}\n".format(err_x).encode())
                        self.last_uart_time = curr_t

                        if self.serial_port.in_waiting > 0:
                            try:
                                cmd = self.serial_port.read(
                                    self.serial_port.in_waiting
                                ).decode('utf-8', errors='ignore')
                                if self.load_mode == 3:
                                    for ch in reversed(cmd):
                                        if ch in ('0', '1', '2'):
                                            ns = int(ch)
                                            if ns != self.state:
                                                self.state = ns
                                                logger.info("UART Mode Sync: %d", self.state)
                                            break
                            except Exception:
                                pass

                    # Send to DisplayProcess
                    if self.display and self.display.is_running():
                        if self.force_square:
                            df = letterbox(prev_frame, (512, 512))[0]
                        else:
                            df = prev_frame.copy()
                        self.display.send_frame(
                            df, target_point=target_point,
                            status=current_status, label=current_label,
                            error_x=err_x, fps=display_fps,
                        )

                # ── 4. Sync GPU — collect current frame's raw output ──
                if vision and meta is not None:
                    tensors = vision.collect_raw_output()
                    prev_raw = {
                        'tensors': tensors,
                        'scale': meta[0], 'pad_x': meta[1], 'pad_y': meta[2],
                        'h': h_orig, 'w': w_orig,
                        'state': self.state,
                    }
                    prev_frame = frame
                    prev_vision = vision
                else:
                    prev_raw = None
                    prev_frame = None
                    prev_vision = None

                # ── 5. FPS ────────────────────────────────────────
                frame_count += 1
                curr_t = time.time()
                if curr_t - last_fps_time >= 0.5:
                    display_fps = frame_count / (curr_t - last_fps_time)
                    print("FPS: {:.1f} | {} | {}".format(
                        display_fps, current_status, current_label
                    ), end='\r')
                    frame_count = 0
                    last_fps_time = curr_t

                # Check display quit
                if self.display and not self.display.is_running():
                    break

        finally:
            self.cleanup()

    def _get_conf_threshold(self):
        """Get confidence threshold for current state."""
        return self._get_conf_threshold_for_state(self.state)

    def _get_conf_threshold_for_state(self, state):
        if state == 1:
            return self.config['v1_model']['conf_threshold']
        elif state == 2:
            return self.config['v2_model']['conf_threshold_yolo']
        return 0.5

    def cleanup(self):
        self.running = False
        if self.display:
            self.display.stop()
        if self.camera:
            self.camera.stop()
        if self.serial_port:
            self.serial_port.close()
        print("\nV2 Cleanup complete.", flush=True)


if __name__ == "__main__":
    SystemManagerV2().run()
