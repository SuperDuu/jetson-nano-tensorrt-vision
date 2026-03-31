"""
Unified Vision module for RBC2026.
Provides YOLO detection and advanced Kalman/EMA tracking.
"""

import cv2
import numpy as np
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional
from .trt_engine import TRTEngine
from .utils import letterbox

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CLASS_ID = 0
DEFAULT_INPUT_SIZE = 512
DEFAULT_CONF_THRESHOLD = 0.5
DEFAULT_NMS_THRESHOLD = 0.45
BACKGROUND_VALUE = 128
KALMAN_PROCESS_NOISE = 1.0
KALMAN_MEASUREMENT_NOISE = 2.0

class DetectedObject:
    def __init__(self, x1: int, y1: int, x2: int, y2: int, conf: float):
        self.xyxy = [np.array([x1, y1, x2, y2])]
        self.conf = conf

class RobotVision:
    def __init__(self, model_path: str, class_id: int = DEFAULT_CLASS_ID, device: str = "GPU"):
        self.class_id = class_id
        self.device = device
        self.logger = logging.getLogger(f"{__name__}.RobotVision")
        self.last_boxes = {} 
        self.alpha = 0.7
        try:
            engine_path = model_path if model_path.endswith('.engine') else f"{model_path}/best.engine"
            if not Path(engine_path).exists():
                raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
            
            self.model = TRTEngine(engine_path)
            self._init_kalman_filter()
        except Exception as e:
            self.logger.error(f"Failed to initialize RobotVision with TensorRT: {e}")
            raise

    def _init_kalman_filter(self) -> None:
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * KALMAN_PROCESS_NOISE
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * KALMAN_MEASUREMENT_NOISE
        self.kalman_initialized = False

    def predict(self, frame: np.ndarray, conf_threshold: float = DEFAULT_CONF_THRESHOLD, imgsz: int = DEFAULT_INPUT_SIZE) -> List[DetectedObject]:
        if frame is None: return []
        h_orig, w_orig = frame.shape[:2]
        canvas, scale, (pad_x, pad_y) = letterbox(frame, (imgsz, imgsz))
        input_data = canvas.transpose((2, 0, 1)).reshape((1, 3, imgsz, imgsz)).astype(np.float32) / 255.0

        outputs = self.model.predict(input_data)
        predictions = np.squeeze(outputs[0])  # Assuming primary output is the first binding

        # Guard: if inference failed or returned unexpected shape, bail out
        if predictions.ndim < 2:
            self.logger.warning("TRT output has unexpected shape: %s, skipping frame", predictions.shape)
            return []

        # ─── Tự detect output format ───────────────────────────
        # YOLO26n: [300, 6] → shape[1] == 6, đã NMS sẵn
        # YOLOv8n: [5, 5376] → shape[0] < shape[1], raw anchors
        is_yolo26_format = (predictions.ndim == 2 and 
                            predictions.shape[1] == 6 and 
                            predictions.shape[0] < 400)

        if is_yolo26_format:
            return self._postprocess_yolo26(predictions, conf_threshold, scale, pad_x, pad_y)
        else:
            return self._postprocess_yolov8(predictions, conf_threshold, scale, pad_x, pad_y)

    def _postprocess_yolo26(self, predictions, conf_threshold, scale, pad_x, pad_y):
        """YOLO26n: output [300, 6] = [x1, y1, x2, y2, conf, class_id] — vectorized"""
        
        # Vectorized filter thay vì for loop
        confs = predictions[:, 4]
        cls_ids = predictions[:, 5].astype(np.int32)
        valid_mask = (confs >= conf_threshold) & (cls_ids == self.class_id)
        valid = predictions[valid_mask]
        
        if len(valid) == 0:
            self.last_boxes = {}
            return []
        
        # Unpad + unscale toàn bộ array cùng lúc
        x1s = ((valid[:, 0] - pad_x) / scale).astype(np.int32)
        y1s = ((valid[:, 1] - pad_y) / scale).astype(np.int32)
        x2s = ((valid[:, 2] - pad_x) / scale).astype(np.int32)
        y2s = ((valid[:, 3] - pad_y) / scale).astype(np.int32)
        confs_valid = valid[:, 4]
        
        # Build results — EMA vẫn cần loop nhưng chỉ trên detected boxes thật
        final_boxes, new_last_boxes = [], {}
        for i in range(len(valid)):
            new_box = [x1s[i], y1s[i], x2s[i], y2s[i]]
            fx1, fy1, fx2, fy2, box_id = self._apply_ema(new_box)
            new_last_boxes[box_id] = [fx1, fy1, fx2, fy2]
            final_boxes.append(DetectedObject(fx1, fy1, fx2, fy2, float(confs_valid[i])))
        
        self.last_boxes = new_last_boxes
        return final_boxes

    def _postprocess_yolov8(self, predictions, conf_threshold, scale, pad_x, pad_y):
        """YOLOv8n: output [5, 5376] = [xc, yc, w, h, class_score] — cần NMS"""
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T  # → [5376, 5]

        scores_all = predictions[:, 4:]
        max_scores = np.max(scores_all, axis=1)
        cls_ids = np.argmax(scores_all, axis=1)
        valid_mask = (max_scores > conf_threshold) & (cls_ids == self.class_id)
        valid_preds = predictions[valid_mask]

        if len(valid_preds) == 0:
            self.last_boxes = {}
            return []

        xc, yc, w, h = valid_preds[:,0], valid_preds[:,1], valid_preds[:,2], valid_preds[:,3]
        x1 = ((xc - w/2 - pad_x) / scale).astype(np.int32)
        y1 = ((yc - h/2 - pad_y) / scale).astype(np.int32)
        raw_boxes = np.column_stack([x1, y1, (w/scale).astype(np.int32), (h/scale).astype(np.int32)]).tolist()
        confidences = max_scores[valid_mask].tolist()
        indices = cv2.dnn.NMSBoxes(raw_boxes, confidences, conf_threshold, DEFAULT_NMS_THRESHOLD)

        final_boxes, new_last_boxes = [], {}
        if len(indices) > 0:
            for i in indices.flatten():
                rx, ry, rw, rh = raw_boxes[i]
                new_box = [rx, ry, rx + rw, ry + rh]
                fx1, fy1, fx2, fy2, box_id = self._apply_ema(new_box)
                new_last_boxes[box_id] = [fx1, fy1, fx2, fy2]
                final_boxes.append(DetectedObject(fx1, fy1, fx2, fy2, confidences[i]))

        self.last_boxes = new_last_boxes
        return final_boxes

    def _apply_ema(self, new_box):
        """EMA smoothing — dùng chung cho cả 2 format"""
        best_iou, best_match_id = 0.0, None
        rx1, ry1, rx2, ry2 = new_box
        for prev_id, prev_box in self.last_boxes.items():
            xA = max(rx1, prev_box[0]); yA = max(ry1, prev_box[1])
            xB = min(rx2, prev_box[2]); yB = min(ry2, prev_box[3])
            inter = max(0, xB-xA) * max(0, yB-yA)
            areaA = (rx2-rx1) * (ry2-ry1)
            areaB = (prev_box[2]-prev_box[0]) * (prev_box[3]-prev_box[1])
            iou = inter / float(areaA + areaB - inter) if (areaA + areaB - inter) > 0 else 0
            if iou > best_iou:
                best_iou, best_match_id = iou, prev_id

        if best_iou > 0.3 and best_match_id:
            prev_box = self.last_boxes[best_match_id]
            fx1, fy1, fx2, fy2 = [int(self.alpha*c + (1-self.alpha)*p) for c, p in zip(new_box, prev_box)]
            box_id = best_match_id
        else:
            fx1, fy1, fx2, fy2 = new_box
            box_id = f"{rx1}_{ry1}_{time.time()}"

        return fx1, fy1, fx2, fy2, box_id

    def update_kalman(self, x: Optional[float] = None, y: Optional[float] = None) -> Tuple[int, int]:
        try:
            if x is not None and y is not None:
                if not self.kalman_initialized:
                    self.kalman.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
                    self.kalman.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
                    self.kalman_initialized = True
                    return int(x), int(y)
                
                # Predict → Correct cycle (đúng thứ tự)
                self.kalman.predict()
                self.kalman.correct(np.array([[np.float32(x)], [np.float32(y)]]))
                state = self.kalman.statePost.flatten()
                return int(state[0]), int(state[1])
            else:
                # No measurement — chỉ predict (coast)
                if not self.kalman_initialized:
                    return 0, 0
                pred = self.kalman.predict()
                px, py = pred.flatten()[:2]
                return int(px), int(py)
        except Exception as e:
            self.logger.error(f"Kalman error: {e}")
            return int(x) if x is not None else 0, int(y) if y is not None else 0
