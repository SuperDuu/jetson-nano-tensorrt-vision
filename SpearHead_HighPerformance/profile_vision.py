"""
Model Performance Profiling Script - Detailed Version for SpearHead_HighPerformance
Đo thời gian inference của YOLO để xác định bottleneck.
"""

import cv2
import numpy as np
import time
import statistics
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Add project root to path to allow importing from core
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from core.vision import RobotVision
from core.config_manager import ConfigManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ModelProfiler:
    """Profiles YOLO model performance."""
    
    def __init__(self, config_path: str = None, force_device: str = None):
        """Initialize profiler with models."""
        self.config = ConfigManager(config_path)
        self.force_device = force_device
        self._init_models()
    
    def _init_models(self):
        """Initialize YOLO model."""
        # Load YOLO model
        yolo_engine = self.config.get_path("paths.models.yolo_engine")
        if not yolo_engine:
             yolo_engine = self.config.get_path("paths.models.yolo_xml")
             yolo_engine = str(Path(yolo_engine).with_suffix('.engine'))
             
        yolo_class_id = self.config.get("models.yolo.class_id", 0)
        logger.info(f"Loading YOLO model (TensorRT): {yolo_engine}")
        
        self.vision = RobotVision(yolo_engine, class_id=yolo_class_id, device="GPU")

    def profile_yolo(self, frame: np.ndarray, num_iterations: int = 100, warmup: int = 20) -> dict:
        """Profile YOLO inference time."""
        logger.info(f"\n{'='*60}")
        logger.info("Profiling YOLO Model")
        logger.info(f"{'='*60}")
        logger.info(f"Frame shape: {frame.shape}")
        
        conf_threshold = self.config.get("models.yolo.conf_threshold", 0.4)
        input_size = self.config.get("models.yolo.input_size", 512)
        
        # Warmup
        logger.info(f"Warming up ({warmup} iterations) to stabilize hardware clocks...")
        start_warmup = time.time()
        for _ in range(warmup):
            _ = self.vision.predict(frame, conf_threshold=conf_threshold, imgsz=input_size)
        
        # Profile
        logger.info(f"Warmup completed in {time.time() - start_warmup:.2f} seconds.")
        logger.info(f"Profiling ({num_iterations} iterations) with input_size={input_size}...")
        times = []
        num_detections = []
        
        for i in range(num_iterations):
            start = time.perf_counter()
            detections = self.vision.predict(frame, conf_threshold=conf_threshold, imgsz=input_size)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
            num_detections.append(len(detections))
        
        # Calculate statistics
        # Remove Top 5% slowest iterations as outliers (often OS jitter or first-frame JITs)
        sorted_times = sorted(times)
        trim_idx = max(1, int(len(sorted_times) * 0.95))
        trimmed_times = sorted_times[:trim_idx]
        
        stats = {
            'mean': statistics.mean(times),
            'trimmed_mean': statistics.mean(trimmed_times),
            'median': statistics.median(times),
            'min': min(times),
            'max': max(times),
            'std': statistics.stdev(times) if len(times) > 1 else 0,
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99),
            'fps': 1000.0 / statistics.mean(trimmed_times),
            'avg_detections': statistics.mean(num_detections),
            'iterations': num_iterations
        }
        
        logger.info(f"\nYOLO Performance Results:")
        logger.info(f"  Mean:          {stats['trimmed_mean']:.3f} ms")
        logger.info(f"  Median:        {stats['median']:.3f} ms")
        logger.info(f"  Min:           {stats['min']:.3f} ms")
        logger.info(f"  Max:           {stats['max']:.3f} ms")
        logger.info(f"  Std Dev:       {stats['std']:.3f} ms")
        logger.info(f"  P95:           {stats['p95']:.3f} ms")
        logger.info(f"  P99:           {stats['p99']:.3f} ms")
        logger.info(f"  FPS:           {stats['fps']:.1f}")
        logger.info(f"  Avg Detections: {stats['avg_detections']:.1f}")
        
        return stats

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile YOLO model performance")
    parser.add_argument("--image", type=str, default=None, help="Path to test image")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warmup iterations")
    
    args = parser.parse_args()
    
    try:
        config_path = args.config if args.config else str(Path(__file__).resolve().parent / "config.yaml")
        logger.info(f"\n{'#'*70}\n### RUNNING ON DEVICE: GPU (TensorRT)\n{'#'*70}")
        profiler = ModelProfiler(config_path=config_path)
    
        if args.image:
            frame = cv2.imread(args.image)
            if frame is None: raise ValueError(f"Failed to load image: {args.image}")
        else:
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            logger.info("Using dummy frame (640x480)")
    
        profiler.profile_yolo(frame=frame, num_iterations=args.iterations, warmup=args.warmup)
    
        logger.info(f"\n{'='*60}\nAll Profiling completed!\n{'='*60}")
    except Exception as e:
        logger.error(f"Error during profiling: {e}", exc_info=True)
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
