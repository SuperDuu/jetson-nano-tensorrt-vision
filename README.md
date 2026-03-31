# Jetson Nano TensorRT Vision

High-performance vision pipeline for Robocon, optimized for NVIDIA Jetson Nano with TensorRT FP16 inference.

## Project Structure

```
├── core/                  # Core modules (reusable)
│   ├── camera.py          # Threaded camera with deque buffer
│   ├── config_manager.py  # YAML config loader with path resolution
│   ├── convert_model.py   # PT/H5 → ONNX → TensorRT engine
│   ├── trt_engine.py      # TensorRT inference wrapper
│   ├── vision.py          # YOLO detection + Kalman tracking
│   ├── label_smoother.py  # Classification stability filter
│   └── utils.py           # Letterbox, preprocessing helpers
├── src/
│   └── system_manager.py  # Main system: camera + AI + UART control
├── models/                # TensorRT engines (.engine) and weights (.pt)
├── global_config.yaml     # System configuration
├── convert.sh             # Quick model conversion script
├── Dockerfile             # Docker environment for Jetson
└── requirements.txt       # Python dependencies
```

## Quick Start

### 1. Convert Models

```bash
# Default: convert models/yolo_kfs.pt → .engine (FP16, 512x512)
./convert.sh

# Custom model
./convert.sh models/custom.pt 512
```

### 2. Configure

Edit `global_config.yaml`:

```yaml
system:
  load_mode: 3  # 1: KFS only, 2: SpearHead only, 3: All + UART switching
  initial_state: 2
```

### 3. Run

```bash
python3 src/system_manager.py
```

## Load Modes

| Mode | Models Loaded | UART Switching | Use Case |
|------|--------------|----------------|----------|
| 1 | KFS (YOLO + CNN) | Disabled | Testing KFS |
| 2 | SpearHead (YOLO) | Disabled | Testing SpearHead |
| 3 | All models | Enabled | Competition |

## Key Features

- **TensorRT FP16** inference on Jetson Nano GPU
- **Selective model loading** to save GPU memory
- **Deque camera buffer** — always processes the latest frame
- **GPU warm-up** — keeps idle engines hot for instant state switching
- **Kalman + EMA tracking** — smooth target following
- **UART control** — bidirectional communication with MCU
- **ONNX simplification** — optimized graph for faster engine builds

## Requirements

- JetPack 4.6+ (TensorRT 8.x, CUDA 10.2)
- Python 3.6+
- OpenCV, PyCUDA, PySerial
