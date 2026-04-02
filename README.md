# Jetson Nano TensorRT Vision V2

High-performance vision pipeline for Robocon, architecturally optimized for NVIDIA Jetson Nano with pure hardware-accelerated GStreamer, PyCUDA, and TensorRT FP16 APIs.

## Project Structure

```
├── core/                  # Core autonomous modules
│   ├── camera.py          # Threaded camera stream
│   ├── gst_camera.py      # Hardware-accelerated GStreamer stream (nvv4l2decoder, nvvidconv)
│   ├── config_manager.py  # YAML config loader
│   ├── trt_engine_v2.py   # TensorRT inference wrapper (Async Context)
│   ├── vision.py          # YOLO + CNN detection with EMA Kalman tracking
│   ├── cuda_preprocess.cu # C++ PyCUDA Kernel for raw BGRx manipulation
│   ├── gpu_preprocess.py  # GPU Pinned Memory allocator
│   ├── udp_streamer.py    # Hardware UDP network video streamer (nvv4l2h264enc)
│   └── async_display.py   # Zero-IPC DisplayThread OSD Drawer
├── src/
│   ├── system_manager_v2.py # Master 60Hz Control Loop
│   └── system_manager.py    # Legacy synchronous manager
├── models/                # TensorRT engines (.engine)
├── global_config.yaml     # System central configuration
└── FULL_PROJECT_ARCHITECTURE_JETSON.md # Comprehensive Technical Architecture Manual
```

## V2 Architectural Milestones

- **Hardware Video Decoding**: Drops `videoconvert` CPU operations. Leverages `nvv4l2decoder mjpeg=1` and `nvvidconv` to offload MJPEG camera feeds onto dedicated Silicon.
- **Zero-CPU PyCUDA Input**: Uses custom CUDA arrays (`uchar4`) to digest `BGRx` hardware memory maps natively, rendering Python CPU slicing obsolete.
- **Decoupled 60Hz Hard Timer**: Main loop detaches from the sensory frame lockstep. An EMA-filtered dynamic `dt` forces the Kalman filter to extrapolate and `Coast` independently if USB network interrupts occur, preventing control stutters.
- **Threaded OSD zero-IPC**: Replaces heavy `multiprocessing` CoW leaks with Native Python Threading and GIL-release (`cv2.waitKey`), saving over 2GB of RAM forks.
- **Hardware H.264 Telemetry**: `UDPStreamer` module transmits Headless realtime annotations using embedded GPU Jetson hardware (`nvv4l2h264enc`), costing nearly 0 CPU cycles.

## Quick Start

### 1. Configuration
Modify target states, Headless UI, or hardware configs in `global_config.yaml`:
```yaml
system:
  load_mode: 3      # 1: KFS only, 2: SpearHead only, 3: All + UART switching
  initial_state: 2  
  headless: false   # Disable GUI X11 window
  udp_stream: true  # Broadcast telemetry implicitly via UDP
```

### 2. Run Engine V2
```bash
python3 src/system_manager_v2.py
```

## Load Modes

| Mode | Models Loaded | Hardware | UART | Primary Use |
|------|--------------|----------|------|----------|
| 1 | KFS | GPU / NVMM | Excluded | Standalone test |
| 2 | SpearHead | GPU / NVMM | Excluded | Standalone test |
| 3 | All states | GPU / NVMM | Stream 50Hz | **Combat System** |

## Required Environment

- NVIDIA Jetson Nano (4GB) 
- JetPack 4.6.x (TensorRT 8.2, CUDA 10.2)
- Python 3.6+
- Dependants: `cv2`, `pycuda`, `pyserial`, `tensorrt`

## Documentation
Please view the extensive Vietnamese technical dictionary: `FULL_PROJECT_ARCHITECTURE_JETSON.md` for mathematically and conceptually exhaustive implementation logic.
