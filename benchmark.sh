#!/bin/bash

# TensorRT Performance Evaluation Script for RBC2026

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_engine_file> [imgsz]"
    echo "Example: $0 models/yolo_kfs.engine 512"
    exit 1
fi

ENGINE_PATH=$1
IMGSZ=${2:-512}

if [ ! -f "$ENGINE_PATH" ]; then
    echo "Error: Engine file $ENGINE_PATH not found."
    exit 1
fi

echo "=========================================================="
echo "  Evaluating Performance: $ENGINE_PATH"
echo "  Input Size: $IMGSZ x $IMGSZ"
echo "=========================================================="

TRTEXEC="/usr/src/tensorrt/bin/trtexec"
if [ ! -x "$TRTEXEC" ]; then
    TRTEXEC="trtexec"
fi

# Run benchmark
# --loadEngine: Load the existing engine
# --avgRuns=100: Average over 100 runs for stability
# --warmUp=500: Warm up for 500ms
# --duration=10: Run for 10 seconds
# --noDataTransfers: (Optional) Focus only on compute if needed, but we want real world
$TRTEXEC --loadEngine="$ENGINE_PATH" \
         --avgRuns=100 \
         --warmUp=500 \
         --duration=10

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================================="
    echo "  Benchmark Completed Successfully"
    echo "=========================================================="
else
    echo ""
    echo "  Benchmark Failed!"
fi
