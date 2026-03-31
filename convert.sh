#!/bin/bash

# Default Paths
DEFAULT_MODEL="models/yolo_kfs.pt"

# Usage Check & Argument Parsing
MODEL_PATH=${1:-$DEFAULT_MODEL}
IMGSZ=${2:-512}

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file $MODEL_PATH not found."
    echo "Usage: $0 [model_path] [imgsz]"
    exit 1
fi

echo "Starting conversion for $MODEL_PATH with imgsz=$IMGSZ..."

# Run the python conversion script
python3 core/convert_model.py "$MODEL_PATH" --imgsz "$IMGSZ"

if [ $? -eq 0 ]; then
    echo "Conversion completed successfully!"
else
    echo "Conversion failed!"
    exit 1
fi
