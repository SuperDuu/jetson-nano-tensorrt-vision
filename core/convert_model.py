import os
import subprocess
import argparse
from pathlib import Path

def convert_h5_to_onnx(h5_path, onnx_path):
    """Converts .h5 (Keras) to .onnx."""
    print(f"Converting {h5_path} to {onnx_path}...")
    import tensorflow as tf
    import tf2onnx
    model = tf.keras.models.load_model(h5_path)
    spec = (tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="input"),)
    tf2onnx.convert.from_keras(model, input_signature=spec, output_path=onnx_path)
    print(f"Successfully saved ONNX model to {onnx_path}")

def convert_pt_to_onnx(pt_path, onnx_path, imgsz=640):
    """Converts .pt (PyTorch) to .onnx."""
    print(f"Converting {pt_path} to {onnx_path}...")
    try:
        # Try Ultralytics first (for YOLOv8/v11)
        from ultralytics import YOLO
        model = YOLO(pt_path)
        model.export(format='onnx', imgsz=imgsz, dynamic=False)
        # Ultralytics saves it in the same dir as model.onnx
        src_onnx = Path(pt_path).with_suffix('.onnx')
        if str(src_onnx) != onnx_path:
            os.rename(src_onnx, onnx_path)
    except (ImportError, Exception):
        # Fallback to generic PyTorch
        import torch
        model = torch.load(pt_path, map_location='cpu')
        if hasattr(model, 'model'): model = model.model # Handle wrapped models
        model.eval()
        dummy_input = torch.randn(1, 3, imgsz, imgsz)
        torch.onnx.export(model, dummy_input, onnx_path, opset_version=12)
    
    print(f"Successfully saved ONNX model to {onnx_path}")

def convert_onnx_to_engine(onnx_path, engine_path, fp16=True):
    """Converts .onnx to .engine using trtexec."""
    print(f"Converting {onnx_path} to {engine_path} (FP16={fp16})...")
    trtexec_path = "/usr/src/tensorrt/bin/trtexec"
    if not os.path.exists(trtexec_path): trtexec_path = "trtexec"
    
    cmd = [trtexec_path, f"--onnx={onnx_path}", f"--saveEngine={engine_path}", "--workspace=1024"]
    if fp16: cmd.append("--fp16")
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Successfully created TensorRT engine at {engine_path}")
    else:
        print(f"Conversion failed!\n{result.stdout}\n{result.stderr}")

def main():
    parser = argparse.ArgumentParser(description="Convert .h5 or .pt model to TensorRT .engine")
    parser.add_argument("input", type=str, help="Path to input (.h5 or .pt) file")
    parser.add_argument("--output", type=str, default=None, help="Path to output .engine file")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default 640)")
    parser.add_argument("--no-fp16", action="store_false", dest="fp16", help="Disable FP16 precision")
    parser.set_defaults(fp16=True)
    
    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found."); return
    
    onnx_path = input_path.with_suffix(".onnx")
    engine_path = Path(args.output) if args.output else input_path.with_suffix(".engine")
    
    try:
        # Step 1: Model -> ONNX
        if input_path.suffix == '.h5':
            convert_h5_to_onnx(str(input_path), str(onnx_path))
        elif input_path.suffix in ['.pt', '.pth']:
            convert_pt_to_onnx(str(input_path), str(onnx_path), imgsz=args.imgsz)
        else:
            print(f"Unsupported file type: {input_path.suffix}"); return
        
        # Step 2: ONNX -> Engine
        convert_onnx_to_engine(str(onnx_path), str(engine_path), fp16=args.fp16)
        
    except ImportError as e:
        print(f"\nError: Missing dependency {e.name}. Please install it.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
