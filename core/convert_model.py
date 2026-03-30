import os
import subprocess
import argparse
from pathlib import Path

def convert_h5_to_onnx(h5_path, onnx_path):
    """
    Converts .h5 (Keras) to .onnx using tf2onnx.
    Note: Requires tensorflow and tf2onnx installed.
    """
    print(f"Converting {h5_path} to {onnx_path}...")
    import tensorflow as tf
    import tf2onnx

    model = tf.keras.models.load_model(h5_path)
    
    # Define input signature
    spec = (tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="input"),)
    
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=onnx_path)
    print(f"Successfully saved ONNX model to {onnx_path}")

def convert_onnx_to_engine(onnx_path, engine_path, fp16=True):
    """
    Converts .onnx to .engine using trtexec.
    This is the most reliable way on Jetson Nano.
    """
    print(f"Converting {onnx_path} to {engine_path} (FP16={fp16})...")
    
    trtexec_path = "/usr/src/tensorrt/bin/trtexec"
    if not os.path.exists(trtexec_path):
        trtexec_path = "trtexec" # Try if in PATH
        
    cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--workspace=1024" # Allocate 1GB for conversion
    ]
    
    if fp16:
        cmd.append("--fp16")
        
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Successfully created TensorRT engine at {engine_path}")
    else:
        print("Conversion failed!")
        print(result.stdout)
        print(result.stderr)

def main():
    parser = argparse.ArgumentParser(description="Convert .h5 model to TensorRT .engine")
    parser.add_argument("input", type=str, help="Path to input .h5 file")
    parser.add_argument("--output", type=str, default=None, help="Path to output .engine file")
    parser.add_argument("--no-fp16", action="store_false", dest="fp16", help="Disable FP16 precision")
    parser.set_defaults(fp16=True)
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return
    
    onnx_path = input_path.with_suffix(".onnx")
    engine_path = Path(args.output) if args.output else input_path.with_suffix(".engine")
    
    try:
        # Step 1: H5 -> ONNX
        convert_h5_to_onnx(str(input_path), str(onnx_path))
        
        # Step 2: ONNX -> Engine
        convert_onnx_to_engine(str(onnx_path), str(engine_path), fp16=args.fp16)
        
    except ImportError:
        print("\nError: Missing dependencies.")
        print("Please install them inside the docker container or on your host:")
        print("pip install tensorflow tf2onnx")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
