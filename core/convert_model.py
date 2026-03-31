import os
import sys
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

def convert_pt_to_onnx(pt_path, onnx_path, imgsz=512, fp16=True):
    """Converts .pt (PyTorch) to .onnx."""
    import torch
    # half=True requires CUDA for export in many versions
    use_half = fp16 and torch.cuda.is_available()
    if fp16 and not torch.cuda.is_available():
        print("WARNING: CUDA not available. Exporting ONNX in FP32 instead of FP16.")

    print(f"Converting {pt_path} to {onnx_path} (FP16={use_half}, imgsz={imgsz})...")
    
    # Delete stale ONNX to force fresh export
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
        print(f"  Removed old ONNX: {onnx_path}")
    
    try:
        # Try Ultralytics first (for YOLOv8/v11)
        from ultralytics import YOLO
        model = YOLO(pt_path)
        model.export(
            format='onnx', 
            imgsz=imgsz, 
            dynamic=False, 
            half=use_half,
            simplify=True,
            opset=11,  # Opset 11 is safer for TensorRT 8.2 on Jetson Nano
            nms=False # Handle NMS in inference code for better performance
        )
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
        torch.onnx.export(model, dummy_input, onnx_path, opset_version=11)
    
    print(f"Successfully saved ONNX model to {onnx_path}")

def patch_mod_nodes(onnx_path):
    """
    Replace unsupported 'Mod' nodes with TensorRT-compatible equivalent:
    Mod(x, y) = Sub(x, Mul(Floor(Div(x, y)), y))
    This is required for TensorRT 8.2 which does not support the Mod operator.
    """
    # Auto-install onnx if not available
    try:
        import onnx
    except ImportError:
        print("  'onnx' package not found, installing...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'onnx'])
        import onnx
    
    from onnx import helper
    model = onnx.load(onnx_path)
    graph = model.graph
    nodes_to_remove = []
    nodes_to_add = []
    
    for node in graph.node:
        if node.op_type == 'Mod':
            print(f"  Patching unsupported Mod node: {node.name}")
            x_input = node.input[0]
            y_input = node.input[1]
            output = node.output[0]
            prefix = node.name or output
            
            div_out = prefix + "_div"
            floor_out = prefix + "_floor"
            mul_out = prefix + "_mul"
            
            div_node = helper.make_node('Div', [x_input, y_input], [div_out], name=prefix + '_Div')
            floor_node = helper.make_node('Floor', [div_out], [floor_out], name=prefix + '_Floor')
            mul_node = helper.make_node('Mul', [floor_out, y_input], [mul_out], name=prefix + '_Mul')
            sub_node = helper.make_node('Sub', [x_input, mul_out], [output], name=prefix + '_Sub')
            
            nodes_to_remove.append(node)
            nodes_to_add.extend([div_node, floor_node, mul_node, sub_node])
    
    if nodes_to_remove:
        for n in nodes_to_remove:
            graph.node.remove(n)
        graph.node.extend(nodes_to_add)
        onnx.save(model, onnx_path)
        print(f"  Patched {len(nodes_to_remove)} Mod node(s) in {onnx_path}")
    else:
        print(f"  No Mod nodes found, ONNX is clean.")

def convert_onnx_to_engine(onnx_path, engine_path, fp16=True):
    """Converts .onnx to .engine using trtexec."""
    print(f"Converting {onnx_path} to {engine_path} (FP16={fp16})...")
    
    # Patch unsupported Mod nodes before conversion
    patch_mod_nodes(onnx_path)
    
    trtexec_path = "/usr/src/tensorrt/bin/trtexec"
    if not os.path.exists(trtexec_path): trtexec_path = "trtexec"
    
    # Jetson Nano (JetPack 4.6) uses TRT 8.2 which supports --workspace only
    cmd_run = cmd + ["--workspace=2048"]
    
    print(f"Running command: {' '.join(cmd_run)}")
    result = subprocess.run(cmd_run, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    if result.returncode == 0:
        print(f"Successfully created TensorRT engine at {engine_path}")
    else:
        print(f"Conversion failed!\n{result.stdout}\n{result.stderr}")
        raise RuntimeError("trtexec failed to generate engine")

def main():
    parser = argparse.ArgumentParser(description="Convert .h5 or .pt model to TensorRT .engine")
    parser.add_argument("input", type=str, help="Path to input (.h5 or .pt) file")
    parser.add_argument("--output", type=str, default=None, help="Path to output .engine file")
    parser.add_argument("--imgsz", type=int, default=512, help="Input image size (default 512)")
    parser.add_argument("--no-fp16", action="store_false", dest="fp16", help="Disable FP16 precision")
    parser.set_defaults(fp16=True)
    
    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found."); return
    
    onnx_path = input_path.with_suffix(".onnx")
    engine_path = Path(args.output) if args.output else input_path.with_suffix(".engine")
    
    try:
        # Step 1: Model -> ONNX (skip if input is already ONNX)
        if input_path.suffix == '.onnx':
            print(f"Input is already ONNX, skipping step 1.")
            onnx_path = input_path
        elif input_path.suffix == '.h5':
            convert_h5_to_onnx(str(input_path), str(onnx_path))
        elif input_path.suffix in ['.pt', '.pth']:
            convert_pt_to_onnx(str(input_path), str(onnx_path), imgsz=args.imgsz, fp16=args.fp16)
        else:
            print(f"Unsupported file type: {input_path.suffix}"); return
        
        # Step 2: ONNX -> Engine
        convert_onnx_to_engine(str(onnx_path), str(engine_path), fp16=args.fp16)
        
    except ImportError as e:
        print(f"\nError: Missing dependency {e.name}. Please install it.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
