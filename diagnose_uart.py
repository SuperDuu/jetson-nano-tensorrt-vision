import os
import sys
import logging

def diagnose():
    print("--- UART Diagnostic Tool ---")
    print(f"Python version: {sys.version}")
    
    try:
        import serial
        print(f"Module 'serial' successfully imported.")
        print(f"File path: {getattr(serial, '__file__', 'Unknown')}")
        
        has_serial = hasattr(serial, 'Serial')
        print(f"Has 'Serial' attribute: {has_serial}")
        
        if not has_serial:
            print("\n[!] ERROR: The 'serial' module does not have the 'Serial' attribute.")
            print("This usually means you have the 'serial' (serialization) package installed instead of 'pyserial'.")
            print("Or you have a file named 'serial.py' in your current directory or PYTHONPATH.")
            
            # Check for shadowing files
            cwd = os.getcwd()
            serial_py = os.path.join(cwd, 'serial.py')
            serial_dir = os.path.join(cwd, 'serial')
            
            if os.path.exists(serial_py):
                print(f"[!] Found shadowing file: {serial_py}")
            if os.path.exists(serial_dir):
                print(f"[!] Found shadowing directory: {serial_dir}")
                
    except ImportError:
        print("[!] ERROR: Module 'serial' not found. Please install 'pyserial'.")
    except Exception as e:
        print(f"[!] Unexpected error during diagnostic: {e}")

if __name__ == "__main__":
    diagnose()
