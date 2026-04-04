import Jetson.GPIO as GPIO
import subprocess
import time
import os
import sys

# Configuration
INPUT_PIN = 16  # Pin 5 (Board numbering)
# Use the actual absolute path for user 'du'
PROJECT_ROOT = "/home/pi/Desktop/jetson-nano-tensorrt-vision"
SCRIPT_PATH = os.path.join(PROJECT_ROOT, "src/system_manager_v2.py")
PYTHON_EXEC = "/usr/bin/python3" # Explicitly use system python3

def main():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(INPUT_PIN, GPIO.IN)

    print(f"--- Boot check on Pin {INPUT_PIN} ---")
    
    # Allow 2 seconds for voltage stabilization
    time.sleep(2)

    # Read state: LOW (0) is CLOSED switch (Active), HIGH (1) is OPEN switch
    input_state = GPIO.input(INPUT_PIN)
    
    if input_state == GPIO.LOW:
        print("Switch ON (LOW): Starting Vision System V2...")
        # Start the vision manager in the background or foreground as requested.
        # systemd will manage this script, so we can run the sub-process.
        try:
            subprocess.run([PYTHON_EXEC, SCRIPT_PATH], cwd=PROJECT_ROOT, check=True)
        except KeyboardInterrupt:
            print("System manually stopped.")
        except Exception as e:
            print(f"Error starting vision system: {e}")
    else:
        print("Switch OFF (HIGH): Vision System bypass.")

    GPIO.cleanup()

if __name__ == "__main__":
    main()
