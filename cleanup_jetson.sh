#!/bin/bash

echo "--- Jetson Nano Camera & Process Cleanup ---"

# 1. Check for running python scripts using /dev/video0
echo "Checking for processes using /dev/video0..."
sudo fuser -v /dev/video0 2>/dev/null

# 2. Key process cleanup
echo "Terminating conflicting vision processes..."
# Kill any existing system_manager instances
sudo pkill -f system_manager_v2.py
sudo pkill -f system_manager.py
# Kill any gstreamer zombie processes
sudo pkill -9 gst-launch-1.0
sudo pkill -9 nvgstplayer

# 3. Fix UART Permissions
echo "Fixing serial port permissions..."
sudo chmod 666 /dev/ttyACM0 2>/dev/null
sudo chmod 666 /dev/ttyUSB0 2>/dev/null
sudo usermod -aG dialout $USER

# 4. Restart nvargus-daemon (often fixes CSI and USB sync issues on Jetson)
echo "Restarting nvargus-daemon..."
sudo systemctl restart nvargus-daemon

# 5. Check camera device availability
if [ -e /dev/video0 ]; then
    echo "SUCCESS: /dev/video0 is present."
    ls -l /dev/video0
else
    echo "ERROR: /dev/video0 is NOT found. Check physical connection."
fi

# 6. Check UART device availability
if [ -e /dev/ttyACM0 ]; then
    echo "SUCCESS: /dev/ttyACM0 is present."
    ls -l /dev/ttyACM0
elif [ -e /dev/ttyUSB0 ]; then
    echo "SUCCESS: /dev/ttyUSB0 is present."
    ls -l /dev/ttyUSB0
else
    echo "WARNING: No UART device found (/dev/ttyACM0 or /dev/ttyUSB0)."
fi

echo "--- Cleanup Complete. Try running the vision system again. ---"
