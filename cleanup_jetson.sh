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

# 3. Restart nvargus-daemon (often fixes CSI and USB sync issues on Jetson)
echo "Restarting nvargus-daemon..."
sudo systemctl restart nvargus-daemon

# 4. Check camera device availability
if [ -e /dev/video0 ]; then
    echo "SUCCESS: /dev/video0 is present."
    ls -l /dev/video0
else
    echo "ERROR: /dev/video0 is NOT found. Check physical connection."
fi

echo "--- Cleanup Complete. Try running the vision system again. ---"
