#!/bin/bash
# Script to view the Robocon UDP Video Stream on Linux Laptop
# Optimized for low latency H.264 over UDP.

echo "--- Starting Robocon Video Stream Receiver (UDP 5000) ---"
echo "Press Ctrl+C to stop."

gst-launch-1.0 udpsrc port=5000 \
    ! application/x-rtp, encoding-name=H264, payload=96 \
    ! rtph264depay ! h264parse ! avdec_h264 \
    ! autovideosink sync=false
