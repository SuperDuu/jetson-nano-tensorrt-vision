#!/bin/bash

echo "--- Fixing UART Dependencies for Jetson Nano ---"

# 1. Check if running on Python 3.6+
PYTHON_BIN=$(which python3)
if [ -z "$PYTHON_BIN" ]; then
    echo "python3 not found. Please install it."
    exit 1
fi

echo "Using: $PYTHON_BIN"

# 2. Uninstall conflicting 'serial' package
echo "Cleaning up conflicting packages..."
$PYTHON_BIN -m pip uninstall -y serial
$PYTHON_BIN -m pip uninstall -y pyserial

# 3. Install correct 'pyserial' package
echo "Installing pyserial..."
$PYTHON_BIN -m pip install pyserial

# 4. Final verification
echo "Verifying installation..."
$PYTHON_BIN -c "import serial; print('SUCCESS: serial.Serial is available') if hasattr(serial, 'Serial') else exit(1)"

if [ $? -eq 0 ]; then
    echo "--- Fix Applied Successfully! ---"
else
    echo "--- Fix Failed. Please check the logs above. ---"
    exit 1
fi
