# Use NVIDIA L4T base image for Jetson Nano
FROM nvcr.io/nvidia/l4t-base:r32.7.1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-libnvinfer \
    python3-libnvinfer-dev \
    libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python packages
# Note: On Jetson Nano (Python 3.6), specific versions maintain stability
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install \
    numpy==1.19.5 \
    opencv-python==4.1.1.26 \
    pycuda \
    pyyaml

# Copy project files
COPY . .

CMD ["python3"]