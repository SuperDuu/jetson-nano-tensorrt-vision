FROM ubuntu:18.04
RUN apt-get update && apt-get install -y \
    python3.6 \
    python3.6-dev \
    python3-pip \
    libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN python3.6 -m pip install --upgrade pip
RUN python3.6 -m pip install numpy==1.19.5 opencv-python==4.1.1.26

WORKDIR /app
CMD ["python3.6"]