# Use an NVIDIA CUDA base image with PyTorch pre-installed
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set environment variables to prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set the working directory to the repo root
WORKDIR /workspace/matfuse-sd

# Install system dependencies required for OpenCV and Git
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy the repository code into the container
COPY..

# Install Python dependencies
# We install RunPod and Upscaling libraries in addition to MatFuse requirements
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir runpod basicsr realesrgan

# Download Real-ESRGAN weights at build time
# This reduces cold-start latency by moving the download to the build phase
RUN mkdir -p checkpoints && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O checkpoints/RealESRGAN_x4plus.pth

# Set the container entrypoint to execute the handler
# The -u flag ensures unbuffered output for real-time logging
CMD [ "python", "-u", "handler.py" ]