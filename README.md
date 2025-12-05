# MatFuse_runpod_deployment


## Testing on Pod

### Installation commands

# 1. Update System
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# 2 clone source repo
git clone --recursive https://github.com/UpLiv-ai/MatFuse_runpod_deployment.git

# 2.5 Go to your repo folder (Assuming you uploaded it to /workspace/my-repo)
cd /workspace/MatFuse_runpod_deployment

# 3. UNINSTALL the newer Torch that comes with the pod (Important!)
pip uninstall -y torch torchvision torchaudio

# 4. INSTALL the older Torch required by MatFuse (1.13.1 + CUDA 11.7)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# 5. Install MatFuse dependencies
# Note: We must be careful not to upgrade torch back to 2.0
cd matfuse-sd
pip install -r requirements.txt

# 6. Install Serving & Upscaling tools
pip install runpod realesrgan basicsr

# 7. Setup Checkpoints folder
cd..
mkdir -p checkpoints

# 8. Download Weights (Manually for testing)
# MatFuse
wget -O checkpoints/matfuse_f8.ckpt "https://huggingface.co/gvecchio/MatFuse/resolve/main/matfuse_f8.ckpt?download=true"
# Real-ESRGAN
wget -O checkpoints/RealESRGAN_x4plus.pth "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

# 9. Run the handler test
python handler.py