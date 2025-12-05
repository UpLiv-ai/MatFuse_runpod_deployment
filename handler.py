import sys
import os
import torch
import numpy as np
import runpod
import base64
from io import BytesIO
from PIL import Image
from omegaconf import OmegaConf

# --- 1. Path Setup ---
# Current directory (where handler.py is)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# MatFuse folder next to handler
MATFUSE_DIR = os.path.join(ROOT_DIR, "matfuse-sd")
# Source folder inside MatFuse
SRC_DIR = os.path.join(MATFUSE_DIR, "src")

# Critical: Add these to sys.path so Python finds the code
sys.path.append(MATFUSE_DIR)
sys.path.append(SRC_DIR)

# --- 2. Imports ---
# These must happen AFTER sys.path modification
try:
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.ddim import DDIMSampler
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    print(f"Checking paths... ROOT: {ROOT_DIR}, SRC: {SRC_DIR}")
    # Continue to allow container to start and log error
    pass

# --- 3. Global Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MATFUSE_MODEL = None
UPSCALE_MODEL = None

# Paths to config and weights
# We assume you download weights to a 'checkpoints' folder in the ROOT
CONFIG_PATH = os.path.join(SRC_DIR, "configs/diffusion/matfuse_ldm_vq_f8.yaml")
CKPT_PATH = os.path.join(ROOT_DIR, "checkpoints/matfuse_f8.ckpt")
UPSCALER_PATH = os.path.join(ROOT_DIR, "checkpoints/RealESRGAN_x4plus.pth")

def load_models():
    global MATFUSE_MODEL, UPSCALE_MODEL
    
    # A. Load MatFuse
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"MatFuse checkpoint not found at {CKPT_PATH}")
        
    print("Loading MatFuse...")
    config = OmegaConf.load(CONFIG_PATH)
    # Instantiate model from config
    model = instantiate_from_config(config.model)
    
    # Load weights
    pl_sd = torch.load(CKPT_PATH, map_location="cpu")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    model.load_state_dict(sd, strict=False)
    
    model.to(DEVICE)
    model.eval()
    MATFUSE_MODEL = model
    
    # B. Load Upscaler (Real-ESRGAN)
    if not os.path.exists(UPSCALER_PATH):
        print(f"Upscaler not found at {UPSCALER_PATH}. Attempting download...")
        import urllib.request
        os.makedirs(os.path.dirname(UPSCALER_PATH), exist_ok=True)
        urllib.request.urlretrieve(
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            UPSCALER_PATH
        )

    print("Loading Real-ESRGAN...")
    model_rrdb = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    UPSCALE_MODEL = RealESRGANer(
        scale=4,
        model_path=UPSCALER_PATH,
        model=model_rrdb,
        tile=400, # Tile size 400 to prevent OOM on smaller GPUs
        tile_pad=10,
        pre_pad=0,
        half=True, 
        device=DEVICE
    )

def normalize_normal_map(img_np):
    """
    Renormalize Normal Map vectors to unit length after upscaling.
    Input: Numpy array (H, W, 3) in uint8 
    Output: Numpy array (H, W, 3) in uint8 
    """
    # Convert to float [-1, 1]
    float_map = (img_np.astype(np.float32) / 255.0) * 2.0 - 1.0
    
    # Compute magnitude
    magnitude = np.linalg.norm(float_map, axis=2, keepdims=True)
    
    # Normalize (avoid div by zero)
    normalized = float_map / (magnitude + 1e-9)
    
    # Convert back to 
    res = (normalized * 0.5 + 0.5) * 255.0
    return np.clip(res, 0, 255).astype(np.uint8)

def handler(job):
    global MATFUSE_MODEL, UPSCALE_MODEL
    
    # Warm start loading
    if MATFUSE_MODEL is None:
        load_models()
        
    job_input = job.get("input", {})
    
    # Parameters
    prompt = job_input.get("prompt", "concrete texture")
    neg_prompt = job_input.get("negative_prompt", "")
    steps = job_input.get("steps", 30)
    cfg = job_input.get("guidance_scale", 7.5)
    seed = job_input.get("seed", -1)
    target_res = job_input.get("resolution", 2048)
    
    if seed == -1: seed = torch.randint(0, 1000000, (1,)).item()
    torch.manual_seed(seed)
    
    # 1. Generate Base Maps (512x512)
    # MatFuse Inference Logic
    sampler = DDIMSampler(MATFUSE_MODEL)
    c = MATFUSE_MODEL.get_learned_conditioning([prompt])
    uc = None
    if cfg!= 1.0:
        uc = MATFUSE_MODEL.get_learned_conditioning([neg_prompt])
        
    shape = [4, 512 // 8, 512 // 8]
    
    with torch.inference_mode(), torch.autocast(DEVICE):
        samples, _ = sampler.sample(S=steps, conditioning=c, batch_size=1, shape=shape, 
                                    verbose=False, unconditional_guidance_scale=cfg, 
                                    unconditional_conditioning=uc, eta=0.0)
        
        x_samples = MATFUSE_MODEL.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        
        # Convert to grid (H, W, C)
        grid = x_samples.cpu().numpy().transpose(1, 2, 0)
        grid = (grid * 255).astype(np.uint8)
        
    # 2. Split Maps (Assumes standard MatFuse horizontal stacking: Diffuse, Normal, Rough, Spec)
    h, w, _ = grid.shape
    single_w = w // 4
    
    maps = {
        "diffuse": grid[:, 0:single_w, :],
        "normal": grid[:, single_w:single_w*2, :],
        "roughness": grid[:, single_w*2:single_w*3, :],
        "specular": grid[:, single_w*3:, :]
    }
    
    # 3. Upscale
    results_b64 = {}
    
    # Real-ESRGAN x4 scales 512 -> 2048
    # If user wants 1024, we upscale to 2048 then downscale
    
    for k, img in maps.items():
        # Upscale
        output, _ = UPSCALE_MODEL.enhance(img, outscale=4)
        
        # Renormalize Normal Map
        if k == "normal":
            output = normalize_normal_map(output)
            
        # Resize if target is 1024
        if target_res!= 2048:
            pil_img = Image.fromarray(output)
            pil_img = pil_img.resize((target_res, target_res), Image.LANCZOS)
            final_img = pil_img
        else:
            final_img = Image.fromarray(output)
            
        # Encode to Base64
        buff = BytesIO()
        final_img.save(buff, format="PNG")
        results_b64[k] = base64.b64encode(buff.getvalue()).decode("utf-8")
        
    return {"images": results_b64, "seed": seed}

if __name__ == "__main__":
    # Local Test
    print("Running Local Test...")
    load_models()
    res = handler({"input": {"prompt": "old rusty metal", "resolution": 1024, "steps": 20}})
    print("Done. Saving local_test_diffuse.png")
    with open("local_test_diffuse.png", "wb") as f:
        f.write(base64.b64decode(res["images"]["diffuse"]))
else:
    runpod.serverless.start({"handler": handler})