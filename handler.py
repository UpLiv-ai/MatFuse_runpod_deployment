import sys
import os
import torch
import numpy as np
import runpod
import base64
from io import BytesIO
from PIL import Image, ImageOps
from omegaconf import OmegaConf

# --- 1. Path Setup ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MATFUSE_DIR = os.path.join(ROOT_DIR, "matfuse-sd")
SRC_DIR = os.path.join(MATFUSE_DIR, "src")

sys.path.append(MATFUSE_DIR)
sys.path.append(SRC_DIR)

# --- 2. Imports ---
try:
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.ddim import DDIMSampler
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    pass

# --- 3. Global Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MATFUSE_MODEL = None
UPSCALE_MODEL = None

# Paths
CONFIG_PATH = os.path.join(SRC_DIR, "configs/diffusion/matfuse_ldm_vq_f8.yaml")
# We stick with v0.1.0 (x4plus) because v0.3.0 (general-x4v3) is a tiny model for video, 
# which lacks the fine texture detail generation needed for PBR.
CKPT_PATH = os.path.join(ROOT_DIR, "checkpoints/matfuse_f8.ckpt")
UPSCALER_PATH = os.path.join(ROOT_DIR, "checkpoints/RealESRGAN_x4plus.pth")

def load_models():
    global MATFUSE_MODEL, UPSCALE_MODEL
    
    # A. Load MatFuse
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"MatFuse checkpoint not found at {CKPT_PATH}")
        
    print("Loading MatFuse...")
    config = OmegaConf.load(CONFIG_PATH)
    model = instantiate_from_config(config.model)
    pl_sd = torch.load(CKPT_PATH, map_location="cpu")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE)
    model.eval()
    MATFUSE_MODEL = model
    
    # B. Load Upscaler (Real-ESRGAN x4plus)
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
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=True, 
        device=DEVICE
    )

def normalize_normal_map(img_np):
    """Renormalize vectors to unit length."""
    float_map = (img_np.astype(np.float32) / 255.0) * 2.0 - 1.0
    magnitude = np.linalg.norm(float_map, axis=2, keepdims=True)
    normalized = float_map / (magnitude + 1e-9)
    res = (normalized * 0.5 + 0.5) * 255.0
    return np.clip(res, 0, 255).astype(np.uint8)

def process_input_image(b64_string):
    """Decodes base64 string to PIL Image."""
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
        image_data = base64.b64decode(b64_string)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        # Resize to 512x512 as MatFuse expects 512 inputs for conditioning
        return ImageOps.fit(image, (512, 512), method=Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"Error processing input image: {e}")
        return None

def handler(job):
    global MATFUSE_MODEL, UPSCALE_MODEL
    
    if MATFUSE_MODEL is None:
        load_models()
        
    job_input = job.get("input", {})
    
    # --- Input Parsing ---
    prompt = job_input.get("prompt", "")
    neg_prompt = job_input.get("negative_prompt", "")
    input_image_b64 = job_input.get("image", None) # Optional Base64 Image
    
    steps = job_input.get("steps", 30)
    cfg = job_input.get("guidance_scale", 7.5)
    seed = job_input.get("seed", -1)
    target_res = job_input.get("resolution", 2048)
    
    if seed == -1: seed = torch.randint(0, 1000000, (1,)).item()
    torch.manual_seed(seed)
    
    # --- Conditioning Logic ---
    # MatFuse handles multimodal conditioning.
    # 1. Text Conditioning
    c_list = []
    if prompt:
        c_list.append(MATFUSE_MODEL.get_learned_conditioning([prompt]))
    else:
        # Empty text conditioning if no prompt provided
        c_list.append(MATFUSE_MODEL.get_learned_conditioning([""]))

    # 2. Image Conditioning (if provided)
    # Note: MatFuse typically expects specific encoders for images.
    # Standard LDM uses CLIP Image Encoder for "style" or "variation".
    if input_image_b64:
        print("Processing Image Input...")
        cond_img = process_input_image(input_image_b64)
        if cond_img:
            # Preprocess image for the model (ToTensor + Normalize)
            # MatFuse uses a specific helper 'get_input' usually, but here we construct the tensor manually
            # to avoid dependency on dataset loaders.
            img_tensor = torch.from_numpy(np.array(cond_img)).float() / 127.5 - 1.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            
            # Encode image using the model's condition encoder
            # We assume the model has a method to encode images, typically 'get_learned_conditioning'
            # handles dictionary inputs in advanced LDMs, but for standard CLIP image conditioning:
            try:
                c_img = MATFUSE_MODEL.get_learned_conditioning({"image": img_tensor}) 
                # Note: If this fails, it might be because the specific MatFuse config expects 
                # a different key or direct tensor. Fallback to simple concatenation if supported.
                if isinstance(c_img, list): c_list.extend(c_img)
                else: c_list.append(c_img)
            except:
                print("Warning: Could not encode image conditioning. Ignoring image.")

    # Combine conditions (Simplified: usually just taking the first valid one or averaging)
    # For this handler, we default to the text conditioning primarily.
    c = c_list 

    uc = None
    if cfg!= 1.0:
        uc = MATFUSE_MODEL.get_learned_conditioning([neg_prompt])
        
    shape = [4, 512 // 8, 512 // 8]
    sampler = DDIMSampler(MATFUSE_MODEL)
    
    print(f"Generating with Seed: {seed}")
    
    with torch.inference_mode(), torch.autocast(DEVICE):
        samples, _ = sampler.sample(S=steps, conditioning=c, batch_size=1, shape=shape, 
                                    verbose=False, unconditional_guidance_scale=cfg, 
                                    unconditional_conditioning=uc, eta=0.0)
        
        x_samples = MATFUSE_MODEL.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        
        grid = x_samples.cpu().numpy().transpose(1, 2, 0)
        grid = (grid * 255).astype(np.uint8)
        
    # Split Maps
    h, w, _ = grid.shape
    single_w = w // 4
    maps = {
        "diffuse": grid[:, 0:single_w, :],
        "normal": grid[:, single_w:single_w*2, :],
        "roughness": grid[:, single_w*2:single_w*3, :],
        "specular": grid[:, single_w*3:, :]
    }
    
    # Upscale
    results_b64 = {}
    for k, img in maps.items():
        output, _ = UPSCALE_MODEL.enhance(img, outscale=4)
        if k == "normal":
            output = normalize_normal_map(output)
            
        pil_img = Image.fromarray(output)
        if target_res!= 2048:
            pil_img = pil_img.resize((target_res, target_res), Image.LANCZOS)
            
        buff = BytesIO()
        pil_img.save(buff, format="PNG")
        results_b64[k] = base64.b64encode(buff.getvalue()).decode("utf-8")
        
    return {"images": results_b64, "seed": seed}

# --- 4. Local Testing Logic ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Local Test for MatFuse Handler")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt")
    parser.add_argument("--image", type=str, default=None, help="Path to local image file")
    parser.add_argument("--res", type=int, default=1024, help="Target Resolution")
    args = parser.parse_args()

    print("--- Running Local Test ---")
    
    job_payload = {
        "input": {
            "prompt": args.prompt,
            "resolution": args.res,
            "steps": 20
        }
    }

    # Handle Local Image File -> Base64 conversion
    if args.image:
        if os.path.exists(args.image):
            print(f"Loading local image: {args.image}")
            with open(args.image, "rb") as img_f:
                b64_str = base64.b64encode(img_f.read()).decode("utf-8")
                job_payload["input"]["image"] = b64_str
        else:
            print(f"Warning: Image file {args.image} not found.")

    try:
        # Run handler
        res = handler(job_payload)
        
        # Save results
        out_dir = "test_output"
        os.makedirs(out_dir, exist_ok=True)
        print(f"Success! Saving outputs to./{out_dir}")
        
        for k, v in res["images"].items():
            with open(f"{out_dir}/{k}.png", "wb") as f:
                f.write(base64.b64decode(v))
                
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()

else:
    runpod.serverless.start({"handler": handler})