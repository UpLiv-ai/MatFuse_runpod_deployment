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
CONFIG_PATH = os.path.join(SRC_DIR, "configs/diffusion/matfuse-ldm-vq_f8.yaml")
# We stick with v0.1.0 (x4plus) because v0.3.0 (general-x4v3) is a tiny model for video, 
# which lacks the fine texture detail generation needed for PBR.
CKPT_PATH = os.path.join(ROOT_DIR, "matfuse-sd", "checkpoints/matfuse_f8.ckpt")
UPSCALER_PATH = os.path.join(ROOT_DIR, "matfuse-sd", "checkpoints/RealESRGAN_x4plus.pth")

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
    conditioning_inputs = {}

    # 1. Process Image Inputs (Style, Structure, and Palette)
    if input_image_b64:
        print("Processing Image Input...")
        cond_img = process_input_image(input_image_b64)
        if cond_img:
            # A. Image Embed (Style) -> RGB Tensor [1, 3, 512, 512]
            img_tensor = torch.from_numpy(np.array(cond_img)).float() / 127.5 - 1.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            conditioning_inputs["image_embed"] = img_tensor
            
            # B. Sketch (Structure) -> Grayscale Tensor [1, 1, 512, 512]
            # Take mean across channels to convert RGB to Grayscale
            sketch_tensor = img_tensor.mean(dim=1, keepdim=True)
            conditioning_inputs["sketch"] = sketch_tensor

            # C. Palette (Color) -> Color Vector [1, 5, 3]
            # Resize image to 5x1 to extract 5 dominant colors
            palette_img = cond_img.resize((5, 1), Image.Resampling.BICUBIC)
            palette_arr = np.array(palette_img) # Shape becomes (1, 5, 3)
            # Normalize to [-1, 1] range
            palette_tensor = torch.from_numpy(palette_arr).float() / 127.5 - 1.0
            conditioning_inputs["palette"] = palette_tensor.to(DEVICE)

    else:
        print("Warning: No image provided. Using blank defaults.")
        # Create blank tensors to prevent crash
        conditioning_inputs["image_embed"] = torch.zeros((1, 3, 512, 512)).to(DEVICE)
        conditioning_inputs["sketch"] = torch.zeros((1, 1, 512, 512)).to(DEVICE)
        conditioning_inputs["palette"] = torch.zeros((1, 5, 3)).to(DEVICE)

    # 2. Process Text
    # We pass the prompt under the "text" key (standard for these models)
    conditioning_inputs["text"] = [prompt] if prompt else [""]

    # 3. Get Combined Conditioning (Call ONLY once)
    c = MATFUSE_MODEL.get_learned_conditioning(conditioning_inputs)

    # 4. Unconditional Conditioning (for Classifier-Free Guidance)
    uc = None
    if cfg != 1.0:
        # For UCF, we usually keep the image conditioning but empty the text
        uc_inputs = conditioning_inputs.copy()
        uc_inputs["text"] = [""] * len(conditioning_inputs["text"])
        uc = MATFUSE_MODEL.get_learned_conditioning(uc_inputs)
        
    # --- 5. Sampling ---
    shape = [12, 512 // 8, 512 // 8]
    sampler = DDIMSampler(MATFUSE_MODEL)
    
    print(f"Generating with Seed: {seed}")
    
    # Step A: Run Sampling in FP16 (Fast & Efficient)
    with torch.inference_mode(), torch.autocast(DEVICE):
        samples, _ = sampler.sample(S=steps, conditioning=c, batch_size=1, shape=shape, 
                                    verbose=False, unconditional_guidance_scale=cfg, 
                                    unconditional_conditioning=uc, eta=0.0)

    # Step B: Run Decoding in FP32 (Precise & Stable)
    # FIX: We exit the 'autocast' block above. The Decoder fails in FP16 (produces black images).
    with torch.inference_mode():
        print("Decoding in Float32...")
        
        # 1. Ensure samples are Float32
        samples = samples.to(dtype=torch.float32)
        
        # 2. Decode using the built-in method (handles scaling automatically)
        x_samples = MATFUSE_MODEL.decode_first_stage(samples)
        
        # 3. Clamp and Normalize
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        
        # 4. Convert to Numpy -> (Height, Width, 9)
        grid = x_samples.cpu().numpy().transpose(0, 2, 3, 1)[0]
        
        # 5. Slice the 9-channel image into 3 separate RGB images
        diffuse_np = (grid[:, :, 0:3] * 255).astype(np.uint8)
        normal_np  = (grid[:, :, 3:6] * 255).astype(np.uint8)
        packed_np  = (grid[:, :, 6:9] * 255).astype(np.uint8)

    # --- 7. Processing Maps ---
    # Extract Roughness and Specular from the packed image
    roughness_np = packed_np[:, :, 0]
    specular_np  = packed_np[:, :, 1]
    
    # Stack them back into 3-channel images
    roughness_np = np.stack([roughness_np]*3, axis=-1)
    specular_np  = np.stack([specular_np]*3, axis=-1)

    maps = {
        "diffuse": diffuse_np,
        "normal": normal_np,
        "roughness": roughness_np,
        "specular": specular_np
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