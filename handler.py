import sys
import os
import torch
import numpy as np
import runpod
import base64
import cv2
import urllib.request
from io import BytesIO
from PIL import Image, ImageOps

# --- 1. Path & Import Setup ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MATFUSE_DIR = os.path.join(ROOT_DIR, "matfuse-sd")
SRC_DIR = os.path.join(MATFUSE_DIR, "src")
sys.path.append(MATFUSE_DIR)
sys.path.append(SRC_DIR)

# --- ARGUMENT HACK START ---
original_argv = sys.argv.copy()
config_path = os.path.join(SRC_DIR, "configs/diffusion/matfuse-ldm-vq_f8.yaml")
ckpt_path = os.path.join(ROOT_DIR, "checkpoints/matfuse_f8.ckpt")
sys.argv = ["inference_helpers.py", "--config", config_path, "--ckpt", ckpt_path]

try:
    from inference_helpers import run_generation
    print("✅ MatFuse Model loaded via inference_helpers.")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

sys.argv = original_argv
# --- ARGUMENT HACK END ---

# --- 2. Upscaler Setup ---
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

UPSCALER_PATH = os.path.join(ROOT_DIR, "checkpoints/RealESRGAN_x4plus.pth")
UPSCALE_MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_upscaler():
    global UPSCALE_MODEL
    if UPSCALE_MODEL is not None: return

    if not os.path.exists(UPSCALER_PATH):
        print(f"Upscaler not found. Downloading...")
        os.makedirs(os.path.dirname(UPSCALER_PATH), exist_ok=True)
        urllib.request.urlretrieve(
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            UPSCALER_PATH
        )

    print("Loading Real-ESRGAN...")
    model_rrdb = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    UPSCALE_MODEL = RealESRGANer(scale=4, model_path=UPSCALER_PATH, model=model_rrdb, tile=400, tile_pad=10, pre_pad=0, half=True, device=DEVICE)

# --- 3. Helper Functions ---
def process_input_image(b64_string):
    try:
        if "," in b64_string: b64_string = b64_string.split(",")[1]
        image_data = base64.b64decode(b64_string)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        return ImageOps.fit(image, (512, 512), method=Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"Error processing input image: {e}")
        return None

def normalize_normal_map(img_np):
    float_map = (img_np.astype(np.float32) / 255.0) * 2.0 - 1.0
    magnitude = np.linalg.norm(float_map, axis=2, keepdims=True)
    normalized = float_map / (magnitude + 1e-9)
    res = (normalized * 0.5 + 0.5) * 255.0
    return np.clip(res, 0, 255).astype(np.uint8)

# --- 4. Main Handler ---
def handler(job):
    load_upscaler()
    
    job_input = job.get("input", {})
    prompt = job_input.get("prompt", "") # Prompt is optional too, though usually required for logic
    input_image_b64 = job_input.get("image", None)
    target_res = job_input.get("resolution", 2048)
    steps = job_input.get("steps", 20)
    seed = job_input.get("seed", -1)
    if seed == -1: seed = int(torch.randint(0, 1000000, (1,)).item())
    
    print(f"--- Processing Job [Seed: {seed}] ---")

    # 1. Prepare Inputs (Flexible Mode)
    # If no image is provided, we pass 'None'. 
    # inference_helpers.py will automatically convert 'None' into Zero Tensors.
    
    pil_image = None
    edges = None
    
    if input_image_b64:
        pil_image = process_input_image(input_image_b64)
        if pil_image:
            # We have an image, so we generate the Canny Sketch
            img_np = np.array(pil_image)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
    else:
        print("No image provided. Running in Text-Only mode (expect lower structural coherence).")

    # 2. Run Generation
    # We pass 'None' for inputs we don't have. The repo code handles the zeros.
    results = run_generation(
        render_emb=pil_image,     # None if no image -> Zero Tensor
        palette_source=pil_image, # None if no image -> Zero Tensor
        sketch=edges,             # None if no image -> Zero Tensor
        prompt=prompt,
        num_samples=1,
        image_resolution=512,
        ddim_steps=steps,
        seed=seed,
        ddim_eta=0.0
    )

    # 3. Process Outputs
    if len(results) < 3:
        return {"error": "Generation failed to produce outputs."}
    
    output_grid = results[2]
    
    # Save Raw Grid for Debugging
    results_b64 = {}
    raw_pil = Image.fromarray(output_grid)
    raw_buff = BytesIO()
    raw_pil.save(raw_buff, format="PNG")
    results_b64["raw_grid"] = base64.b64encode(raw_buff.getvalue()).decode("utf-8")
    
    # Slice Grid
    h, w, c = output_grid.shape
    half_h, half_w = h // 2, w // 2
    
    raw_maps = {
        "diffuse": output_grid[0:half_h, 0:half_w, :],
        "normal": output_grid[half_h:h, 0:half_w, :],
        "roughness": output_grid[0:half_h, half_w:w, :],
        "specular": output_grid[half_h:h, half_w:w, :]
    }
    
    # 4. Upscale & Encode
    for k, img in raw_maps.items():
        output, _ = UPSCALE_MODEL.enhance(img, outscale=4)
        if k == "normal": output = normalize_normal_map(output)
            
        pil_img = Image.fromarray(output)
        if target_res != 2048:
            pil_img = pil_img.resize((target_res, target_res), Image.LANCZOS)
            
        buff = BytesIO()
        pil_img.save(buff, format="PNG")
        results_b64[k] = base64.b64encode(buff.getvalue()).decode("utf-8")

    return {"images": results_b64, "seed": seed}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="")
    args = parser.parse_args()

    print("--- Running Local Test ---")
    
    job_payload = {
        "input": {
            "prompt": args.prompt,
            "resolution": 1024,
            "steps": 20
        }
    }

    if args.image and os.path.exists(args.image):
        print(f"Loading local image: {args.image}")
        with open(args.image, "rb") as img_f:
            b64_str = base64.b64encode(img_f.read()).decode("utf-8")
            job_payload["input"]["image"] = b64_str

    try:
        res = handler(job_payload)
        out_dir = "test_output"
        os.makedirs(out_dir, exist_ok=True)
        print(f"Success! Saving outputs to ./{out_dir}")
        for k, v in res["images"].items():
            with open(f"{out_dir}/{k}.png", "wb") as f:
                f.write(base64.b64decode(v))
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()

else:
    runpod.serverless.start({"handler": handler})