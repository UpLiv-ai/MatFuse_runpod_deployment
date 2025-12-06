import sys
import os
import torch
import numpy as np
import cv2
from PIL import Image

# --- 1. Setup ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MATFUSE_DIR = os.path.join(ROOT_DIR, "matfuse-sd")
SRC_DIR = os.path.join(MATFUSE_DIR, "src")
sys.path.append(MATFUSE_DIR)
sys.path.append(SRC_DIR)

# Mock args for inference_helpers
config_path = os.path.join(SRC_DIR, "configs/diffusion/matfuse-ldm-vq_f8.yaml")
ckpt_path = os.path.join(ROOT_DIR, "checkpoints/matfuse_f8.ckpt")
sys.argv = ["inference_helpers.py", "--config", config_path, "--ckpt", ckpt_path]

# --- 2. Import Repo Code ---
try:
    from inference_helpers import run_generation
    print("✅ Model loaded.")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# --- 3. The Test ---
def run_brick_test():
    image_path = "./test_image_6.png" # Your brick image
    prompt = "A material of stone bricks with green moss"
    
    if not os.path.exists(image_path):
        print(f"❌ Could not find {image_path}. Please upload it.")
        return

    print(f"--- Processing {image_path} ---")
    
    # 1. Load and Prepare Sketch (Canny)
    pil_image = Image.open(image_path).convert("RGB")
    img_np = np.array(pil_image)
    
    # Generate Canny Edges (The "Coloring Book Lines")
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    
    # Save the sketch so you can verify it exists!
    cv2.imwrite("sanity_debug_sketch.png", edges)
    print("Saved debug sketch to sanity_debug_sketch.png")
    
    # 2. Run Generation
    # We pass the edges as the 'sketch' input
    results = run_generation(
        render_emb=pil_image,     # Style reference
        palette_source=pil_image, # Color reference
        sketch=edges,             # Structure reference (CRITICAL)
        prompt=prompt,
        num_samples=1,
        image_resolution=512,
        ddim_steps=20,
        seed=42,
        ddim_eta=0.0
    )

    # 3. Save Output
    # results format: [sketch, palette, map_grid]
    if len(results) > 2:
        output_grid = results[2]
        Image.fromarray(output_grid).save("sanity_brick_result.png")
        print(f"✅ Success! Saved result to sanity_brick_result.png")
    else:
        print("❌ Generation failed.")

if __name__ == "__main__":
    run_brick_test()