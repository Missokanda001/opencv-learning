import os
import pickle
import numpy as np
import cv2
from PIL import Image
import torch
from diffusers import DiffusionPipeline

# Import YOUR existing detector
from detectors import FaceDetector

# --------------------------
# Configuration
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENERATE_PER_CLASS = 40  # generate 40 images per expression; many will be filtered out
EXPRESSIONS = ["neutral", "happy", "sad", "surprised", "angry", "disgusted"]

# Output folders
RAW_IMG_DIR = "synthetic_raw"
ALIGNED_DIR = "synthetic_aligned_faces"
os.makedirs(RAW_IMG_DIR, exist_ok=True)
os.makedirs(ALIGNED_DIR, exist_ok=True)

# Your Haar cascade paths (same as main.py)
script_dir = os.path.dirname(os.path.abspath(__file__))
face_casc_path = os.path.join(script_dir, "params", "haarcascade_frontalface_default.xml")
left_eye_path = os.path.join(script_dir, "params", "haarcascade_lefteye_2splits.xml")
right_eye_path = os.path.join(script_dir, "params", "haarcascade_righteye_2splits.xml")

# --------------------------
# Load SD pipeline
# --------------------------
print(f"Loading Stable Diffusion on {DEVICE} ...")
pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    safety_checker=None
)
pipe = pipe.to(DEVICE)
if DEVICE == "cpu":
    pipe.enable_attention_slicing()

# --------------------------
# Load your face detector
# --------------------------
detector = FaceDetector(
    face_casc=face_casc_path,
    left_eye_casc=left_eye_path,
    right_eye_casc=right_eye_path,
    scale_factor=4
)

# --------------------------
# Prompt template: match your webcam data
# --------------------------
def get_prompt(expression: str):
    prompt = (
        "photorealistic frontal close-up of one person’s face, "
        f"{expression} facial expression, eyes open, "
        "even soft frontal lighting, plain light gray background, "
        "no accessories, no makeup, neutral camera angle, sharp eyes"
    )
    negative_prompt = (
        "profile, tilted head, multiple people, glasses, mask, "
        "dark shadows, strong color tint, blurry, cartoon, painting"
    )
    return prompt, negative_prompt

# --------------------------
# Generate images
# --------------------------
generated_image_paths = []
for expr in EXPRESSIONS:
    expr_folder = os.path.join(RAW_IMG_DIR, expr)
    os.makedirs(expr_folder, exist_ok=True)
    prompt, neg_prompt = get_prompt(expr)
    print(f"\nGenerating {GENERATE_PER_CLASS} images for {expr} ...")

    for i in range(GENERATE_PER_CLASS):
        out_path = os.path.join(expr_folder, f"{expr}_{i:04d}.jpg")
        if os.path.exists(out_path):
            generated_image_paths.append((out_path, expr))
            continue

        result = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=25,
            guidance_scale=7.0
        )
        img = result.images[0]
        img.save(out_path)
        generated_image_paths.append((out_path, expr))
print(f"\nGenerated {len(generated_image_paths)} raw synthetic images")

# --------------------------
# Run your detector + alignment; collect valid samples
# --------------------------
samples = []
labels = []
valid_count = 0

for img_path, expr in generated_image_paths:
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        continue

    # Detect face
    success, frame, head = detector.detect(img_bgr)
    if not success or head is None:
        continue

    # Align (this is where your "eye detection" filter lives)
    align_ok, aligned_head = detector.align_head(head)
    if not align_ok:
        continue

    # Save aligned face for visual inspection
    save_name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(ALIGNED_DIR, save_name), aligned_head)

    # Append in exactly the same format as your main.py
    samples.append(aligned_head.flatten())
    labels.append(expr)
    valid_count += 1

print(f"\nValid aligned samples after filtering: {valid_count}")
print(f"Class breakdown:")
for e in EXPRESSIONS:
    print(f"  {e}: {labels.count(e)}")

# --------------------------
# Merge into your training pickle format
# --------------------------
output_pkl = os.path.join("datasets", "faces_synthetic.pkl")
with open(output_pkl, "wb") as f:
    pickle.dump(samples, f)
    pickle.dump(labels, f)
print(f"\nSaved synthetic dataset to {output_pkl}")

# Optional: merge with your existing real data
real_pkl = os.path.join("datasets", "faces_training.pkl")
if os.path.exists(real_pkl):
    with open(real_pkl, "rb") as f:
        real_samples = pickle.load(f)
        real_labels = pickle.load(f)
    combined_samples = real_samples + samples
    combined_labels = real_labels + labels
    combined_path = os.path.join("datasets", "faces_combined_real_synth.pkl")
    with open(combined_path, "wb") as f:
        pickle.dump(combined_samples, f)
        pickle.dump(combined_labels, f)
    print(f"Merged real + synthetic saved to {combined_path}")
    print(f"Total combined samples: {len(combined_samples)}")