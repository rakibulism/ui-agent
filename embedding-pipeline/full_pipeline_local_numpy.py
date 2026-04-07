# embedding-pipeline/full_pipeline_local_numpy.py
import json
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# Paths
IMAGES_DIR = Path("../dataset/images/sample")
ANNOTATIONS_DIR = Path("../dataset/annotations/sample")

# Load CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Functions
def embed_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return features[0].numpy()

def embed_text(text: str) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    return features[0].numpy()

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -------------------------------
# Step 1: Index all images locally
# -------------------------------
index = []

for ann_file in ANNOTATIONS_DIR.glob("*.json"):
    with open(ann_file) as f:
        ann = json.load(f)
    img_file = IMAGES_DIR / f"{ann_file.stem}.png"
    if not img_file.exists():
        continue
    img_vec = embed_image(img_file)
    txt_vec = embed_text(f"{ann['brief_description']} {' '.join(ann.get('keywords', []))}")
    index.append({
        "image_path": str(img_file),
        "annotation": ann,
        "img_vec": img_vec,
        "txt_vec": txt_vec
    })
    print(f"Indexed {img_file.name}")

# -------------------------------
# Step 2: Query
# -------------------------------
query_text = "minimal mobile dashboard with buttons"
query_vec = embed_text(query_text)

# Compute cosine similarity vs. all indexed text embeddings
results = sorted(index, key=lambda x: cosine_sim(query_vec, x["txt_vec"]), reverse=True)[:5]

print("\nTop matches:")
for r in results:
    print(r["image_path"], "-", r["annotation"]["brief_description"])