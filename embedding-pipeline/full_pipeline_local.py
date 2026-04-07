import os
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import json

# -------------------------------
# Config
# -------------------------------
DATASET_DIR = Path("../dataset/images/sample")
ANNOTATIONS_DIR = Path("../dataset/annotations/sample")
COLLECTION_NAME = "ui_designs"
QDRANT_PATH = "qdrant_local"

# -------------------------------
# Qdrant Client (Local Storage)
# -------------------------------
client = QdrantClient(path=QDRANT_PATH)

# Create collection if not exists
if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "image": VectorParams(size=768, distance=Distance.COSINE),
            "text": VectorParams(size=768, distance=Distance.COSINE),
        }
    )

# -------------------------------
# Load CLIP
# -------------------------------
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def embed_image(image_path: Path) -> list[float]:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return features[0].numpy().tolist()

def embed_text(text: str) -> list[float]:
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    return features[0].numpy().tolist()

# -------------------------------
# Index Images
# -------------------------------
for img_file in DATASET_DIR.glob("*"):
    ann_file = ANNOTATIONS_DIR / f"{img_file.stem}.json"
    if not ann_file.exists():
        continue  # skip if annotation missing

    with open(ann_file) as f:
        annotation = json.load(f)

    text_content = f"{annotation.get('brief_description','')} {' '.join(annotation.get('keywords',[]))} {' '.join(annotation.get('components',[]))}"
    img_vec = embed_image(img_file)
    txt_vec = embed_text(text_content)

    point = PointStruct(
        id=hash(img_file.name) % (2**63),
        vector={"image": img_vec, "text": txt_vec},
        payload={
            "image_path": str(img_file),
            "ui_type": annotation.get("ui_type"),
            "components": annotation.get("components"),
            "design_style": annotation.get("design_style"),
            "quality_score": annotation.get("quality_score", {}).get("overall", 0),
            "color_mode": annotation.get("color_mode"),
            "brief_description": annotation.get("brief_description"),
            "keywords": annotation.get("keywords"),
        }
    )

    client.upsert(collection_name=COLLECTION_NAME, points=[point])
    print(f"Indexed {img_file.name}")

# -------------------------------
# Test Search
# -------------------------------
query_text = "dark dashboard with sidebar and table"
query_vec = embed_text(query_text)

results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vec,
    vector_name="text",
    limit=5,
    with_payload=True
)

print("\nTop results:")
for r in results:
    print(r.payload["image_path"], "-", r.payload["brief_description"])