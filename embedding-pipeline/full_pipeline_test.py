import json
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Paths
images_folder = Path("../dataset/images/sample")
annotations_folder = Path("../dataset/annotations/sample")

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Connect to in-memory Qdrant
client = QdrantClient(":memory:")
collection_name = "ui_designs"
client.recreate_collection(
    collection_name=collection_name,
    vectors_config={
        "image": VectorParams(size=768, distance=Distance.COSINE),
        "text": VectorParams(size=768, distance=Distance.COSINE),
    },
)

# Functions to embed image and text
def embed_image(image_path: Path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return features[0].numpy().tolist()

def embed_text(text: str):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    return features[0].numpy().tolist()

# Index all sample images
for annotation_file in annotations_folder.glob("*.json"):
    with open(annotation_file, "r") as f:
        annotation = json.load(f)

    image_path = images_folder / f"{annotation_file.stem}.png"
    if not image_path.exists():
        print(f"Image not found: {image_path}, skipping")
        continue

    img_vec = embed_image(image_path)
    txt_content = f"{annotation['brief_description']} " + " ".join(annotation.get("keywords", [])) + " " + " ".join(annotation.get("components", []))
    txt_vec = embed_text(txt_content)

    point = PointStruct(
        id=hash(annotation_file.stem) % (2**63),
        vector={"image": img_vec, "text": txt_vec},
        payload={
            "image_path": str(image_path),
            **annotation
        }
    )

    client.upsert(collection_name=collection_name, points=[point])
    print(f"Indexed {annotation_file.stem}")

print("\nAll images indexed! Now you can run queries.\n")

# Simple text query
query_text = "minimal mobile dashboard with buttons"
query_vec = embed_text(query_text)

# Search
results = client.search_points(
    collection_name=collection_name,
    query_vector=query_vec,
    limit=5,
    with_payload=True
)

print(f"Top matches for query: '{query_text}'\n")
for r in results:
    print(r.payload["image_path"], "-", r.payload["brief_description"])