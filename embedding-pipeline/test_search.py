# import torch
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.models import Filter

# Connect to in-memory Qdrant (same as embed.py)
client = QdrantClient(":memory:")
collection_name = "ui_designs"

# Load CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def embed_text(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    return features[0].numpy().tolist()

query = "minimal mobile dashboard with buttons"
query_vec = embed_text(query)

# Use search_points instead of search
results = client.search_points(
    collection_name=collection_name,
    query_vector=query_vec,
    limit=5,
    with_payload=True
)

for r in results:
    print(r.payload["image_path"], "-", r.payload["brief_description"])