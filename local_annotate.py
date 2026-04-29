from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import json
from pathlib import Path

def extract_components_from_captions(captions):
    text = " ".join(captions).lower()
    
    components = []

    if any(word in text for word in ["chart", "graph"]):
        components.append("chart")
    if any(word in text for word in ["table", "grid"]):
        components.append("table")
    if any(word in text for word in ["sidebar", "menu"]):
        components.append("sidebar")
    if any(word in text for word in ["button", "cta"]):
        components.append("button")
    if any(word in text for word in ["card", "panel"]):
        components.append("card")

    return list(set(components))

def detect_ui_type_from_captions(captions):
    text = " ".join(captions).lower()

    if "dashboard" in text or "analytics" in text:
        return "saas_dashboard"
    if "mobile" in text or "app" in text:
        return "mobile_app"
    if "website" in text or "landing" in text:
        return "landing_page"

    return "unknown"

# Load model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load 3 images safely
image_files = [
    p for p in Path("dataset/images/sample").glob("*.*")
    if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
][:3]

for img_path in image_files:
    image = Image.open(img_path).convert("RGB")

    # Generate multiple captions with hint
    inputs = processor(image, text="this is a ui design of", return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=50,
        num_return_sequences=3,
        do_sample=True,
        top_k=50
    )
    
    captions = [processor.decode(out, skip_special_tokens=True) for out in outputs]

    print(f"\n--- Output for {img_path.name} ---")
    print("Captions:", captions)
    
    # Convert to simple structured JSON
    annotation = {
        "ui_type": detect_ui_type_from_captions(captions),
        "components": extract_components_from_captions(captions),
        "layout_pattern": "unknown",
        "color_mode": "unknown",
        "descriptions": captions
    }

    print(json.dumps(annotation, indent=2))
