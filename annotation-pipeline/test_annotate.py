import json
from pathlib import Path

# Folder with your sample images
images_folder = "dataset/images/sample"  # correct relative path from project root
output_folder = "dataset/annotations/sample"

images = list(Path(images_folder).glob("*.*"))

# Ensure output folder exists
Path(output_folder).mkdir(parents=True, exist_ok=True)

for img in images:
    # Placeholder annotation
    annotation = {
        "ui_type": "mobile_app",
        "components": ["button", "navbar"],
        "layout_pattern": "single_column",
        "color_mode": "light",
        "dominant_colors": ["#ffffff", "#000000"],
        "design_style": ["minimal"],
        "design_patterns": ["onboarding"],
        "quality_score": {
            "overall": 7,
            "visual_hierarchy": 7,
            "spacing_consistency": 6,
            "typography": 7,
            "color_harmony": 7,
            "component_consistency": 7
        },
        "keywords": ["simple", "clean", "dashboard"],
        "brief_description": f"Placeholder annotation for {img.name}"
    }

    # Save JSON
    out_path = Path(output_folder) / f"{img.stem}.json"
    with open(out_path, "w") as f:
        json.dump(annotation, f, indent=2)

    print(f"Created placeholder JSON: {img.name}")

print(f"\nDone! Created {len(images)} placeholder JSON files in {output_folder}")