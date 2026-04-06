import requests, json, base64
from pathlib import Path

API_KEY = "__Your__OpenRouter__API__Key__"  # Replace with your actual key
URL = "https://api.openrouter.ai/v1/chat/completions"

images = list(Path("../dataset/images/sample").glob("*.*"))

PROMPT_TEMPLATE = """
You are a senior UI/UX design analyst. Analyze this UI design image and return ONLY a valid JSON object with these fields:
ui_type, components, layout_pattern, color_mode, dominant_colors, design_style, design_patterns, quality_score, keywords, brief_description.
"""

for img in images:
    with open(img, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    data = {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "user", "content": PROMPT_TEMPLATE + f"\nImage (base64): {img_b64}"}
        ]
    }

    headers = {"Authorization": f"Bearer {API_KEY}"}
    resp = requests.post(URL, headers=headers, json=data)
    annotation = resp.json()["choices"][0]["message"]["content"]

    # Save JSON
    out_path = Path(f"../dataset/annotations/sample/{img.stem}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(annotation)

    print(f"Annotated: {img.name}")