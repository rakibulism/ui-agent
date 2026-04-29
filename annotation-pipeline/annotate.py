import os
import requests, json, base64
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variable without requiring python-dotenv
env_file = BASE_DIR / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY="):
                os.environ["ALIBABA_API_KEY"] = line.split("=", 1)[1]

API_KEY = os.environ.get("ALIBABA_API_KEY", "")
URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"

images = list((BASE_DIR / "dataset/images/sample").glob("*.*"))
# Process just 1 image for testing
images = images[:1]

PROMPT_TEMPLATE = """
You are a senior UI/UX design analyst. Analyze this UI design image and return ONLY a valid JSON object with these fields:
ui_type, components, layout_pattern, color_mode, dominant_colors, design_style, design_patterns, quality_score, keywords, brief_description.
"""

for img in images:
    with open(img, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Correct multimodal formatting for DashScope/OpenAI compatible
    data = {
        "model": "qwen-vl-plus",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_TEMPLATE},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            }
        ]
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    resp = requests.post(URL, headers=headers, json=data)
    
    if resp.status_code != 200:
        print(f"API Error! {resp.status_code}\n{resp.text}")
        continue
        
    try:
        annotation = resp.json()["choices"][0]["message"]["content"]
        # Basic parsing to verify JSON structure
        if annotation.startswith("```json"):
            annotation = annotation.split("```json")[1].split("```")[0].strip()
        parsed_json = json.loads(annotation)
        print(json.dumps(parsed_json, indent=2))
        
        # Save JSON
        out_path = BASE_DIR / f"dataset/annotations/sample/{img.stem}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(parsed_json, f, indent=2)
            
        print(f"\nSuccessfully annotated and saved: {img.name}")
    except Exception as e:
        print(f"JSON Error: {e}\nRaw Output:\n{annotation}")