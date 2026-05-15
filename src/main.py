from src.pipeline.analyzer import analyze_panel
import json

if __name__ == "__main__":
    results = analyze_panel("data/test.jpg")
    print(json.dumps(
        [{k: v for k, v in r.items() if k != "image_path"} for r in results],
        indent=2
    ))
