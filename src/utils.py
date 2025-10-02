import os
import json

USE_UNSLOTH = True

def validate_path(path: str):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    return path

def write_json(data: dict, path: str):
    validate_path(path)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved data to {path}")