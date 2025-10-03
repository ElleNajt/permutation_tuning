import os
import json
import shutil
from dataclasses import dataclass

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

def save_dataset(dataset: list, path: str):
    validate_path(path)
    with open(path, 'w') as f:
        json.dump([x.to_dict() for x in dataset], f, indent=2)
    print(f"Saved dataset to {path}")


def copy_move_file(src: str, dst: str):
    shutil.copy(src, dst)