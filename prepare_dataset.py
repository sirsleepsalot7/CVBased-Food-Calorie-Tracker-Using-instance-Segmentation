import os
import shutil
import json
from ultralytics.data.converter import convert_coco

# 1. Locate the extracted directory
base_dir = "food_dataset"
subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and "coco-seg" in d]

if not subdirs:
    raise FileNotFoundError("Could not find the extracted coco-seg folder inside food_dataset.")

coco_dir = os.path.join(base_dir, subdirs[0])
print(f"Converting annotations in: {coco_dir}")

# 2. Run Ultralytics COCO -> YOLO segmentation converter
# use_segments=True extracts full polygon masks instead of bounding boxes
convert_coco(labels_dir=coco_dir, save_dir="food_dataset_yolo", use_segments=True)

# 3. Read classes from the train annotations JSON to create data.yaml
train_json_path = os.path.join(coco_dir, "train", "_annotations.coco.json")
with open(train_json_path, "r") as f:
    coco_data = json.load(f)

# Sort categories by id to ensure index alignment
categories = sorted(coco_data["categories"], key=lambda x: x["id"])
class_names = {idx: cat["name"] for idx, cat in enumerate(categories)}

print(f"Extracted classes ({len(class_names)}): {class_names}")

# 4. Generate data.yaml for training
yaml_content = f"""path: {os.path.abspath('food_dataset_yolo')}
train: train/images
val: valid/images
test: test/images

nc: {len(class_names)}
names: {class_names}
"""

yaml_path = os.path.join("food_dataset_yolo", "data.yaml")
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"Created configuration at {yaml_path}")
print("Ready for training!")