import os
import shutil
import glob
import yaml

base_dir = os.path.abspath("food_dataset")

# Find the extracted roboflow coco folder
coco_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and "coco" in d.lower()]
if not coco_dirs:
    raise FileNotFoundError("Could not find coco folder in food_dataset")

source_root = os.path.join(base_dir, coco_dirs[0])
print(f"Found source dataset at: {source_root}")

# Map and check train / valid / test directories
splits = {}
for split_name in ["train", "valid", "val", "test"]:
    p = os.path.join(source_root, split_name)
    if os.path.exists(p):
        splits[split_name] = p

train_dir = splits.get("train")
val_dir = splits.get("valid") or splits.get("val")
test_dir = splits.get("test")

# If food_dataset_yolo generated labels, copy them into the split directories
converted_yolo_dir = os.path.abspath("food_dataset_yolo")
if os.path.exists(converted_yolo_dir):
    # Find all label txt files
    for root, _, files in os.walk(converted_yolo_dir):
        txt_files = [f for f in files if f.endswith(".txt") and not f.startswith("README")]
        if txt_files:
            # Determine target split
            target_split = None
            if "train" in root.lower() and train_dir:
                target_split = os.path.join(train_dir, "labels")
            elif ("valid" in root.lower() or "val" in root.lower()) and val_dir:
                target_split = os.path.join(val_dir, "labels")
            elif "test" in root.lower() and test_dir:
                target_split = os.path.join(test_dir, "labels")

            if target_split:
                os.makedirs(target_split, exist_ok=True)
                for f in txt_files:
                    src = os.path.join(root, f)
                    dst = os.path.join(target_split, f)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)

print("Synchronized label files with image folders.")

# Read existing classes from food_dataset_yolo/data.yaml if present
yaml_ref = os.path.join(converted_yolo_dir, "data.yaml")
names = {}
if os.path.exists(yaml_ref):
    with open(yaml_ref, "r") as f:
        old_cfg = yaml.safe_load(f)
        names = old_cfg.get("names", {})

# Generate the robust data.yaml
final_yaml = {
    "path": source_root.replace("\\", "/"),
    "train": "train",
    "val": "valid" if "valid" in splits else "val",
    "nc": len(names) if names else 1,
    "names": names if names else {0: "fruit"}
}
if test_dir:
    final_yaml["test"] = "test"

final_yaml_path = os.path.join(source_root, "data.yaml")
with open(final_yaml_path, "w") as f:
    yaml.dump(final_yaml, f, default_flow_style=False, sort_keys=False)

print(f"\nCreated verified data.yaml at: {final_yaml_path}")
print(f"Update your train_food.py to use: data='{final_yaml_path.replace(chr(92), '/')}'")