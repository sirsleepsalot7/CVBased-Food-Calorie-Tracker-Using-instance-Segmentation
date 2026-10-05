import os
import yaml

base_dir = os.path.abspath("food_dataset_yolo")

# Identify the actual validation folder name ('val' vs 'valid')
val_name = "val" if os.path.exists(os.path.join(base_dir, "val")) else "valid"
train_name = "train" if os.path.exists(os.path.join(base_dir, "train")) else "images"

# Also check whether images are inside 'images/' subfolder or directly in the split folder
def resolve_split_path(split):
    p1 = os.path.join(base_dir, split, "images")
    p2 = os.path.join(base_dir, split)
    if os.path.exists(p1):
        return os.path.join(split, "images")
    return split

train_path = resolve_split_path(train_name)
val_path = resolve_split_path(val_name)
test_path = resolve_split_path("test") if os.path.exists(os.path.join(base_dir, "test")) else None

yaml_file = os.path.join(base_dir, "data.yaml")
with open(yaml_file, "r") as f:
    cfg = yaml.safe_load(f)

cfg["path"] = base_dir.replace("\\", "/")
cfg["train"] = train_path.replace("\\", "/")
cfg["val"] = val_path.replace("\\", "/")
if test_path:
    cfg["test"] = test_path.replace("\\", "/")

with open(yaml_file, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

print(f"Fixed data.yaml:")
print(f"  Root:  {cfg['path']}")
print(f"  Train: {cfg['train']}")
print(f"  Val:   {cfg['val']}")