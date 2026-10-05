import os
import json
import shutil
from collections import defaultdict

root_dir = r"C:\Users\Lenovo\Desktop\VFCT - Copy\food_dataset\fruits-.v1i.coco-segmentation"

for split in ["train", "valid", "test"]:
    split_dir = os.path.join(root_dir, split)
    if not os.path.exists(split_dir):
        continue

    json_path = os.path.join(split_dir, "_annotations.coco.json")
    if not os.path.exists(json_path):
        print(f"Skipping {split}, no _annotations.coco.json found.")
        continue

    print(f"\nProcessing {split} split...")
    with open(json_path, "r") as f:
        coco = json.load(f)

    # Map category id to contiguous 0-indexed index
    cats = sorted(coco["categories"], key=lambda x: x["id"])
    cat_to_idx = {cat["id"]: i for i, cat in enumerate(cats)}

    images = {img["id"]: img for img in coco["images"]}
    img_to_anns = defaultdict(list)
    for ann in coco.get("annotations", []):
        img_to_anns[ann["image_id"]].append(ann)

    # Target directories
    images_dest = os.path.join(split_dir, "images")
    labels_dest = os.path.join(split_dir, "labels")
    os.makedirs(images_dest, exist_ok=True)
    os.makedirs(labels_dest, exist_ok=True)

    # Delete any stale cache files
    cache_path = os.path.join(root_dir, f"{split}.cache")
    if os.path.exists(cache_path):
        os.remove(cache_path)

    converted_count = 0
    for img_id, img_info in images.items():
        file_name = img_info["file_name"]
        w = img_info["width"]
        h = img_info["height"]

        # Move/copy image into images/ folder
        src_img = os.path.join(split_dir, file_name)
        dst_img = os.path.join(images_dest, file_name)
        if os.path.exists(src_img) and not os.path.exists(dst_img):
            shutil.move(src_img, dst_img)

        # Build normalized segmentation label lines
        lines = []
        for ann in img_to_anns.get(img_id, []):
            cat_id = ann["category_id"]
            cls_idx = cat_to_idx[cat_id]
            segs = ann.get("segmentation", [])

            for seg in segs:
                if len(seg) < 6:
                    continue
                # Normalize polygon coordinates (x / w, y / h)
                norm_coords = []
                for idx in range(0, len(seg), 2):
                    norm_coords.append(f"{seg[idx] / w:.6f}")
                    norm_coords.append(f"{seg[idx + 1] / h:.6f}")
                lines.append(f"{cls_idx} " + " ".join(norm_coords))

        label_base = os.path.splitext(file_name)[0] + ".txt"
        label_file = os.path.join(labels_dest, label_base)
        with open(label_file, "w") as lf:
            lf.write("\n".join(lines))
        converted_count += 1

    print(f"  Successfully organized {converted_count} images & labels for {split}.")

# Write clean data.yaml
names_dict = {i: c["name"] for i, c in enumerate(cats)}
yaml_text = f"""path: {root_dir.replace(chr(92), '/')}
train: train/images
val: valid/images
test: test/images

nc: {len(names_dict)}
names: {names_dict}
"""

yaml_path = os.path.join(root_dir, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(yaml_text)

print(f"\nConfiguration updated at: {yaml_path}")
print("Classes:", names_dict)