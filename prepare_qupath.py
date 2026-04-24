import os
import shutil
import cv2
import numpy as np
import csv

SLIDE_ID = "158927_B_S_1_MGG_06_01"

tmp_dir = os.environ.get('TMPDIR')
input_slide_dir = os.path.join(tmp_dir, "benign", SLIDE_ID)
csv_file = f"/home/piomac5688/projekt_wsi/csv_filtered_patches/{SLIDE_ID}/good_patches.csv"
output_slide_dir = f"/home/piomac5688/projekt_wsi/qupath_visualization/{SLIDE_ID}"

print(f"Preparing slide {SLIDE_ID}...")

good_patches = set()
with open(csv_file, mode='r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        good_patches.add(row[0])

print(f"Found {len(good_patches)} good patches in CSV file.")

os.makedirs(output_slide_dir, exist_ok=True)

all_patches = [p for p in os.listdir(input_slide_dir) if p.lower().endswith('.png')]
print(f"Processing {len(all_patches)} patches...")

for patch in all_patches:
    src_path = os.path.join(input_slide_dir, patch)
    dst_path = os.path.join(output_slide_dir, patch)

    if patch in good_patches:
        shutil.copy2(src_path, dst_path)
    else:
        img = cv2.imread(src_path)
        if img is not None:
            black_img = np.zeros_like(img)
            cv2.imwrite(dst_path, black_img)

print(f"Done! Files are located in: {output_slide_dir}")