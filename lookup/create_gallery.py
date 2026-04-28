import os
import cv2
import numpy as np
import csv
import math

SLIDE_ID = "128249_M_3_S_MGG_14_03"
MAX_PATCHES = 209995
PATCH_DISPLAY_SIZE = 32

tmp_dir = os.environ.get('TMPDIR')
input_slide_dir = os.path.join(tmp_dir, "malignant", SLIDE_ID)
csv_file = f"/home/piomac5688/projekt_wsi/csv_filtered_patches/malignant/{SLIDE_ID}/good_patches.csv"
output_image_path = f"/home/piomac5688/projekt_wsi/lookup/{SLIDE_ID}_lookup_filtered.jpg"

print(f"Rozpoczynam generowanie galerii dla: {SLIDE_ID}")

good_patches = set()
if os.path.exists(csv_file):
    with open(csv_file, mode='r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            good_patches.add(row[0])
else:
    print(f"Błąd: Nie znaleziono pliku {csv_file}. Najpierw przepuść tego pacjenta przez główny skrypt filtracji!")
    exit()

all_patches = [p for p in os.listdir(input_slide_dir) if p.lower().endswith('.png')]
all_patches.sort()
patches_to_show = all_patches[:MAX_PATCHES]

print(f"Tworzę siatkę z {len(patches_to_show)} patchy...")

# grid_size = math.ceil(math.sqrt(len(patches_to_show)))
grid_size = 656

grid_rows = []
current_row = []

for patch_name in patches_to_show:
    src_path = os.path.join(input_slide_dir, patch_name)
    img = cv2.imread(src_path)

    if img is None:
        continue

    if patch_name not in good_patches:
        img = np.zeros_like(img)

    img_resized = cv2.resize(img, (PATCH_DISPLAY_SIZE, PATCH_DISPLAY_SIZE))
    current_row.append(img_resized)

    if len(current_row) == grid_size:
        grid_rows.append(np.hstack(current_row))
        current_row = []

if len(current_row) > 0:
    while len(current_row) < grid_size:
        current_row.append(np.zeros((PATCH_DISPLAY_SIZE, PATCH_DISPLAY_SIZE, 3), dtype=np.uint8))
    grid_rows.append(np.hstack(current_row))

final_image = np.vstack(grid_rows)


success = cv2.imwrite(output_image_path, final_image)

if success:
    print(f"Sukces! Galeria zapisana u Ciebie w folderze jako: {SLIDE_ID}_lookup_filtered.jpg")
else:
    print(f"BŁĄD ZAPISU! OpenCV nie mogło zapisać obrazu (prawdopodobnie obraz nadal jest zbyt duży lub zła ścieżka).")
