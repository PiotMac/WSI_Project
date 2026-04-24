import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import csv
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

original_catalog = "/lustre/pd03/hpc-ljelen-1692966897/WSI/WSI_patched_png/benign/215793_B_S_1_MGG_05_01"
csv_file = "/home/piomac5688/projekt_wsi/csv_filtered_patches/215793_B_S_1_MGG_05_01/good_patches.csv"

good_patches = []
with open(csv_file, mode='r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        good_patches.append(row[0])

print(f"Found {len(good_patches)} accepted patches in CSV file.")

number_to_show = min(36, len(good_patches))
chosen_patches = random.sample(good_patches, number_to_show)

fig, axes = plt.subplots(6, 6, figsize=(12, 12))
fig.suptitle(f"Random {number_to_show} filtered patches", fontsize=16)

for i, ax in enumerate(axes.flat):
    if i < number_to_show:
        filename = chosen_patches[i]
        full_path = os.path.join(original_catalog, filename)

        img = mpimg.imread(full_path)
        print(f"Statystyki patcha {filename}: Min={img.min()}, Max={img.max()}, Mean={img.mean()}, Std={img.std()}")
        ax.imshow(img)
        ax.set_title(filename, fontsize=8)

    ax.axis('off')

plt.tight_layout()
plt.savefig("/home/piomac5688/projekt_wsi/chosen_patches_final.png")