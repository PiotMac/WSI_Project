import os
import cv2
import numpy as np
import csv

tmp_dir = os.environ.get('TMPDIR')
input_base_dir = os.path.join(tmp_dir, "malignant")
# input_base_dir = "/lustre/pd03/hpc-ljelen-1692966897/WSI/WSI_patched_png/benign"
output_base_dir = "/home/piomac5688/projekt_wsi/csv_filtered_patches/malignant"

TISSUE_THRESHOLD = 10.0


def filter_patches(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return 0.0, 0.0, 0.0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    sat_blurred = cv2.medianBlur(saturation, 5)
    _, mask = cv2.threshold(sat_blurred, 20, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    no_pixels = img.shape[0] * img.shape[1]
    no_tissue_pixels = cv2.countNonZero(mask_closed)
    tissue_percentage = (no_tissue_pixels / no_pixels) * 100

    return tissue_percentage, img.mean() / 255.0, img.std() / 255.0


if __name__ == "__main__":
    os.makedirs(output_base_dir, exist_ok=True)

    slide_folders = [f for f in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, f))]

    no_slides = len(slide_folders)
    print(f"Found {no_slides} slides to process.")
    print("-" * 50)

    for i, slide_name in enumerate(slide_folders, 1):
        slide_path = os.path.join(input_base_dir, slide_name)

        patches = [p for p in os.listdir(slide_path) if p.lower().endswith('.png')]
        no_patches = len(patches)

        if no_patches == 0:
            print(f"[{i}/{no_slides}] Slide {slide_name} is empty!.")
            continue

        print(f"[{i}/{no_slides}] Slide analysis: {slide_name} ({no_patches} patches)...")

        dest_slide_folder = os.path.join(output_base_dir, slide_name)
        os.makedirs(dest_slide_folder, exist_ok=True)

        file_csv_path = os.path.join(dest_slide_folder, "good_patches.csv")

        accepted_patches_per_slide = 0

        with open(file_csv_path, mode='w', newline='') as plik_csv:
            writer = csv.writer(plik_csv)
            writer.writerow(['Patch_name', 'Percentage_of_tissue', 'Mean', 'Std'])

            for patch in patches:
                full_patch_path = os.path.join(slide_path, patch)
                tissue_percentage, img_mean, img_std = filter_patches(full_patch_path)

                if tissue_percentage >= TISSUE_THRESHOLD and img_mean >= 0.15 and img_std >= 0.05:
                    writer.writerow([patch, round(tissue_percentage, 2), round(img_mean, 3), round(img_std, 3)])
                    accepted_patches_per_slide += 1

        print(f"    -> Done! Saved {accepted_patches_per_slide} 'good patches' from {no_patches}.")

    print("-" * 50)
    print(f"Finished. Results are located in: {output_base_dir}")