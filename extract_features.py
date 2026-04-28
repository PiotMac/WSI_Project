import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from PIL import Image
from TransPath.ctran import ctranspath
import subprocess
import shutil

CSV_BASE_DIR = "/home/piomac5688/projekt_wsi/csv_filtered_patches/malignant"
OUTPUT_FEATURES_DIR = "/home/piomac5688/projekt_wsi/extracted_features_malignant"
WEIGHTS_PATH = "/home/piomac5688/projekt_wsi/ctranspath.pth"
ZIP_PATH = "/lustre/pd03/hpc-ljelen-1692966897/WSI/WSI_patched_png/malignant.zip"
TMPDIR = os.environ.get('TMPDIR', '/tmp')

os.makedirs(OUTPUT_FEATURES_DIR, exist_ok=True)


class WSIPatchDataset(Dataset):
    def __init__(self, csv_file, patch_dir):
        self.patch_dir = patch_dir
        self.patches = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.patches.append(row[0])

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_name = self.patches[idx]
        img_path = os.path.join(self.patch_dir, patch_name)
        image = Image.open(img_path).convert('RGB')
        return self.transform(image), patch_name


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Używane urządzenie: {device}")
    if device.type == 'cpu':
        print("OSTRZEŻENIE: Brak GPU!")

    print("Ładowanie modelu CTransPath...")
    model = ctranspath()
    model.head = nn.Identity()
    td = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(td['model'], strict=True)
    model = model.to(device)

    model.eval()

    slides = [d for d in os.listdir(CSV_BASE_DIR) if os.path.isdir(os.path.join(CSV_BASE_DIR, d))]
    print(f"Znaleziono {len(slides)} pacjentów do ekstrakcji cech.")

    with torch.no_grad():
        for i, slide_name in enumerate(slides, 1):
            output_pt_file = os.path.join(OUTPUT_FEATURES_DIR, f"{slide_name}.pt")

            if os.path.exists(output_pt_file):
                print(f"[{i}/{len(slides)}] Pomijam {slide_name} - cechy już wyekstrahowane.")
                continue

            print(f"[{i}/{len(slides)}] Rozpakowywanie: {slide_name}...")
            target_path_in_zip = f"malignant/{slide_name}/*"
            subprocess.run(["unzip", "-q", ZIP_PATH, target_path_in_zip, "-d", TMPDIR], check=False)

            csv_file = os.path.join(CSV_BASE_DIR, slide_name, "good_patches.csv")
            patch_dir = os.path.join(TMPDIR, "malignant", slide_name)

            if not os.path.exists(patch_dir):
                print(f"[{i}/{len(slides)}] POMIJAM {slide_name} - Brak wypakowanych zdjęć w TMPDIR!")
                continue

            print(f"[{i}/{len(slides)}] Ekstrakcja cech dla: {slide_name}...")

            dataset = WSIPatchDataset(csv_file, patch_dir)
            dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)

            slide_features = []

            for batch_idx, (images, _) in enumerate(dataloader):
                images: torch.Tensor
                images = images.to(device)
                features = model(images)
                slide_features.append(features.cpu())

            if len(slide_features) > 0:
                final_tensor = torch.cat(slide_features, dim=0)
                torch.save(final_tensor, output_pt_file)
                print(f"   -> Zapisano tensor o wymiarach {final_tensor.shape} do pliku {slide_name}.pt")
            else:
                print(f"   -> Ostrzeżenie: Slajd {slide_name} nie miał żadnych dobrych patchy.")

            shutil.rmtree(patch_dir, ignore_errors=True)

if __name__ == "__main__":
    main()