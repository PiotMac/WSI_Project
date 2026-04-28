import os
import pandas as pd

BASE_DIR = "/home/piomac5688/projekt_wsi"
PATH_BENIGN = os.path.join(BASE_DIR, "extracted_features/benign")
PATH_MALIGNANT = os.path.join(BASE_DIR, "extracted_features/malignant")

dataset = []

if os.path.exists(PATH_BENIGN):
    for filename in os.listdir(PATH_BENIGN):
        if filename.endswith(".pt"):
            dataset.append({
                'slide_id': filename.replace('.pt', ''),
                'label': 0,
                'path': os.path.join(PATH_BENIGN, filename)
            })
else:
    print(f"Ostrzeżenie: Nie znaleziono folderu {PATH_BENIGN}")

if os.path.exists(PATH_MALIGNANT):
    for filename in os.listdir(PATH_MALIGNANT):
        if filename.endswith(".pt"):
            dataset.append({
                'slide_id': filename.replace('.pt', ''),
                'label': 1,
                'path': os.path.join(PATH_MALIGNANT, filename)
            })
else:
    print(f"Ostrzeżenie: Nie znaleziono folderu {PATH_MALIGNANT}")

df = pd.DataFrame(dataset)

df = df.sort_values('slide_id').reset_index(drop=True)

OUTPUT_FILE = os.path.join(BASE_DIR, "dataset_abmil.csv")
df.to_csv(OUTPUT_FILE, index=False)

print("-" * 30)
print(f"SUKCES: Utworzono plik {OUTPUT_FILE}")
print(f"Łączna liczba pacjentów: {len(df)}")
print(f"  - Benign (0): {len(df[df.label == 0])}")
print(f"  - Malignant (1): {len(df[df.label == 1])}")
print("-" * 30)