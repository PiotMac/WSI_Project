import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score, confusion_matrix

from abmil_model import WSIFeatureDataset, AttentionMIL

CSV_FILE = "/home/piomac5688/projekt_wsi/dataset_abmil.csv"
MODEL_DIR = "models/ens_drop"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_ensemble():
    print("Loading the data and recreating test set...")
    full_dataset = WSIFeatureDataset(CSV_FILE)
    labels = full_dataset.data['label'].values
    indices = np.arange(len(labels))

    _, test_idx, _, _ = train_test_split(
        indices, labels, test_size=0.20, stratify=labels, random_state=42
    )

    test_sub = Subset(full_dataset, test_idx)
    test_loader = DataLoader(test_sub, batch_size=1, shuffle=False)

    ensemble_models = []

    model_path = f"{MODEL_DIR}/global_best_model_fold_1_drop_0.05.pth"
    model = AttentionMIL().to(DEVICE)

    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        ensemble_models.append(model)
    except FileNotFoundError:
        print(f"ERROR: No model {model_path} found!")
        return

    test_labels = []
    test_preds = []

    with torch.no_grad():
        for features, label in test_loader:
            features, label = features.to(DEVICE), label.to(DEVICE)

            y_prob, _ = model(features)

            test_labels.append(label.item())
            test_preds.append(y_prob.item())

    test_labels = np.array(test_labels)
    test_preds = np.array(test_preds)
    test_preds_binary = (test_preds > 0.5).astype(int)

    test_auc = roc_auc_score(test_labels, test_preds)
    test_acc = accuracy_score(test_labels, test_preds_binary)
    test_rec = recall_score(test_labels, test_preds_binary, zero_division=0)
    test_prec = precision_score(test_labels, test_preds_binary, zero_division=0)
    test_f1 = f1_score(test_labels, test_preds_binary, zero_division=0)

    cm = confusion_matrix(test_labels, test_preds_binary)
    tn, fp, fn, tp = cm.ravel()

    print(f"{'*' * 40}")
    print("SINGLE MODEL SCORE ON TEST SET (Threshold = 0.5):")
    print(f"{'*' * 40}")
    print(f"AUC:       {test_auc:.4f}")
    print(f"Accuracy:  {test_acc:.4f} ({test_acc * 100:.1f}%)")
    print(f"Precision: {test_prec:.4f} ({test_prec * 100:.1f}%)")
    print(f"Recall:    {test_rec:.4f} ({test_rec * 100:.1f}%)")
    print(f"F1-Score:  {test_f1:.4f} ({test_f1 * 100:.1f}%)")
    print(f"\nCONFUSION MATRIX:")
    print(f"                 Predicted: BENIGN(0) | MALIGNANT(1)")
    print(f"Actual BENIGN(0):      {tn:4d} (TN)   |   {fp:4d} (FP)")
    print(f"Actual MALIGNANT(1):   {fn:4d} (FN)   |   {tp:4d} (TP)")
    print(f"{'*' * 40}")


if __name__ == "__main__":
    evaluate_ensemble()