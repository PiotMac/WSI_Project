import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score, confusion_matrix, \
    roc_curve
import numpy as np

from abmil_model import WSIFeatureDataset, AttentionMIL

CSV_FILE = "/home/piomac5688/projekt_wsi/dataset_abmil.csv"
EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DROPOUT_RATES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


def train_and_evaluate():
    print(f"Beginning training on: {DEVICE}")

    full_dataset = WSIFeatureDataset(CSV_FILE)

    labels = full_dataset.data['label'].values
    indices = np.arange(len(labels))

    tv_idx, test_idx, tv_labels, test_labels = train_test_split(
        indices, labels, test_size=0.20, stratify=labels, random_state=42
    )

    test_sub = Subset(full_dataset, test_idx)
    test_loader = DataLoader(test_sub, batch_size=1, shuffle=False)

    tv_sub = Subset(full_dataset, tv_idx)
    tv_loader = DataLoader(tv_sub, batch_size=1, shuffle=False)

    print(f"\nData split:")
    print(f"-> Train set + validation set (80%): {len(tv_idx)} patients")
    print(f"-> Test set (20%): {len(test_idx)} patients")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_overall_drop_rate = None
    best_overall_val_auc = 0.0
    saved_ensemble_paths_for_best_drop = []

    for drop_rate in DROPOUT_RATES:
        print(f"\n{'#' * 60}")
        print(f"TESTING DROPOUT RATE = {drop_rate}")
        print(f"{'#' * 60}")

        fold_metrics = []
        current_drop_model_paths = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(tv_idx, tv_labels)):
            print(f"\n{'=' * 10} FOLD {fold + 1}/5 (Dropout: {drop_rate}) {'=' * 10}")

            train_sub = Subset(full_dataset, tv_idx[train_idx])
            val_sub = Subset(full_dataset, tv_idx[val_idx])

            train_loader = DataLoader(train_sub, batch_size=1, shuffle=True)
            val_loader = DataLoader(val_sub, batch_size=1, shuffle=False)

            model = AttentionMIL(dropout_rate=drop_rate).to(DEVICE)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

            best_auc = 0.0
            best_val_loss = float('inf')
            epochs_no_improve = 0

            fold_model_path = f"models/ens_drop/best_model_fold_{fold + 1}_drop_{drop_rate}.pth"

            for epoch in range(EPOCHS):
                model.train()
                train_loss = 0.0

                for features, label in train_loader:
                    features, label = features.to(DEVICE), label.to(DEVICE)

                    optimizer.zero_grad()
                    y_prob, _ = model(features)

                    loss = criterion(y_prob.squeeze(), label.squeeze())
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                model.eval()
                val_labels = []
                val_preds = []
                val_loss = 0.0

                with torch.no_grad():
                    for features, label in val_loader:
                        features, label = features.to(DEVICE), label.to(DEVICE)

                        y_prob, _ = model(features)
                        loss = criterion(y_prob.squeeze(), label.squeeze())
                        val_loss += loss.item()

                        val_labels.append(label.item())
                        val_preds.append(y_prob.item())

                val_labels = np.array(val_labels)
                val_preds = np.array(val_preds)
                val_preds_binary = (val_preds > 0.5).astype(int)

                try:
                    auc = roc_auc_score(val_labels, val_preds)
                except ValueError:
                    auc = 0.5

                acc = accuracy_score(val_labels, val_preds_binary)

                print(f"Epoch {epoch + 1:02d}/{EPOCHS} | Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {val_loss / len(val_loader):.4f} | Val AUC: {auc:.4f} | ACC: {acc:.4f}")

                if auc > best_auc or (auc == best_auc and val_loss < best_val_loss):
                    best_auc = auc
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), fold_model_path)
                else:
                    epochs_no_improve += 1
                    print(f"   -> No better AUC result since {epochs_no_improve} epochs.")

                if epochs_no_improve >= PATIENCE:
                    print(f"EARLY STOPPING: Training in fold {fold + 1} interrupted in epoch {epoch + 1}!")
                    break

            fold_metrics.append(best_auc)
            current_drop_model_paths.append(fold_model_path)
            print(f"-> The best AUC in fold {fold + 1}: {best_auc:.4f}")

        mean_val_auc = np.mean(fold_metrics)
        print(f"\n-> Mean AUC from 5 folds (Dropout={drop_rate}): {mean_val_auc:.4f} (+/- {np.std(fold_metrics):.4f})")

        if mean_val_auc > best_overall_val_auc:
            best_overall_val_auc = mean_val_auc
            best_overall_drop_rate = drop_rate
            saved_ensemble_paths_for_best_drop = current_drop_model_paths

    print(f"\n{'=' * 50}")
    print(f"THE BEST DROPOUT RATE: {best_overall_drop_rate} (Mean Val AUC: {best_overall_val_auc:.4f})")
    print(f"{'=' * 50}")

    ensemble_models = []
    for path in saved_ensemble_paths_for_best_drop:
        m = AttentionMIL(dropout_rate=best_overall_drop_rate).to(DEVICE)
        m.load_state_dict(torch.load(path))
        m.eval()
        ensemble_models.append(m)

    print(f"\n{'*' * 40}")
    print("TUNING DECISION THRESHOLD ON TRAIN/VAL SET (80%)")
    print(f"{'*' * 40}")

    tv_eval_labels = []
    tv_eval_preds = []
    with torch.no_grad():
        for features, label in tv_loader:
            features, label = features.to(DEVICE), label.to(DEVICE)
            patient_probs = [m(features)[0].item() for m in ensemble_models]
            tv_eval_preds.append(np.mean(patient_probs))
            tv_eval_labels.append(label.item())

    fpr, tpr, thresholds = roc_curve(tv_eval_labels, tv_eval_preds)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    print(f"Default threshold: 0.5000")
    print(f"NEW OPTIMAL THRESHOLD: {optimal_threshold:.4f}\n")

    print(f"\n{'*' * 40}")
    print("EVALUATING THE WINNING ENSEMBLE MODEL ON TEST SET")
    print(f"{'*' * 40}")

    test_labels, test_preds = [], []
    with torch.no_grad():
        for features, label in test_loader:
            features, label = features.to(DEVICE), label.to(DEVICE)

            patient_probs = []
            for m in ensemble_models:
                y_prob, _ = m(features)
                patient_probs.append(y_prob.item())

            avg_prob = np.mean(patient_probs)

            test_labels.append(label.item())
            test_preds.append(avg_prob)

    test_labels = np.array(test_labels)
    test_preds = np.array(test_preds)
    test_preds_binary_default = (test_preds > 0.5).astype(int)

    test_auc = roc_auc_score(test_labels, test_preds)
    test_acc = accuracy_score(test_labels, test_preds_binary_default)
    test_rec = recall_score(test_labels, test_preds_binary_default, zero_division=0)
    test_prec = precision_score(test_labels, test_preds_binary_default, zero_division=0)
    test_f1 = f1_score(test_labels, test_preds_binary_default, zero_division=0)

    cm = confusion_matrix(test_labels, test_preds_binary_default)
    tn, fp, fn, tp = cm.ravel()

    print("ENSEMBLE SCORE ON TEST SET (DEFAULT THRESHOLD = 0.5):")
    print(f"AUC:       {test_auc:.4f}")
    print(f"Accuracy:  {test_acc:.4f} ({test_acc * 100:.1f}%)")
    print(f"Precision: {test_prec:.4f} ({test_prec * 100:.1f}%)")
    print(f"Recall:    {test_rec:.4f} ({test_rec * 100:.1f}%)")
    print(f"F1-Score:  {test_f1:.4f} ({test_f1 * 100:.1f}%)")
    print(f"{'*' * 40}\n")
    print("CONFUSION MATRIX:")
    print(f"                 Predicted: BENIGN(0) | MALIGNANT(1)")
    print(f"Actual BENIGN(0):      {tn:4d} (TN)   |   {fp:4d} (FP)")
    print(f"Actual MALIGNANT(1):   {fn:4d} (FN)   |   {tp:4d} (TP)")

    preds_binary_optimal = (test_preds >= optimal_threshold).astype(int)

    acc_opt = accuracy_score(test_labels, preds_binary_optimal)
    rec_opt = recall_score(test_labels, preds_binary_optimal, zero_division=0)
    prec_opt = precision_score(test_labels, preds_binary_optimal, zero_division=0)
    f1_opt = f1_score(test_labels, preds_binary_optimal, zero_division=0)
    cm_opt = confusion_matrix(test_labels, preds_binary_optimal)
    tn_opt, fp_opt, fn_opt, tp_opt = cm_opt.ravel()

    print(F"ENSEMBLE SCORE ON TEST SET (OPTIMAL THRESHOLD = {optimal_threshold})):")
    print(f"AUC:       {test_auc:.4f}")
    print(f"Accuracy:  {acc_opt:.4f} ({acc_opt * 100:.1f}%)")
    print(f"Precision: {prec_opt:.4f} ({prec_opt * 100:.1f}%)")
    print(f"Recall:    {rec_opt:.4f} ({rec_opt * 100:.1f}%)")
    print(f"F1-Score:  {f1_opt:.4f} ({f1_opt * 100:.1f}%)")
    print(f"-" * 50)
    print(f"CONFUSION MATRIX:")
    print(f"                 Predicted: BENIGN(0) | MALIGNANT(1)")
    print(f"Actual BENIGN(0):      {tn_opt:4d} (TN)   |   {fp_opt:4d} (FP)")
    print(f"Actual MALIGNANT(1):   {fn_opt:4d} (FN)   |   {tp_opt:4d} (TP)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    train_and_evaluate()