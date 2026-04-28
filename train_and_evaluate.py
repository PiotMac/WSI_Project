import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score, confusion_matrix
import numpy as np


from abmil_model import WSIFeatureDataset, AttentionMIL

CSV_FILE = "/home/piomac5688/projekt_wsi/dataset_abmil.csv"
EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    print(f"\nData split:")
    print(f"-> Train set + validation set (80%): {len(tv_idx)} patients")
    print(f"-> Test set (20%): {len(test_idx)} patients")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_metrics = []
    global_best_auc = 0.0
    best_model_path = "models/ens_tie/best_global_abmil_model.pth"

    saved_model_paths = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(tv_idx, tv_labels)):
        print(f"\n{'=' * 10} FOLD {fold + 1}/5 {'=' * 10}")

        train_sub = tv_idx[train_idx]
        val_sub = tv_idx[val_idx]

        train_sub = Subset(full_dataset, train_sub)
        val_sub = Subset(full_dataset, val_sub)

        train_loader = DataLoader(train_sub, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=1, shuffle=False)

        model = AttentionMIL().to(DEVICE)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

        best_auc = 0.0
        best_val_loss = float('inf')
        epochs_no_improve = 0

        fold_model_path = f"models/ens_tie/best_model_fold_{fold + 1}.pth"

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
            rec = recall_score(val_labels, val_preds_binary, zero_division=0)
            prec = precision_score(val_labels, val_preds_binary, zero_division=0)
            f1 = f1_score(val_labels, val_preds_binary, zero_division=0)

            print(
                f"Epoch {epoch + 1:02d}/{EPOCHS} | Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {val_loss / len(val_loader):.4f} | Val AUC: {auc:.4f} | Val ACC: {acc:.4f} | Val REC: {rec:.4f} | Val PREC: {prec:.4f} | Val F1: {f1:.4f}")

            if auc > best_auc or (auc == best_auc and val_loss < best_val_loss):
                best_auc = auc
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), fold_model_path)

                if auc > global_best_auc:
                    global_best_auc = auc
                    torch.save(model.state_dict(), best_model_path)
            else:
                epochs_no_improve += 1
                print(f"   -> No better AUC result since {epochs_no_improve} epochs.")

            if epochs_no_improve >= PATIENCE:
                print(f"EARLY STOPPING: Training in fold {fold + 1} interrupted in epoch {epoch + 1}!")
                break

        fold_metrics.append(best_auc)
        saved_model_paths.append(fold_model_path)
        print(f"-> The best AUC in fold {fold + 1}: {best_auc:.4f}")

    print(f"\n{'=' * 30}")
    print(f"Training has finished!")
    print(f"Mean AUC from 5 folds: {np.mean(fold_metrics):.4f} (+/- {np.std(fold_metrics):.4f})")
    print(f"{'=' * 30}")

    print(f"\n{'*' * 40}")
    print("EVALUATING THE MODEL")
    print(f"{'*' * 40}")

    ensemble_models = []
    for path in saved_model_paths:
        m = AttentionMIL().to(DEVICE)
        m.load_state_dict(torch.load(path))
        m.eval()
        ensemble_models.append(m)

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

    # final_model = AttentionMIL().to(DEVICE)
    # final_model.load_state_dict(torch.load(best_model_path))
    # final_model.eval()
    #
    # test_labels, test_preds = [], []
    # with torch.no_grad():
    #     for features, label in test_loader:
    #         features, label = features.to(DEVICE), label.to(DEVICE)
    #         y_prob, _ = final_model(features)
    #
    #         test_labels.append(label.item())
    #         test_preds.append(y_prob.item())

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

    # print("MODEL'S SCORE ON TEST SET:")
    print("ENSEMBLE SCORE ON TEST SET:")
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

if __name__ == "__main__":
    train_and_evaluate()