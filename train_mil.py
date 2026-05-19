import os
import glob
import argparse
import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score, confusion_matrix, \
    roc_curve

from mil_models import MILFactory


class SmoothTop1SVM(nn.Module):
    """
    Smooth SVM Loss described in CLAM article.
    Used only in instance-level clusterization.
    """

    def __init__(self, n_classes=2, tau=1.0, alpha=1.0):
        super(SmoothTop1SVM, self).__init__()
        self.tau = tau
        self.alpha = alpha
        self.n_classes = n_classes

    def forward(self, preds, labels):
        # preds: [batch_size, n_classes], labels: [batch_size]
        device = preds.device
        batch_size = preds.shape[0]

        # One-hot mask for correct classes
        y_hot = torch.zeros(batch_size, self.n_classes, device=device)
        y_hot.scatter_(1, labels.unsqueeze(1), 1)

        # Margin is equal to: alpha * 1(j != y)
        margin = self.alpha * (1.0 - y_hot)

        # s_y (predictions for correct classes)
        s_y = torch.gather(preds, 1, labels.unsqueeze(1))

        # (margin + s_j - s_y) / tau
        diff = (margin + preds - s_y) / self.tau

        # Smooth SVM loss
        loss = self.tau * torch.log(torch.sum(torch.exp(diff), dim=1))
        return loss.mean()


class WSIFeatureDataset(Dataset):
    def __init__(self, folders, label_map):
        self.files = []
        self.labels = []
        self.composite_labels = []

        for folder in folders:
            label = next((val for key, val in label_map.items() if key in folder.lower()), None)
            if label is None:
                continue

            stain_type = 'mgg' if 'mgg' in folder.lower() else 'hes'

            pt_files = glob.glob(os.path.join(folder, "*.pt"))
            self.files.extend(pt_files)
            self.labels.extend([label] * len(pt_files))
            self.composite_labels.extend([f"{label}_{stain_type}"] * len(pt_files))

        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in the provided folders: {folders}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        features = torch.load(file_path)
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return features, label, file_path


class WSIClassifierTrainer:
    def __init__(self, model, device, pos_weight=None, lr=1e-4, bag_weight=0.7, inst_weight=0.3):
        self.model = model.to(device)
        self.device = device

        self.is_clam = hasattr(self.model, 'k_sample')
        self.bag_weight = bag_weight
        self.inst_weight = inst_weight

        if self.is_clam:
            # For CLAM using BCE on bag-level
            # and Smooth SVM on instance-level
            if pos_weight is not None:
                pw_tensor = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device)
                self.bag_loss_fn = nn.CrossEntropyLoss(weight=pw_tensor)
            else:
                self.bag_loss_fn = nn.CrossEntropyLoss()

            self.inst_loss_fn = SmoothTop1SVM(n_classes=2)

        else:
            # For Gated/TransMIL using BCEWithLogits
            if pos_weight is not None:
                pw_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
                self.bag_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
            else:
                self.bag_loss_fn = nn.BCEWithLogitsLoss()

        # # Added weights for imbalanced classes
        # if pos_weight is not None:
        #     pw_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
        #     self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
        # else:
        #     self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        for features, label, _ in loader:
            features, label = features.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()

            if self.is_clam:
                label_long = label.long().view(-1)
                logits, _, instance_dict = self.model(features, label=label_long, instance_eval=True)
                loss_bag = self.bag_loss_fn(logits, label_long)

                true_class = int(label_long.item())

                # Because the classes are NOT mutually exclusive
                # Omit out-of-the-class branches
                preds = instance_dict[true_class]
                top_logits = preds['top_preds']
                bottom_logits = preds['bottom_preds']

                # Creating pseudo-labels
                top_labels = torch.ones(top_logits.shape[0], dtype=torch.long, device=self.device)
                bottom_labels = torch.zeros(bottom_logits.shape[0], dtype=torch.long, device=self.device)

                all_inst_logits = torch.cat([top_logits, bottom_logits], dim=0)
                all_inst_labels = torch.cat([top_labels, bottom_labels], dim=0)

                loss_inst = self.inst_loss_fn(all_inst_logits, all_inst_labels)

                # Summing losses (L_total = c1 * L_bag + c2 * L_inst)
                loss = (self.bag_weight * loss_bag) + (self.inst_weight * loss_inst)
            else:
                logits, _ = self.model(features)
                loss = self.bag_loss_fn(logits, label)
            # logits, _ = self.model(features)
            # loss = self.criterion(logits, label)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for features, label, _ in loader:
                features, label = features.to(self.device), label.to(self.device)

                if self.is_clam:
                    label_long = label.long().view(-1)
                    # During evaluation instance-level clusterization is off
                    logits, _ = self.model(features, instance_eval=False)
                    loss = self.bag_loss_fn(logits, label_long)

                    probs = torch.softmax(logits, dim=1)[0, 1].unsqueeze(0)
                else:
                    logits, _ = self.model(features)
                    loss = self.bag_loss_fn(logits, label)
                    probs = torch.sigmoid(logits).view(-1)

                # logits, _ = self.model(features)
                # loss = self.criterion(logits, label)


                total_loss += loss.item()

                # probs = torch.sigmoid(logits)
                all_labels.extend(label.cpu().numpy().flatten())
                all_probs.extend(probs.cpu().numpy().flatten())

        avg_loss = total_loss / len(loader)

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5

        return avg_loss, auc, all_labels, all_probs


def get_data_folders(stain_type, extractor_name, base_dir):
    if stain_type == 'mgg':
        return [os.path.join(base_dir, "benignMGG", extractor_name),
                os.path.join(base_dir, "malignantMGG", extractor_name)]
    elif stain_type == 'hes':
        return [os.path.join(base_dir, "benignHES", extractor_name),
                os.path.join(base_dir, "malignantHES", extractor_name)]
    elif stain_type == 'both':
        return [
            os.path.join(base_dir, "benignMGG", extractor_name),
            os.path.join(base_dir, "malignantMGG", extractor_name),
            os.path.join(base_dir, "benignHES", extractor_name),
            os.path.join(base_dir, "malignantHES", extractor_name)
        ]
    else:
        print("Unknown stain type: options for stains are: 'mgg', 'hes' and 'both'!")
        exit()


def main():
    parser = argparse.ArgumentParser(description="MIL Training and Evaluation")
    parser.add_argument('--stain', choices=['mgg', 'hes', 'both'], required=True, help="Dataset type")
    parser.add_argument('--extractor', choices=['resnet50', 'ctranspath', 'uni', 'phikon'], required=True)
    parser.add_argument('--mil_model', choices=['gated_attention', 'transmil', 'clam'], required=True)
    parser.add_argument('--base_dir', type=str, default="/lustre/pd03/hpc-ljelen-1692966897/WSI_features")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=None,
                        help="Force specific dropout rate. If None, reads from JSON.")

    parser.add_argument('--clam_b', type=int, default=8, help="Hyperparameter B (k_sample) for CLAM clustering")
    parser.add_argument('--clam_c1', type=float, default=0.7,
                        help="Bag loss weight (c1) for CLAM. Instance weight (c2) will be 1 - c1")
    args = parser.parse_args()

    if args.dropout is not None:
        final_dropout = args.dropout
    else:
        try:
            with open('best_dropouts.json', 'r') as f:
                best_drops = json.load(f)
            final_dropout = best_drops[args.stain][args.extractor][args.mil_model]
        except (FileNotFoundError, KeyError):
            final_dropout = 0.25 if args.mil_model == 'clam' else 0.05
            print(f"[!] No JSON config found. Using default dropout: {final_dropout}")

    inst_weight = 1.0 - args.clam_c1

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Beginning training on: {DEVICE}")
    print(
        f"Configuration: Stain={args.stain.upper()} | Extractor={args.extractor.upper()} | MIL={args.mil_model.upper()}")

    # Feature dimensions for feature extractors
    DIM_MAP = {'uni': 1024, 'phikon': 1024, 'resnet50': 2048, 'ctranspath': 768}
    INPUT_DIM = DIM_MAP[args.extractor]

    # DROPOUT_RATES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    LABEL_MAP = {'benign': 0, 'malignant': 1}

    models_save_dir = f"saved_models/{args.stain}_{args.extractor}_{args.mil_model}"
    os.makedirs(models_save_dir, exist_ok=True)

    folders = get_data_folders(args.stain, args.extractor, args.base_dir)
    full_dataset = WSIFeatureDataset(folders, LABEL_MAP)

    labels = np.array(full_dataset.labels)
    composite_labels = np.array(full_dataset.composite_labels)
    indices = np.arange(len(labels))
    patient_groups = []

    for file_path in full_dataset.files:
        file_name = os.path.basename(file_path).replace('.pt', '')

        if file_name.startswith("Image_"):
            patient_id = file_name.split('_')[1]
        else:
            patient_id = file_name.split('_')[0]

        patient_groups.append(patient_id)

    patient_groups = np.array(patient_groups)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tv_idx, test_idx = next(gss.split(indices, labels, groups=patient_groups))

    tv_labels = labels[tv_idx]
    tv_comp_labels = composite_labels[tv_idx]
    tv_groups = patient_groups[tv_idx]

    test_sub = Subset(full_dataset, test_idx)
    test_loader = DataLoader(test_sub, batch_size=1, shuffle=False)

    tv_sub = Subset(full_dataset, tv_idx)
    tv_loader = DataLoader(tv_sub, batch_size=1, shuffle=False)

    unique_tv_patients = len(np.unique(tv_groups))
    unique_test_patients = len(np.unique(patient_groups[test_idx]))

    print(f"\nData split (Patient-Level Isolation):")
    print(f"-> Train + Val set (80%): {unique_tv_patients} unique patients ({len(tv_idx)} slides)")
    print(f"-> Test set (20%):        {unique_test_patients} unique patients ({len(test_idx)} slides)")

    print(f"\n{'=' * 60}")
    print("DETAILED DATA SPLIT - TEST SET")
    print(f"{'=' * 60}")

    test_info_benign = []
    test_info_malignant = []

    for idx in test_idx:
        file_path = full_dataset.files[idx]
        label = full_dataset.labels[idx]

        file_name = os.path.basename(file_path).replace('.pt', '')
        if file_name.startswith("Image_"):
            patient_id = file_name.split('_')[1]
        else:
            patient_id = file_name.split('_')[0]

        stain_type = "MGG" if "mgg" in file_path.lower() else "HES" if "hes" in file_path.lower() else "UNKNOWN"

        info_str = f" -> ID: {patient_id:25s} | Slide: {file_name:25s} | Stain: {stain_type}"

        if label == 0:
            test_info_benign.append(info_str)
        else:
            test_info_malignant.append(info_str)

    print(f"MALIGNANT ({len(test_info_malignant)} slides):")
    for info in sorted(test_info_malignant):
        print(info)

    print(f"\nBENIGN ({len(test_info_benign)} slides):")
    for info in sorted(test_info_benign):
        print(info)

    mgg_test_count = sum(1 for info in test_info_benign + test_info_malignant if 'MGG' in info)
    hes_test_count = sum(1 for info in test_info_benign + test_info_malignant if 'HES' in info)

    print(f"\nTest Set Stain Balance: MGG = {mgg_test_count}, HES = {hes_test_count}")
    print(f"{'=' * 60}\n")

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # best_overall_drop_rate = None
    # best_overall_val_auc = 0.0
    # saved_ensemble_paths_for_best_drop = []
    #
    # for drop_rate in DROPOUT_RATES:
    #     print(f"\n{'#' * 60}")
    #     print(f"TESTING DROPOUT RATE = {drop_rate}")
    #     print(f"{'#' * 60}")

    fold_metrics = []
    current_drop_model_paths = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(tv_idx, tv_comp_labels, groups=tv_groups)):
        print(f"\n{'=' * 10} FOLD {fold + 1}/5 (Dropout: {final_dropout}) {'=' * 10}")

        fold_train_labels = tv_labels[train_idx]
        num_neg_fold = np.sum(fold_train_labels == 0)
        num_pos_fold = np.sum(fold_train_labels == 1)

        fold_pos_weight = num_neg_fold / num_pos_fold if num_pos_fold > 0 else 1.0

        train_sub = Subset(full_dataset, tv_idx[train_idx])
        val_sub = Subset(full_dataset, tv_idx[val_idx])

        train_loader = DataLoader(train_sub, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=1, shuffle=False)

        model = MILFactory.create(
            args.mil_model,
            input_dim=INPUT_DIM,
            n_classes=1,
            dropout_rate=final_dropout,
            k_sample=args.clam_b
        )

        trainer = WSIClassifierTrainer(
            model,
            DEVICE,
            pos_weight=fold_pos_weight,
            lr=args.lr,
            bag_weight=args.clam_c1,
            inst_weight=inst_weight
        )

        best_auc = 0.0
        best_val_loss = float('inf')
        epochs_no_improve = 0

        fold_model_path = os.path.join(models_save_dir, f"best_fold_{fold + 1}_drop_{final_dropout}.pth")

        for epoch in range(args.epochs):
            train_loss = trainer.train_epoch(train_loader)
            val_loss, val_auc, val_labels_arr, val_probs_arr = trainer.evaluate(val_loader)

            val_preds_binary = (np.array(val_probs_arr) > 0.5).astype(int)
            acc = accuracy_score(val_labels_arr, val_preds_binary)

            print(
                f"Epoch {epoch + 1:02d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | ACC: {acc:.4f}")

            if val_auc > best_auc or (val_auc == best_auc and val_loss < best_val_loss):
                best_auc = val_auc
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(trainer.model.state_dict(), fold_model_path)
            else:
                epochs_no_improve += 1
                print(f"   -> No better AUC result since {epochs_no_improve} epochs.")

            if epochs_no_improve >= args.patience:
                print(f"EARLY STOPPING: Training in fold {fold + 1} interrupted in epoch {epoch + 1}!")
                break

        fold_metrics.append(best_auc)
        current_drop_model_paths.append(fold_model_path)
        print(f"-> The best AUC in fold {fold + 1}: {best_auc:.4f}")

    mean_val_auc = np.mean(fold_metrics)
    print(f"\n-> Mean AUC from 5 folds (Dropout={final_dropout}): {mean_val_auc:.4f} (+/- {np.std(fold_metrics):.4f})")

        # if mean_val_auc > best_overall_val_auc:
        #     best_overall_val_auc = mean_val_auc
        #     best_overall_drop_rate = drop_rate
        #     saved_ensemble_paths_for_best_drop = current_drop_model_paths

    # print(f"\n{'=' * 50}")
    # print(f"THE BEST DROPOUT RATE: {best_overall_drop_rate} (Mean Val AUC: {best_overall_val_auc:.4f})")
    # print(f"{'=' * 50}")

    ensemble_models = []
    for path in current_drop_model_paths:
        m = MILFactory.create(
            args.mil_model,
            input_dim=INPUT_DIM,
            n_classes=1,
            dropout_rate=final_dropout,
            k_sample=args.clam_b
        )
        m.load_state_dict(torch.load(path))
        m.to(DEVICE)
        m.eval()
        ensemble_models.append(m)

    print(f"\n{'*' * 40}")
    print("TUNING DECISION THRESHOLD ON TRAIN/VAL SET (80%)")
    print(f"{'*' * 40}")

    is_clam_model = args.mil_model.lower() == 'clam'

    tv_eval_labels = []
    tv_eval_preds = []
    with torch.no_grad():
        for features, label, _ in tv_loader:
            features = features.to(DEVICE)

            patient_probs = []
            for m in ensemble_models:
                if is_clam_model:
                    logits, _ = m(features, instance_eval=False)
                    probs = torch.softmax(logits, dim=1)[0, 1].item()
                else:
                    logits, _ = m(features)
                    probs = torch.sigmoid(logits).item()
                patient_probs.append(probs)

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
        for features, label, _ in test_loader:
            features = features.to(DEVICE)

            patient_probs = []
            for m in ensemble_models:
                if is_clam_model:
                    logits, _ = m(features, instance_eval=False)
                    probs = torch.softmax(logits, dim=1)[0, 1].item()
                else:
                    logits, _ = m(features)
                    probs = torch.sigmoid(logits).item()
                patient_probs.append(probs)

            avg_prob = np.mean(patient_probs)
            test_labels.append(label.item())
            test_preds.append(avg_prob)

    test_labels = np.array(test_labels)
    test_preds = np.array(test_preds)

    test_preds_binary_default = (test_preds > 0.5).astype(int)
    test_preds_binary_optimal = (test_preds >= optimal_threshold).astype(int)

    def print_metrics(labels, binary_preds, probs, threshold):
        auc = roc_auc_score(labels, probs)
        acc = accuracy_score(labels, binary_preds)
        rec = recall_score(labels, binary_preds, zero_division=0)
        prec = precision_score(labels, binary_preds, zero_division=0)
        f1 = f1_score(labels, binary_preds, zero_division=0)
        cm = confusion_matrix(labels, binary_preds)
        tn, fp, fn, tp = cm.ravel()

        print(f"ENSEMBLE SCORE ON TEST SET (THRESHOLD = {threshold:.4f}):")
        print(f"AUC:       {auc:.4f}")
        print(f"Accuracy:  {acc:.4f} ({acc * 100:.1f}%)")
        print(f"Precision: {prec:.4f} ({prec * 100:.1f}%)")
        print(f"Recall:    {rec:.4f} ({rec * 100:.1f}%)")
        print(f"F1-Score:  {f1:.4f} ({f1 * 100:.1f}%)")
        print(f"-" * 50)
        print(f"CONFUSION MATRIX:")
        print(f"                 Predicted: BENIGN(0) | MALIGNANT(1)")
        print(f"Actual BENIGN(0):      {tn:4d} (TN)   |   {fp:4d} (FP)")
        print(f"Actual MALIGNANT(1):   {fn:4d} (FN)   |   {tp:4d} (TP)")
        print(f"{'=' * 50}\n")

    print_metrics(test_labels, test_preds_binary_default, test_preds, 0.5)
    print_metrics(test_labels, test_preds_binary_optimal, test_preds, optimal_threshold)


if __name__ == "__main__":
    main()