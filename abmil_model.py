import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd


class WSIFeatureDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        features = torch.load(row['path'])
        label = torch.tensor(row['label'], dtype=torch.float32)

        return features, label


class AttentionMIL(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, dropout_rate=0.05):
        super(AttentionMIL, self).__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)
        A = self.attention(x)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)

        M = torch.mm(A, x)

        Y_prob = self.classifier(M)

        return Y_prob, A