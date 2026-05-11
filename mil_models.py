import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import math
from abc import ABC, abstractmethod
from nystrom_attention import NystromAttention


class BaseMIL(nn.Module, ABC):
    """
    Abstract base class for all Multiple Instance Learning models.
    """
    def __init__(self):
        super(BaseMIL, self).__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Feature tensors widh dimensions wymiarze [Batch, N_patches, Feature_dim]
                              or [N_patches, Feature_dim].
        Returns:
            tuple:
                - logits (torch.Tensor): Raw predictions for BCEWithLogitsLoss.
                - attention_weights (torch.Tensor): Attention weights for each patch [Batch, N_patches].
        """
        pass


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


class GatedAttentionMIL(BaseMIL):
    def __init__(self, input_dim=768, hidden_dim=256, n_classes=1, dropout_rate=0.05):
        super(GatedAttentionMIL, self).__init__()

        self.attention_v = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_u = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)

        # self.attention = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.Tanh(),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(hidden_dim, 1)
        # )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.squeeze(0)

        attention_v = self.attention_v(x)
        attention_u = self.attention_u(x)

        attention = self.attention_weights(attention_v * attention_u)

        # attention = self.attention(x)
        attention = torch.transpose(attention, 1, 0)
        attention = F.softmax(attention, dim=1)

        M = torch.mm(attention, x)

        y_prob = self.classifier(M)

        return y_prob, attention


class PPEG(nn.Module):
    """Pyramid Position Encoding Generator for TransMIL."""

    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj_3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.proj_5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.proj_7x7 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        cls_token, feat_tokens = x[:, 0:1, :], x[:, 1:, :]
        cnn_feat = feat_tokens.transpose(1, 2).view(B, C, H, W)
        x_peg = self.proj_3x3(cnn_feat) + self.proj_5x5(cnn_feat) + self.proj_7x7(cnn_feat) + cnn_feat
        x_peg = x_peg.flatten(2).transpose(1, 2)
        x_out = torch.cat((cls_token, x_peg), dim=1)
        return x_out


class TransMIL(BaseMIL):
    def __init__(self, input_dim=1024, embed_dim=512, n_classes=1, dropout_rate=0.1):
        super(TransMIL, self).__init__()

        self.fc1 = nn.Linear(input_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.layer1_attention = NystromAttention(
            dim=embed_dim, dim_head=embed_dim // 8, heads=8,
            num_landmarks=256, pinv_iterations=6, residual=True, dropout=dropout_rate
        )
        self.layer1_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim)
        )
        self.layer1_norm1 = nn.LayerNorm(embed_dim)
        self.layer1_norm2 = nn.LayerNorm(embed_dim)

        self.ppeg = PPEG(dim=embed_dim)

        self.layer2_attention = NystromAttention(
            dim=embed_dim, dim_head=embed_dim // 8, heads=8,
            num_landmarks=256, pinv_iterations=6, residual=True, dropout=dropout_rate,
            return_attn=True
        )
        self.layer2_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim)
        )
        self.layer2_norm1 = nn.LayerNorm(embed_dim)
        self.layer2_norm2 = nn.LayerNorm(embed_dim)

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # Wymuszenie wymiaru Batch

        B, N, D = x.shape
        h = self.fc1(x)

        length = math.ceil(math.sqrt(N))
        pad_size = length * length - N

        if pad_size > 0:
            zeros = torch.zeros((B, pad_size, h.shape[2]), device=h.device)
            h = torch.cat([h, zeros], dim=1)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        # Warstwa 1
        h_attn = self.layer1_attention(self.layer1_norm1(h))
        h = h + h_attn
        h = h + self.layer1_ffn(self.layer1_norm2(h))

        # PPEG
        h = self.ppeg(h, length, length)

        # Warstwa 2 z wyciągnięciem uwagi
        attn_out, attention_matrix = self.layer2_attention(self.layer2_norm1(h))
        h = h + attn_out
        h = h + self.layer2_ffn(self.layer2_norm2(h))

        h_cls = self.norm(h[:, 0])
        logits = self.classifier(h_cls)

        # Przygotowanie mapy uwagi dla patchów
        cls_attention = attention_matrix.mean(dim=1)
        wsi_attention = cls_attention[:, 0, 1:]

        if pad_size > 0:
            wsi_attention = wsi_attention[:, :-pad_size]

        return logits, wsi_attention



class MILFactory:
    """Factory class for dynamic MIL models instantiation."""

    @staticmethod
    def create(model_name: str, input_dim: int, n_classes: int = 1) -> BaseMIL:
        name = model_name.lower()
        if name == 'gated_attention':
            print("Initialisation: Gated Attention MIL")
            return GatedAttentionMIL(input_dim=input_dim, n_classes=n_classes)
        elif name == 'transmil':
            print("Initialisation: TransMIL (Nystromformer)")
            return TransMIL(input_dim=input_dim, n_classes=n_classes)
        else:
            raise ValueError(f"Unknown MIL model: {model_name}. Models available: 'gated_attention', 'transmil'.")