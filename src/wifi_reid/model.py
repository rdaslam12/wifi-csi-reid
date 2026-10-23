"""Siamese CSI re-identification model with subcarrier/packet attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubcarrierAttention(nn.Module):
    def __init__(self, n_subcarriers: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, n_subcarriers // reduction)
        self.fc = nn.Sequential(
            nn.Linear(n_subcarriers, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_subcarriers),
            nn.Sigmoid(),
        )
        self.last_weights = None

    def forward(self, x):
        weights = self.fc(x.mean(dim=(1, 2)))
        self.last_weights = weights.detach()
        return x * weights.unsqueeze(1).unsqueeze(1)


class PacketAttention(nn.Module):
    def __init__(self, n_packets: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, n_packets // reduction)
        self.fc = nn.Sequential(
            nn.Linear(n_packets, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_packets),
            nn.Sigmoid(),
        )
        self.last_weights = None

    def forward(self, x):
        weights = self.fc(x.mean(dim=(1, 3)))
        self.last_weights = weights.detach()
        return x * weights.unsqueeze(1).unsqueeze(-1)


class DualAxisAttention(nn.Module):
    def __init__(self, n_packets: int, n_subcarriers: int, reduction: int = 4, dual=True):
        super().__init__()
        self.subcarrier = SubcarrierAttention(n_subcarriers, reduction)
        self.packet = PacketAttention(n_packets, reduction) if dual else None

    def forward(self, x):
        if self.packet is None:
            return self.subcarrier(x)
        return 0.5 * (self.subcarrier(x) + self.packet(x))


class ConvTrunk(nn.Module):
    def __init__(self, channels_in: int, c1: int = 32, c2: int = 64):
        super().__init__()

        def block(channels_a, channels_b):
            return nn.Sequential(
                nn.Conv2d(channels_a, channels_b, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )

        self.block1 = block(channels_in, c1)
        self.block2 = block(c1, c2)

    def forward(self, x):
        return self.block2(self.block1(x))


class FlattenSignatureHead(nn.Module):
    def __init__(self, channels, packets, subcarriers, hidden=512, embed=128, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * packets * subcarriers, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed),
        )

    def forward(self, x):
        return self.net(x)


class PooledSignatureHead(nn.Module):
    """Lower-capacity exploratory head from the refinement notebook."""

    def __init__(self, channels, packets, subcarriers, pool=(4, 4), hidden=512, embed=128, dropout=0.5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * pool[0] * pool[1], hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed),
        )

    def forward(self, x):
        return self.net(self.pool(x))


class SubCarrierReID(nn.Module):
    """Shared-weight Siamese encoder for Wi-Fi CSI windows."""

    def __init__(
        self,
        input_shape=(1, 100, 52),
        n_classes: int = 81,
        embed: int = 128,
        dual_axis: bool = True,
        head: str = "flatten",
    ):
        super().__init__()
        self.trunk = ConvTrunk(input_shape[0])
        with torch.no_grad():
            features = self.trunk(torch.zeros(1, *input_shape))
        _, channels, packets, subcarriers = features.shape
        self.attention = DualAxisAttention(packets, subcarriers, dual=dual_axis)
        head_cls = PooledSignatureHead if head == "pooled" else FlattenSignatureHead
        self.signature = head_cls(channels, packets, subcarriers, embed=embed)
        self.identity_head = nn.Linear(embed, n_classes)

    def embed(self, x):
        return self.signature(self.attention(self.trunk(x)))

    def forward(self, x_i, x_j):
        z_i, z_j = self.embed(x_i), self.embed(x_j)
        return z_i, z_j, self.identity_head(z_i), self.identity_head(z_j)


class JointReIDLoss(nn.Module):
    """Pairwise cosine-margin loss plus identity classification loss on both branches."""

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
        self.ce = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j, logits_i, logits_j, class_i, class_j, same):
        cosine = F.cosine_similarity(z_i, z_j, dim=1)
        metric_loss = (
            same * (1 - cosine) + (1 - same) * torch.clamp(cosine - self.margin, min=0)
        ).mean()
        total = metric_loss + self.ce(logits_i, class_i) + self.ce(logits_j, class_j)
        return total, metric_loss.detach()
