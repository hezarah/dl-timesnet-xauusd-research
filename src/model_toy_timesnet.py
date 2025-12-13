from __future__ import annotations
import torch
import torch.nn as nn

class ToyTimesNet(nn.Module):
    """A lightweight sequence model inspired by temporal mixing ideas.
    NOTE: This is NOT the original TimesNet; it is a public-safe educational architecture.

    Input:  (B, L) close-price window (normalized)
    Output: logits for 3 classes
    """
    def __init__(self, d_model: int = 64, num_layers: int = 3, dropout: float = 0.1, num_classes: int = 3):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        layers = []
        for _ in range(num_layers):
            layers += [
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        self.temporal = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)   # (B, L, 1)
        x = self.proj(x)      # (B, L, d)
        x = x.transpose(1, 2) # (B, d, L)
        x = self.temporal(x)  # (B, d, L)
        return self.head(x)   # (B, C)
