"""One small 1D convolutional model shared by all selected product–store series."""

import torch
from torch import nn

from m5_forecast.data import CALENDAR_FEATURES, HISTORY, HORIZON


class DemandCNN(nn.Module):
    """Combine recent sales patterns, a series embedding, and future calendar dates."""

    def __init__(
        self, num_series: int, seasonal_residual: bool = False, reconcile_alpha: float = 0
    ):
        super().__init__()
        features = len(CALENDAR_FEATURES)
        self.seasonal_residual = seasonal_residual
        self.reconcile_alpha = reconcile_alpha
        self.encoder = nn.Sequential(
            nn.Conv1d(1 + features, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.embedding = nn.Embedding(num_series, 8)
        self.head = nn.Sequential(
            nn.Linear(16 * HISTORY + 8 + HORIZON * features, 64),
            nn.ReLU(),
            nn.Linear(64, HORIZON),
            nn.Identity() if seasonal_residual else nn.Softplus(),
        )
        if seasonal_residual:
            nn.init.zeros_(self.head[2].weight)
            nn.init.zeros_(self.head[2].bias)

    def forward(self, history, future_calendar, series):
        """Return seven nonnegative sales estimates in training-normalized units."""
        inputs = torch.cat(
            [self.encoder(history), self.embedding(series), future_calendar.flatten(1)], dim=1
        )
        correction = self.head(inputs)
        if self.seasonal_residual:
            weekly_pattern = history[:, 0, -28:].reshape(-1, 4, HORIZON).mean(dim=1)
            return weekly_pattern.clamp_min(0.01) * torch.exp(correction.clamp(-3, 3))
        return correction
