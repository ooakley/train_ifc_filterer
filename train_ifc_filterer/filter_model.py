"""Defining model and model trainer for filtering binary classifier."""
from __future__ import annotations
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class FilterDataset(Dataset):

    def __init__(
            self, image_array: np.ndarray, label_array: np.ndarray, device: str = "cpu"
            ) -> None:
        assert image_array.shape[0] == label_array.shape[0]
        self.image_array = image_array
        self.label_array = label_array

    def __len__(self) -> int:
        return self.image_array.shape[0]

    def __getitem__(self, idx: Any) -> tuple[Any, Any]:
        image_selection = torch.from_numpy(np.expand_dims(self.image_array[idx, :, :], axis=0))
        label_selection = torch.from_numpy(np.asarray([self.label_array[idx]]))
        return image_selection, label_selection


class FilterClassifier(nn.Module):
    """Defining structure of classifier network."""

    def __init__(self) -> None:
        super(FilterClassifier, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 20, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 50, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.linear = nn.Sequential(
            nn.Linear(12800, 500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_pool = self.conv(x)
        logit = self.linear(max_pool.view(-1, 12800))
        return logit


class FilterHandler:

    def __init__(self, image_array: np.ndarray, label_array: np.ndarray) -> None:
        self.model = FilterClassifier().double()
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.BCEWithLogitsLoss()
        self.dataset = FilterDataset(image_array, label_array)
        self.loss_history: list[float] = []

    def _train_on_epoch(self) -> None:
        dataloader = DataLoader(self.dataset, batch_size=50, shuffle=True, num_workers=0)
        for data, labels in dataloader:
            logit_out = self.model(data)
            loss = self.criterion(logit_out, labels)
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            self.loss_history.append(loss.item())

    def train(self, num_epochs: int) -> None:
        for _ in range(num_epochs):
            self._train_on_epoch()
