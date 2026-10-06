import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAlgorithm


class DummyAlgorithm(BaseAlgorithm):
    """Trivial MLP regressor. Placeholder that keeps the loop green end to end.

    Replace with the real method, keeping the loss/predict signatures.
    """

    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, cfg.target_dim),
        )

    def forward(self, obs):
        return self.net(obs)

    def loss(self, batch):
        pred = self(batch["observation"])
        return {"loss": F.mse_loss(pred, batch["target"])}

    def predict(self, obs):
        self.eval()
        with torch.no_grad():
            return self(obs)

    def summary(self):
        return {**super().summary(), "hidden_dim": self.net[0].out_features}
