import torch
import torch.nn.functional as F

from ...datasets import build_dataset
from .base import EvalResult


class DummyEval:
    """Held-out MSE on freshly sampled synthetic data. Stands in for a real rollout /
    world.evaluate() so the train -> checkpoint -> eval path is exercised end to end.

    The held-out set is defined entirely by the eval config: a matching weight_seed means
    the same task as training, a different seed means unseen observations.
    """

    def __init__(self, cfg):
        self.dataset = build_dataset(cfg, augment=False)
        self.n_eval = cfg.get("n_eval", 256)

    def run(self, model):
        device = next(model.parameters()).device
        n = min(len(self.dataset), self.n_eval)
        obs = torch.stack([self.dataset[i]["observation"] for i in range(n)]).to(device)
        target = torch.stack([self.dataset[i]["target"] for i in range(n)]).to(device)
        pred = model.predict(obs)
        return EvalResult(metrics={"mse": F.mse_loss(pred, target).item()})
