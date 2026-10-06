import torch

from .base import BaseSource


class DummySource(BaseSource):
    """Synthetic linear-regression data. No backend, no download, runs in milliseconds.
    The gate every change passes before spending real compute.
    """

    provided_modalities = {"observation", "target"}

    def __init__(self, cfg):
        n = cfg.get("n", 1024)
        obs_dim = cfg.get("obs_dim", 16)
        target_dim = cfg.get("target_dim", 4)
        # weight_seed fixes the task (the linear map); seed only draws observations, so
        # a held-out split shares the same map
        gw = torch.Generator().manual_seed(cfg.get("weight_seed", 0))
        gx = torch.Generator().manual_seed(cfg.get("seed", 0))
        weight = torch.randn(obs_dim, target_dim, generator=gw)
        self.obs = torch.randn(n, obs_dim, generator=gx)
        self.target = self.obs @ weight

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return {"observation": self.obs[idx], "target": self.target[idx]}
