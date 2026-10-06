"""Held-out reconstruction: the decoder's loss curve, plus the pictures it exists for."""

import math

from .offline import OfflineEval


def panel(columns, gap=2):
    """One (H, W, 3) uint8 image: a row per held-out sample, the named columns side by side.

    `columns` maps a label to a uint8 (B, steps, 3, H, W) stack -- the real frame, its
    decoded latent, and (for a world-model probe) the decoded prediction. They sit next to
    each other because every question here is a comparison between two of them: what the
    encoder dropped is column 1 against column 2, what the predictor got wrong is column 2
    against column 3.

    Returns (image, caption); the caption names the column order, which no pixel can.
    """
    import torch

    stacks = list(columns.values())
    b, steps = stacks[0].shape[:2]
    height, width = stacks[0].shape[-2:]
    white = torch.full((3, height, gap), 255, dtype=stacks[0].dtype, device=stacks[0].device)

    rows = []
    for i in range(b):
        cells = []
        for t in range(steps):
            for stack in stacks:
                cells += [stack[i, t], white]
        rows.append(torch.cat(cells[:-1], dim=-1))
    row_gap = torch.full(
        (3, gap, rows[0].shape[-1]), 255, dtype=stacks[0].dtype, device=stacks[0].device
    )
    stacked = [x for row in rows for x in (row, row_gap)][:-1]
    image = torch.cat(stacked, dim=-2).permute(1, 2, 0).cpu().numpy()

    order = " | ".join(columns)
    caption = f"{order}" + (f", per step (x{steps}); {width}x{height}" if steps > 1 else "")
    return image, caption


class ReconstructionEval(OfflineEval):
    """The offline held-out loss, plus a reconstruction panel per decoded stream.

    The metrics are OfflineEval's, so `eval/loss` stays comparable with `train/loss`, with
    a PSNR per stream derived from it: the decoder's loss terms are a plain per-pixel MSE
    in [0, 1] units (see algorithms/latent_decoder.py), which makes -10*log10(mse) the
    honest dB figure and saves a second pass over the split to compute it.

    The panel is drawn from a fixed, seeded set of held-out samples and cached, and the
    decode itself runs under the same seed -- a world-model probe integrates its prediction
    from noise, so without that the sequence of panels would mix a sharpening decoder with
    a different draw each time. Same frames, same noise, one thing changing.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.panel_rows = int(cfg.get("panel_rows", 6))
        self.ar_windows = int(cfg.get("ar_windows", 0))
        self._panel_batch = None
        self._ar_batches = None

    def _samples(self, dataset):
        import torch
        from torch.utils.data import default_collate

        if self._panel_batch is None:
            generator = torch.Generator().manual_seed(self.seed)
            picked = torch.randperm(len(dataset), generator=generator)[: self.panel_rows]
            self._panel_batch = default_collate([dataset[int(i)] for i in picked])
        return self._panel_batch

    def run(self, model):
        import torch

        result = super().run(model)
        for key, value in list(result.metrics.items()):
            if key.startswith("loss/") and value > 0:
                result.metrics[f"psnr/{key.removeprefix('loss/')}"] = -10 * math.log10(value)

        # a world-model probe also scores the decoded ROLLOUT against the real future
        # frames, over the same split and seed as the loss pass -- the decoder metrics
        # above bound what the latent keeps, these bound what the predictor gets right
        if hasattr(model, "prediction_errors"):
            from torch.utils.data import DataLoader

            loader = DataLoader(
                self._build(model),
                batch_size=self.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(self.seed),
                num_workers=self.num_workers,
            )
            device = next(model.parameters()).device
            devices = [device] if device.type == "cuda" else []
            totals, seen, batches = {}, 0, 0
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.seed)
                for batch in loader:
                    if self.max_batches and batches >= int(self.max_batches):
                        break
                    batch = {
                        k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in batch.items()
                    }
                    errors = model.prediction_errors(batch)
                    n = next(len(v) for v in batch.values() if torch.is_tensor(v))
                    for name, mse in errors.items():
                        totals[name] = totals.get(name, 0.0) + mse.item() * n
                    seen += n
                    batches += 1
            for name, total in totals.items():
                mse = total / seen
                result.metrics[f"loss/pred_{name}"] = mse
                if mse > 0:
                    result.metrics[f"psnr/pred_{name}"] = -10 * math.log10(mse)

        if not hasattr(model, "reconstruct"):
            raise ValueError(
                f"the reconstruction eval renders what a model decodes, which "
                f"{type(model).__name__} does not do; it needs a `reconstruct(batch)` "
                f"(algorithm=latent_decoder or wam_decoder), or use eval=holdout for the "
                f"loss alone"
            )
        device = next(model.parameters()).device
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in self._samples(self._build(model)).items()
        }
        devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed)
            decoded = model.reconstruct(batch)
        for name, columns in decoded.items():
            result.images[name] = panel(columns)

        # the same panel, autoregressively over consecutive windows: window k's context
        # comes from window k-1's predictions, so the columns show compounding error
        if self.ar_windows > 1 and hasattr(model, "reconstruct_ar"):
            batches = [
                {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
                for b in self._ar_samples(self._build(model), model)
            ]
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.seed)
                decoded = model.reconstruct_ar(batches)
            for name, columns in decoded.items():
                result.images[f"{name}_ar"] = panel(columns)
        return result

    def _ar_samples(self, dataset, model):
        """`ar_windows` aligned batches, each one prediction-horizon later: rows are
        held-out moments whose whole 3-window future exists inside one episode."""
        import torch
        from torch.utils.data import default_collate

        if self._ar_batches is not None:
            return self._ar_batches
        src = dataset
        while not hasattr(src, "_index") and hasattr(src, "dataset"):
            src = src.dataset
        by_frame = {entry[0]: i for i, entry in enumerate(src._index)}
        episode = {entry[0]: entry[2] for entry in src._index}
        stride = model.ar_window_stride()
        frames = sorted(by_frame)
        good = [f for f in frames
                if all(f + k * stride in by_frame
                       and episode[f + k * stride] == episode[f]
                       for k in range(self.ar_windows))]
        generator = torch.Generator().manual_seed(self.seed)
        picked = [good[int(i)] for i in
                  torch.randperm(len(good), generator=generator)[: self.panel_rows]]
        self._ar_batches = [
            default_collate([dataset[by_frame[f + k * stride]] for f in picked])
            for k in range(self.ar_windows)
        ]
        return self._ar_batches
