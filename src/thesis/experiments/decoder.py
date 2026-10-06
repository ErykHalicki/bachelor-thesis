from pathlib import Path

from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict

from .base import BaseExperiment
from .eval.base import EvalMixin
from .training.base import TrainingMixin

# the spec parts belonging to the model being probed, not to the probe: a decoder
# run declares its own decoders, `decode:` links and losses
INHERITED = ("model", "encoders", "conditioning", "predict", "num_flow_steps", "cfg_scale",
             "cfg_drop", "execute_len")


class DecoderExperiment(TrainingMixin, EvalMixin, BaseExperiment):
    """The generic train-then-evaluate schedule plus `encoder_init`: the run this decoder
    is a probe of, as a wandb run id (or a local .pt).

    That key is what makes a decoder a probe of a PARTICULAR model rather than of a
    pretrained backbone, and it brings across both halves of that model:

      config   the run's STORED algorithm section (`model`, `encoders`, the spec) replaces
               what this config composes, so the probe cannot drift from what actually
               trained -- the same guarantee post-hoc eval gives (see main.py). The probe's
               own keys win: its decoders are `encoders:` entries the run never had, and
               `decode:`/`losses:` describe the probe alone. A run id is required for this;
               a local .pt carries no config, so there the composed one stands.
      weights  every parameter the two share, matched by name (encoders AND trunk), plus
               the normalization stats those weights were trained under.

    Both stay optional: with no `encoder_init` the config builds as written, which for a
    bare pretrained backbone is already the right thing.

    Resuming ignores the whole mechanism: the checkpoint being resumed carries its own copy
    of every weight, and re-seeding from another run would overwrite them.
    """

    def _build_algo(self):
        if self.algo is None:
            # before the base builds from root_cfg.algorithm -- and before the dataset, which
            # interpolates the spec to decide which columns and rows to load
            self._inherit_algorithm()
        fresh = self.algo is None
        algo = super()._build_algo()
        if fresh:
            self._init_encoders(algo)
        return algo

    def _source_run(self):
        """`<entity>/<project>/<run>` for the probed run, or None when there is none to
        name (no `encoder_init`, a local checkpoint, or a resume that supersedes it).
        """
        from ..utils.ckpt_utils import is_run_id, split_alias

        ref = self.cfg.get("encoder_init")
        if not ref or self.ckpt_path is not None:
            return None
        run_id, _ = split_alias(ref)
        if not is_run_id(run_id):
            return None
        wandb_cfg = self.root_cfg.wandb
        return f"{wandb_cfg.entity}/{wandb_cfg.project}/{run_id}"

    def _inherit_algorithm(self):
        from ..utils.ckpt_utils import fetch_run_config
        from ..utils.wandb import update_run_config

        run_path = self._source_run()
        if run_path is None:
            return
        try:
            stored = fetch_run_config(run_path).algorithm
        except Exception as err:  # noqa: BLE001 - wandb raises many flavors
            print(f"WARNING: could not fetch the stored config of {run_path} ({err}); "
                  f"probing the locally composed algorithm config instead")
            return

        local = self.root_cfg.algorithm
        # the run's version of each inherited key wins key by key: a probe's decoders sit
        # in the same `encoders:` map as the run's and must survive it, so those merge
        merged = {k: OmegaConf.merge(local[k], stored[k]) if k in local else stored[k]
                  for k in INHERITED if k in stored}
        with open_dict(self.root_cfg):
            self.root_cfg.algorithm = OmegaConf.merge(local, merged)
        print(f"encoder_init {run_path}: probing that run's stored "
              f"{', '.join(k for k in INHERITED if k in stored)}")
        update_run_config({"algorithm": OmegaConf.to_container(self.root_cfg.algorithm,
                                                              resolve=True)})

    def _init_encoders(self, algo):
        from ..utils.checkpoint import load_pretrained_weights
        from ..utils.ckpt_utils import download_latest_checkpoint, split_alias

        ref = self.cfg.get("encoder_init")
        if not ref:
            return
        if self.ckpt_path is not None:
            print(f"resuming from {self.ckpt_path}: ignoring encoder_init={ref}")
            return

        run_path = self._source_run()
        if run_path is None:
            path = str(ref)
        else:
            # its own download dir: main.py has already resolved a checkpoint into
            # outputs/downloaded, and each download wipes the dir it writes to
            path = download_latest_checkpoint(run_path, Path("outputs/downloaded-encoder-init"),
                                              alias=split_alias(ref)[1])
        loaded, kept = load_pretrained_weights(algo, path)
        print(f"encoder_init {ref}: loaded {', '.join(loaded)}"
              + (f"; kept the init of {', '.join(kept)}" if kept else ""))
