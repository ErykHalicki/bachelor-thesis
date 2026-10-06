# configs

Hydra config groups. Everything runnable is one file in `configs/run/`, which selects the
dataset, algorithm and eval together — so one name is a whole run.

```bash
source .venv/bin/activate
python main.py run=flow_wam_b601_pnpt
```

`base.yaml` is Hydra's default config and the entry point, so `run=` is the whole command
line. It owns the run-level keys — `debug`, `wandb`, `resume`/`load`, `posthoc_prefix` —
and any of them can be overridden on the CLI or by a run config.

New variants are new yaml files, not code branches.

## The groups

| group | holds |
| --- | --- |
| `run/` | one file per runnable arm; the only thing `run=` accepts |
| `algorithm/` | the model: its modality spec, encoders and losses |
| `dataset/` | which repo, which episodes, which columns |
| `eval/` | how a trained model is scored |
| `experiment/` | training-loop knobs (batch size, steps, checkpoint cadence) |
| `embodiment/` | the physical rig an on-robot eval drives |

`algorithm/vision/` is a group nested inside `algorithm/`: it holds the camera head,
chosen with `- override vision: <option>` at the END of an algorithm's defaults list.

## Sweeps

A `configs/sweep_*.yaml` lists its arms in `hydra.sweeper.params.run`.

```bash
python main.py run=<arm>                                    # one arm of it
```

A sweep config must also name one arm with `- override /run:`, because the test suite
composes every top-level config with no `run=` and a config that trains but selects no
algorithm is an error. Multirun arms share one process, so a crash in the third takes the
rest with it; loop over `run=` in the shell when you want process isolation.

## Reference

- **[docs/.claude/config-spec.md](../docs/.claude/config-spec.md)** — every key of the modality spec:
  `conditioning`/`predict` entries, encoders, losses, A2A sources, augmentation, probes.
