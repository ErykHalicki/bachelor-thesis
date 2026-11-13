# Fine-tuning the 8-task pretrains: UWM-frozen vs attnpool

Both arms were pretrained on the 8-task OCBench visual mix (`configs/dataset/ocbench_multi8_visual.yaml`).
Each pretrain was then fine-tuned for 25k steps on four single tasks. Runs finished 2026-10-20/21.

| Arm | Pretrain config | Checkpoint | Fine-tuned on |
|---|---|---|---|
| UWM-frozen | `run=pretrain8_uwm_frozen` | `uwm_frozen_qbkz8azv_v21/model.pt` | eryk-pc, Slurm jobs 69-72, half a V100 each (MPS shard) |
| attnpool | `run=pretrain8_attnpool` | `mxavewkg/model.pt` | Sockeye, jobs 13603007/13603008, two runs per V100 under MPS (`scripts/sockeye/train_pair.sbatch`) |

Fine-tune recipe, the same for both arms:
- `+experiment.init_from=<ckpt> experiment.max_steps=25000 experiment.eval_every=12500`
- effective batch 256
- each task's own data: `dataset=ocbench_block_single_visual_official` for block-single, `ocbench_multi8/<task>` for the others

On Sockeye, `init_from` loaded every module (encoders, pools, BNs, predictor) and kept nothing at its fresh init.

Each run is evaluated three times, each on 200 episodes with EMA weights. The 12.5k and 25k evals happen during training. The third is the `validation` task after training, which scores the same final weights as the 25k eval.

## Results (success rate, %)

Values are printed rounded to whole percents. Half-points come from the exact "new best" lines.

| Task | Arm | 12.5k | 25k | validation (final weights) | best |
|---|---|---|---|---|---|
| block-single-task1 | UWM-frozen | 77.5 | 48 | 51 | **77.5** |
| | attnpool | 76 | 76 | 75 | 76 |
| hanoi-single-task1 | UWM-frozen | 74 | 58 | 57 | 74 |
| | attnpool | 65 | 72 | 63 | 72 |
| switch-3x3-task1 | UWM-frozen | 5 | 7 | 8 | 7 |
| | attnpool | 25.5 | 28.5 | 29 | **28.5** |
| block-double-task1 | UWM-frozen | 31 | 52.5 | 48 | 52.5 |
| | attnpool | 52.5 | 62 | 58 | **62** |
| **mean over 4 tasks** | UWM-frozen | 46.9 | 41.4 | 41.0 | 52.8 |
| | attnpool | 54.8 | 59.6 | 56.3 | 59.6 |

## Takeaways

- **attnpool is the stronger pretrain.** It leads the 4-task mean by about 15-18 points on the final weights and by about 7 points on the best checkpoint. Its largest margins are on the two harder tasks: switch-3x3 (28.5 vs 7) and block-double (62 vs 52.5).
- **UWM-frozen degrades on the single-object tasks.** It peaks at 12.5k, then falls by about 25 points on block-single (77.5 → 48/51) and about 16 on hanoi (74 → 58/57). attnpool holds steady or improves over the same span. Only best-checkpoint selection makes UWM-frozen look competitive there, and that selection uses the eval set itself.
- **UWM-frozen still improves on block-double** (31 → 52.5), so the degradation does not appear on every task. On switch-3x3 it barely learns anything (5-8%).
- **Eval noise:** the 25k eval and the validation pass score identical weights, yet they differ by up to 9 points (attnpool hanoi: 72 vs 63) and typically by 1-5. At n=200 the binomial standard error alone is about 3.5 points. Treat single-task differences under about 8 points as unresolved.

## Caveats

- The two arms ran on different hardware: eryk-pc MPS shard vs Sockeye MPS pair, both V100. Only wall-clock should differ; the recipe and evals are identical.
- There is one seed per cell.
- The 12.5k/25k evals pick the best checkpoint on the same episodes they report, so the "best" column is optimistic. Use the validation column for an unbiased comparison.

## Where things live

- eryk-pc logs: `~/ocb_bench/abl/{ft1_uwm_frozen,ft_uwm_frozen_<task>}.log`; W&B project `halicki/thesis`
- Sockeye logs: `/scratch/st-dushan20-1/ehalicki/logs/ft-attnpool-pair{1,2}-1360300{7,8}.out`. These runs logged W&B offline, so they need a `wandb sync` to appear in the project.
