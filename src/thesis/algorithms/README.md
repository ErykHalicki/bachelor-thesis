# algorithms

The method layer: models, loss/step, and inference API. May import external packages
(torch, encoders, backend libs) but never from `datasets/`, `experiments/`, or `eval/`.

The modality spec and the batch contract it implies are documented in
[docs/.claude/config-spec.md](../../../docs/.claude/config-spec.md). In this layer, the `PredictiveModel`
base is the only code that reads `from`/`encoder`; the `GenericViTTrunk` it feeds sees
only geometry keys and tensors.

There is one predictor (`name: generic_vit_predictor`). Each predicted stream declares
`type: flow | direct`, and `_forward_train` derives its seeding and target from that:
`flow` seeds noise and targets `x1 - x0`; `direct` reads its `input` stream's tokens and
targets the next latent. Adding a paradigm is a config change, not a subclass.


## World models (`name: hjepa`)

`hjepa/` is H-JEPA (external/h-jepa) ported module for module: its `level1..levelN` config
schema, its parameter names (released checkpoints load through
`HJEPAWorldModel.load_reference_checkpoint`, without unpickling their code), and its
numerics (`tests/test_hjepa_parity.py` runs the reference beside it). `num_levels: 1` is flat
LeWorldModel. It predicts no actions (`action_stream = None`), so eval drives it with
`PlanDriver` toward `eval.goal_image`; the planning itself lives in `thesis/planning/`
and is chosen by the `plan:` section. The data windows it needs are documented at the
top of `hjepa/world_model.py` and checked at build.
