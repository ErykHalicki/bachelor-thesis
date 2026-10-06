import numpy as np
import pytest
from omegaconf import OmegaConf

from thesis.algorithms import build_algorithm
from thesis.experiments.eval.chunking import ChunkDriver

ROPE = {"time": {"share": 0.75, "period": "auto"},
        "seq": {"share": 0.25, "period": "auto"}}

OBS_FEATURES = {"observation.state": {"dtype": "float32", "shape": (6,),
                                      "names": [f"s{i}" for i in range(6)]}}


def _model(action_index="-5..-1", action_dim=4):
    cfg = OmegaConf.create({
        "name": "generic_vit_predictor",
        "model": {"rope": ROPE, "model_dim": 64, "depth": 2, "num_heads": 4, "mlp_ratio": 2.0,
                  "dropout": 0.0},
        "num_flow_steps": 2,
        "execute_len": 3,
        "conditioning": {
            "state_context": {"from": "observation.state", "dim": 6, "index": "-2..0",
                              "fps": 30},
            "action_context": {"from": "action", "via": "cross", "dim": action_dim,
                               "index": action_index, "fps": 30},
        },
        "predict": {"action": {"from": "action", "type": "flow", "dim": 4, "index": "0..7",
                               "fps": 30}},
    })
    model = build_algorithm(cfg).eval()
    model.norm_stats = {
        "action": {"mean": [10.0] * 4, "std": [2.0] * 4, "min": [-1.0] * 4, "max": [1.0] * 4},
        "observation.state": {"mean": [0.0] * 6, "std": [1.0] * 6, "min": [-1.0] * 6,
                              "max": [1.0] * 6},
    }
    model.norm_method = "mean_std"
    return model


def _driver(model=None):
    return ChunkDriver(model or _model(), OmegaConf.create({}), OBS_FEATURES)


def _run(driver, ticks):
    """Drive `ticks` control steps, returning (emitted actions, past-action window per replan)."""
    seen = []
    original = ChunkDriver._executed_window

    def spy(self, actions):
        seen.append([None if a is None else float(a[0]) for a in actions])
        return original(self, actions)

    ChunkDriver._executed_window = spy
    try:
        emitted = [float(driver.step({"observation.state": np.full(6, t, np.float32)})[0])
                   for t in range(ticks)]
    finally:
        ChunkDriver._executed_window = original
    return emitted, seen


def test_the_action_field_is_not_demanded_of_the_robot():
    driver = _driver()
    # no rig reports the action that produced a frame, so it must not be a buffered column
    assert driver.columns == ["observation.state"]
    assert "action" not in driver.field_to_col
    assert driver.action_offsets == [-5, -4, -3, -2, -1]
    assert driver.action_len == 5


def test_each_replan_sees_the_actions_this_driver_commanded():
    driver = _driver()
    emitted, seen = _run(driver, 10)
    assert seen[0] == [None] * 5
    assert seen[1] == [None, None] + emitted[0:3]
    assert seen[2] == emitted[1:6]
    assert seen[3] == emitted[4:9]


def test_missing_history_front_clamps_onto_the_oldest_action():
    """Training clamps a window running off the start of an episode onto its first row,
    so a rollout must not hand the model a constant-zero history instead."""
    driver = _driver()
    block = driver._executed_window(
        [None] * 3 + [np.full(4, 14.0, np.float32), np.full(4, 16.0, np.float32)]
    )
    assert block.shape == (1, 5, 4)
    oldest = pytest.approx((14.0 - 10.0) / 2.0)
    assert all(block[0, i, 0] == oldest for i in range(4))
    assert block[0, -1, 0] == pytest.approx((16.0 - 10.0) / 2.0)


def test_history_with_nothing_to_clamp_to_pads_with_normalized_zero_not_raw_zero():
    """Before the first action is commanded there is no oldest action, so the rows fall
    back to the dataset's centre -- which is zero AFTER normalization, not raw zero."""
    driver = _driver()
    block = driver._executed_window([None] * 5)
    assert block.shape == (1, 5, 4)
    assert (block[0] == 0).all()


def test_reset_warms_the_history_so_an_episode_never_opens_on_a_gap():
    driver = _driver()
    hold = np.arange(4, dtype=np.float32)
    driver.reset(hold)
    _, seen = _run(driver, 1)
    assert seen[0] == [pytest.approx(float(hold[0]))] * 5


def test_hold_action_copies_absolute_dims_and_zeroes_relative_ones():
    """An action named by the observation is absolute, so holding re-commands the measured
    value; one the observation cannot report is relative and holds at zero."""
    from thesis.experiments.eval.lerobot import LeRobotEval

    obs_features = {"observation.state": {"names": ["j0.pos", "j1.pos", "j0.torq"]}}
    frame = {"observation.state": np.array([11.0, 22.0, 99.0], np.float32)}
    assert LeRobotEval._hold_action(frame, ["j0.pos", "j1.pos"], obs_features).tolist() == [11.0, 22.0]
    assert LeRobotEval._hold_action(frame, ["j0.pos", "x.vel"], obs_features).tolist() == [11.0, 0.0]


def test_relative_hold_action_still_carries_the_latched_dimensions():
    """A relative action space rests at zero, but a latched dimension carries a mode rather
    than a magnitude: RoboCasa records -1 for control-mode and gripper in every episode's
    first row, and the history the policy reads has to match the data, not the controller
    (which thresholds, and so reads 0 and -1 alike)."""
    from thesis.experiments.eval.robocasa import LATCHED_ACTION_DIMS, RoboCasaEval

    class Driver:
        action_context_dim = 12

    hold = RoboCasaEval._hold_action(Driver)
    assert hold.shape == (12,)
    for dim in range(12):
        assert hold[dim] == LATCHED_ACTION_DIMS.get(dim, 0.0)
    assert RoboCasaEval._hold_action(type("D", (), {"action_context_dim": 0})) is None


def test_reset_drops_the_previous_episodes_actions():
    driver = _driver()
    _run(driver, 4)
    assert len(driver._acts) > 0
    driver.reset()
    assert len(driver._acts) == 0
    _, seen = _run(driver, 1)
    assert seen[0] == [None] * 5


def test_a_model_without_action_conditioning_is_unaffected():
    cfg = OmegaConf.create({
        "name": "generic_vit_predictor",
        "model": {"rope": ROPE, "model_dim": 64, "depth": 2, "num_heads": 4, "mlp_ratio": 2.0,
                  "dropout": 0.0},
        "num_flow_steps": 2,
        "conditioning": {"state_context": {"from": "observation.state", "dim": 6,
                                           "index": "-2..0", "fps": 30}},
        "predict": {"action": {"from": "action", "type": "flow", "dim": 4, "index": "0..7",
                               "fps": 30}},
    })
    model = build_algorithm(cfg).eval()
    driver = ChunkDriver(model, OmegaConf.create({}), OBS_FEATURES)
    assert driver.action_context is None
    assert driver.action_offsets == [] and driver.action_len == 0
    assert driver.step({"observation.state": np.zeros(6, np.float32)}).shape == (4,)


def test_conditioning_on_a_non_past_action_is_rejected():
    with pytest.raises(ValueError, match="every offset must be negative"):
        _driver(_model(action_index="-2..0"))


def test_a_history_dim_that_cannot_hold_the_executed_chunk_is_rejected():
    with pytest.raises(ValueError, match="they must match"):
        _driver(_model(action_dim=6))


def _codec_model(enc_dim=4):
    """A latent-action arm: the history is encoded into one token and the chunk decoded back
    out of one, so both streams carry a 64-dim latent over 4-dim actions."""
    cfg = OmegaConf.create({
        "name": "generic_vit_predictor",
        "model": {"rope": ROPE, "model_dim": 64, "depth": 2, "num_heads": 4, "mlp_ratio": 2.0,
                  "dropout": 0.0},
        "num_flow_steps": 2,
        "encoders": {
            "act_enc": {"type": "chunk_mlp", "in_dim": enc_dim, "in_steps": 8, "out_dim": 64,
                        "out_steps": 1, "hidden_dim": 32, "depth": 2, "norm": "layer"},
            "act_dec": {"type": "chunk_mlp", "in_dim": 64, "in_steps": 1, "out_dim": 4,
                        "out_steps": 8, "hidden_dim": 32, "depth": 2, "norm": "layer"},
        },
        "conditioning": {
            "state_context": {"from": "observation.state", "dim": 6, "index": "-2..0",
                              "fps": 30},
            "action_context": {"from": "action", "role": "action", "encoder": "act_enc",
                               "dim": 64, "index": "-1", "raw_index": "-8..-1", "fps": 30,
                               "grid": [1, 1], "block_size": 1},
        },
        "predict": {"action": {"from": "action", "type": "flow", "dim": 64, "index": "0",
                               "raw_index": "0..7", "chunk_len": 8, "encoder": "act_enc",
                               "decoder": "act_dec", "fps": 30, "block_size": 1}},
    })
    model = build_algorithm(cfg).eval()
    model.norm_stats = {
        "action": {"mean": [10.0] * 4, "std": [2.0] * 4, "min": [-1.0] * 4, "max": [1.0] * 4},
        "observation.state": {"mean": [0.0] * 6, "std": [1.0] * 6, "min": [-1.0] * 6,
                              "max": [1.0] * 6},
    }
    model.norm_method = "mean_std"
    return model


def test_a_codec_history_is_buffered_in_raw_actions_not_latents():
    driver = _driver(_codec_model())
    assert driver.action_offsets == [-8, -7, -6, -5, -4, -3, -2, -1]
    # the encoder eats raw actions, so the rows are 4-dim even though the stream is 64-dim
    assert driver.action_context_dim == 4
    assert driver._executed_window([None] * 8).shape == (1, 8, 4)
    assert driver.step({"observation.state": np.zeros(6, np.float32)}).shape == (4,)


def test_a_codec_history_encoding_actions_the_chunk_cannot_fill_is_rejected():
    with pytest.raises(ValueError, match="they must match"):
        _driver(_codec_model(enc_dim=6))
