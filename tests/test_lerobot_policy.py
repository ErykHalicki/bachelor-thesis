"""The lerobot-policy wrapper, exercised against a stub policy.

The translation between this repo's windowed batch and the batch a lerobot policy expects
is the whole wrapper, and getting it wrong is a shape error several frames inside lerobot
rather than a message. These pin it down without building a real policy or touching a
dataset.
"""

import pytest
import torch
from omegaconf import OmegaConf

from thesis.algorithms.lerobot_policy import LeRobotPolicy

STATE = "observation.state"
IMAGE = "observation.images.image"


class StubConfig:
    type = "stub"

    def __init__(self, n_obs_steps=1, inputs=(STATE, IMAGE)):
        self.n_obs_steps = n_obs_steps
        self.input_features = dict.fromkeys(inputs)


class StubPolicy(torch.nn.Module):
    """Records the batch it was handed, so a test can assert on the translation."""

    def __init__(self, config, chunk=4, action_dim=7):
        super().__init__()
        self.config = config
        self.chunk = chunk
        self.action_dim = action_dim
        self.linear = torch.nn.Linear(4, 4)
        self.seen = None

    def forward(self, batch):
        self.seen = batch
        return torch.tensor(1.5, requires_grad=True), {"l1": torch.tensor(0.25)}

    def predict_action_chunk(self, batch):
        self.seen = batch
        size = next(iter(batch.values())).shape[0]
        return torch.zeros(size, self.chunk, self.action_dim)

    def get_optim_params(self):
        return [{"params": list(self.parameters()), "lr": 0.5}]

    def reset(self):
        pass


def build(monkeypatch, n_obs_steps=1, index="0", chunk="0..3", inputs=(STATE, IMAGE)):
    cfg = OmegaConf.create({
        "name": "lerobot_policy",
        "dataset": {"repo_id": "stub/dataset", "root": None, "revision": None},
        "policy": {"type": "stub"},
        "execute_len": 2,
        "conditioning": {
            STATE: {"index": index, "fps": 10},
            IMAGE: {"index": index, "fps": 10},
        },
        "predict": {
            "action": {"from": "action", "role": "action", "dim": 7,
                       "index": chunk, "fps": 10},
        },
    })
    policy = StubPolicy(StubConfig(n_obs_steps=n_obs_steps, inputs=inputs))
    monkeypatch.setattr(LeRobotPolicy, "_build_policy", lambda self, _cfg: policy)
    return LeRobotPolicy(cfg), policy


def batch(steps=1, size=2):
    return {
        STATE: torch.zeros(size, steps, 8),
        IMAGE: torch.full((size, steps, 3, 4, 4), 255, dtype=torch.uint8),
        "action": torch.zeros(size, 4, 7),
    }


def test_single_step_windows_lose_their_time_axis(monkeypatch):
    """lerobot's own dataloader hands a policy an unwindowed observation when it asked
    for one step, and its policies are written against that shape."""
    model, policy = build(monkeypatch)
    model.loss(batch(steps=1))
    assert policy.seen[STATE].shape == (2, 8)
    assert policy.seen[IMAGE].shape == (2, 3, 4, 4)


def test_multi_step_windows_keep_their_time_axis(monkeypatch):
    model, policy = build(monkeypatch, n_obs_steps=2, index="-1..0")
    model.loss(batch(steps=2))
    assert policy.seen[STATE].shape == (2, 2, 8)
    assert policy.seen[IMAGE].shape == (2, 2, 3, 4, 4)


def test_images_arrive_as_float_in_0_1(monkeypatch):
    """The dataset and the rollout driver both serve uint8; lerobot policies normalize
    from [0, 1] floats, and feeding them 0-255 is a silent 255x input scale."""
    model, policy = build(monkeypatch)
    model.loss(batch())
    image = policy.seen[IMAGE]
    assert image.dtype == torch.float32
    assert float(image.max()) == pytest.approx(1.0)


def test_actions_carry_an_all_false_pad_mask(monkeypatch):
    """ACT's VAE encoder reads `action_is_pad` unconditionally. `drop_boundary` means no
    chunk here is ever padded, so the mask is all false -- but it has to be present."""
    model, policy = build(monkeypatch)
    model.loss(batch())
    mask = policy.seen["action_is_pad"]
    assert mask.shape == (2, 4) and mask.dtype == torch.bool
    assert not mask.any()


def test_an_existing_pad_mask_is_passed_through(monkeypatch):
    model, policy = build(monkeypatch)
    supplied = torch.ones(2, 4, dtype=torch.bool)
    model.loss({**batch(), "action_is_pad": supplied})
    assert torch.equal(policy.seen["action_is_pad"], supplied)


def test_loss_reports_the_scalar_and_its_components(monkeypatch):
    model, _ = build(monkeypatch)
    out = model.loss(batch())
    assert out["loss"].requires_grad
    assert out["loss"].item() == pytest.approx(1.5)
    assert float(out["loss/l1"]) == pytest.approx(0.25)


def test_predict_returns_the_action_stream_the_driver_executes(monkeypatch):
    model, _ = build(monkeypatch)
    out = model.predict({k: v for k, v in batch().items() if k != "action"})
    assert set(out) == {"action"}
    assert out["action"].shape == (2, 4, 7)


def test_predict_is_given_no_action_field(monkeypatch):
    """A rollout has no future actions to hand over; leaking the training batch's would
    make an eval score a model that saw its own answer."""
    model, policy = build(monkeypatch)
    model.predict({k: v for k, v in batch().items() if k != "action"})
    assert "action" not in policy.seen


def test_driver_facing_attributes_come_from_the_spec(monkeypatch):
    model, _ = build(monkeypatch, n_obs_steps=2, index="-1..0")
    assert model.action_stream == "action"
    assert model.action_field == "action"
    assert model.chunk_len == 4
    assert model.execute_len == 2
    assert model.obs_len == 2
    assert model.field_offsets == {STATE: [-1, 0], IMAGE: [-1, 0]}


def test_normalization_is_left_to_the_policy(monkeypatch):
    """The policy carries Normalize layers built from the dataset's statistics. Stats
    here would normalize a second time, in the dataset wrapper and the rollout driver."""
    model, _ = build(monkeypatch)
    assert model.norm_stats is None
    assert model.norm_method is None


def test_optim_params_are_the_policys_own_groups(monkeypatch):
    """ACT gives its vision backbone a separate learning rate; dropping the groups would
    train the baseline with a recipe lerobot never used."""
    model, _ = build(monkeypatch)
    groups = model.optim_params()
    assert isinstance(groups, list) and groups[0]["lr"] == 0.5


def test_a_missing_input_is_caught_at_build_time(monkeypatch):
    with pytest.raises(ValueError, match="which no `conditioning:` entry provides"):
        build(monkeypatch, inputs=(STATE, IMAGE, "observation.images.image2"))


def test_an_unread_input_is_caught_at_build_time(monkeypatch):
    with pytest.raises(ValueError, match="does not read"):
        build(monkeypatch, inputs=(STATE,))


def test_a_window_the_policy_cannot_take_is_caught_at_build_time(monkeypatch):
    with pytest.raises(ValueError, match="n_obs_steps"):
        build(monkeypatch, n_obs_steps=1, index="-1..0")


def test_execute_len_must_fit_the_chunk(monkeypatch):
    with pytest.raises(ValueError, match="outside the"):
        build(monkeypatch, chunk="0..0")


class FakeTokenizer:
    padding_side = "right"

    def __init__(self):
        self.calls = []

    def __call__(self, prompts, padding, truncation, max_length, return_tensors):
        self.calls.append(list(prompts))
        n = len(prompts)
        return {"input_ids": torch.arange(n * 4).reshape(n, 4),
                "attention_mask": torch.ones(n, 4, dtype=torch.long)}


def _lang_build(monkeypatch, state_in_prompt):
    tok = FakeTokenizer()
    lang = {"tokenizer": tok, "padding": "max_length", "max_length": 8,
            "state_in_prompt": state_in_prompt}
    monkeypatch.setattr(LeRobotPolicy, "_build_language", lambda self: lang)
    model, policy = build(monkeypatch)
    return model, tok


def test_smolvla_language_gets_newline_and_tokens(monkeypatch):
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    model, tok = _lang_build(monkeypatch, state_in_prompt=False)
    b = batch()
    b["task"] = ["pick the cube", "stack it\n"]
    out = model._to_policy_batch(b)
    assert tok.calls == [["pick the cube\n", "stack it\n"]]
    assert out[OBS_LANGUAGE_TOKENS].shape == (2, 4)
    assert out[OBS_LANGUAGE_ATTENTION_MASK].dtype == torch.bool


def test_pi05_prompt_discretizes_the_normalized_state(monkeypatch):
    from thesis.algorithms.lerobot_policy import pi05_prompts

    state = torch.tensor([[-1.0, 0.0, 1.0]])
    (prompt,) = pi05_prompts(["Wipe_the\ntable"], state, max_state_dim=5)
    # -1 -> bin 0, 0 -> mid bin, +1 (clamped) -> top bin, zero padding -> mid bin
    assert prompt.startswith("Task: Wipe the table, State: 0 128 255 128 128;")
    assert prompt.endswith("\nAction: ")


def test_language_policy_without_task_refuses(monkeypatch):
    model, _ = _lang_build(monkeypatch, state_in_prompt=False)
    with pytest.raises(ValueError, match="task"):
        model._to_policy_batch(batch())
