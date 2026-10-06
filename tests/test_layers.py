import torch

from thesis.algorithms.layers import (FinalLayer, InputLayer, MLPFinalLayer,
                                     TimestepEmbedder)

DIM = 32


def test_input_layer_projects_and_conditions():
    layer = InputLayer(in_dim=6, hidden_size=DIM)
    x = torch.randn(2, 5, 6)
    cond_a = torch.randn(2, DIM)
    cond_b = torch.randn(2, DIM)
    out = layer(x, cond_a)
    assert out.shape == (2, 5, DIM)
    assert not torch.allclose(out, layer(x, cond_b))


def test_final_layer_zero_at_init():
    layer = FinalLayer(hidden_size=DIM, out_dim=6)
    out = layer(torch.randn(2, 5, DIM), torch.randn(2, DIM))
    assert out.shape == (2, 5, 6)
    assert torch.equal(out, torch.zeros(2, 5, 6))


def test_mlp_final_layer_zero_at_init_and_trainable():
    """DreamZero's decoder departs from theirs here: an A2A arm starts with x0 near x1, so a
    nonzero velocity at init pushes the endpoint off an answer that is already nearly right."""
    layer = MLPFinalLayer(hidden_size=DIM, out_dim=6, mlp_dim=4)
    x = torch.randn(2, 5, DIM)
    assert torch.equal(layer(x), torch.zeros(2, 5, 6))

    opt = torch.optim.SGD(layer.parameters(), lr=1.0)
    torch.nn.functional.mse_loss(layer(x), torch.ones(2, 5, 6)).backward()
    opt.step()
    assert not torch.equal(layer(x), torch.zeros(2, 5, 6))


def test_sinusoidal_features_resolve_the_unit_interval():
    """Nearly every channel must actually move across t in [0, 1].

    DiT's max_period=10000 assumes t is an integer step in [0, 1000]; fed a t in [0, 1] it
    makes every period longer than the range, so most channels sit near-constant and the
    timestep signal all but vanishes. Pinned because that failure is silent -- shapes,
    grads and training all look fine.
    """
    feats = TimestepEmbedder.timestep_embedding(torch.linspace(0, 1, 101), 512)
    swing = feats.max(0).values - feats.min(0).values
    assert (swing > 0.1).float().mean() > 0.95
    assert swing.mean() > 1.0
