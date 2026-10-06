"""The B601 gravity model, checked against physics rather than against itself.

gravity_torque() propagates moments down the chain; potential_energy() sums
m*g*h over the same links. The two share only the forward kinematics, so
agreeing to finite-difference precision means the moment arms are right. The
remaining tests pin down conventions that are easy to get silently backwards:
sign, gravity direction, and which joints a payload can possibly load.
"""

import numpy as np
import pytest

pytest.importorskip("lerobot")

gm = pytest.importorskip("lerobot.robots.rebot_b601_follower.gravity_model")

RNG = np.random.default_rng(0)
POSES = RNG.uniform(-np.pi, np.pi, size=(64, 6))


def numeric_gradient(q, **kwargs):
    """dU/dq by central differences."""
    eps = 1e-6
    grad = np.zeros(gm.NUM_JOINTS)
    for i in range(gm.NUM_JOINTS):
        step = np.zeros(gm.NUM_JOINTS)
        step[i] = eps
        grad[i] = (
            gm.potential_energy(q + step, **kwargs)
            - gm.potential_energy(q - step, **kwargs)
        ) / (2 * eps)
    return grad


@pytest.mark.parametrize("q", POSES)
def test_torque_is_the_potential_energy_gradient(q):
    np.testing.assert_allclose(gm.gravity_torque(q), numeric_gradient(q), atol=1e-6)


@pytest.mark.parametrize("q", POSES[:8])
def test_payload_also_matches_the_gradient(q):
    payload = dict(payload=1.5, payload_com=(0.03, 0.0, 0.06))
    np.testing.assert_allclose(
        gm.gravity_torque(q, **payload), numeric_gradient(q, **payload), atol=1e-6
    )


@pytest.mark.parametrize("q", POSES[:8])
def test_zero_gravity_means_zero_torque(q):
    torque = gm.gravity_torque(q, gravity=np.zeros(3))
    np.testing.assert_allclose(torque, np.zeros(gm.NUM_JOINTS), atol=1e-12)


@pytest.mark.parametrize("q", POSES[:8])
def test_flipping_gravity_flips_the_torque(q):
    np.testing.assert_allclose(
        gm.gravity_torque(q, gravity=-gm.GRAVITY), -gm.gravity_torque(q), atol=1e-12
    )


@pytest.mark.parametrize("q", POSES[:8])
def test_first_joint_is_vertical_so_gravity_never_loads_it(q):
    # joint1 spins about world z at any pose, so no mass distribution can produce a
    # moment about it; a nonzero value means the base transform or axis convention drifted
    assert abs(gm.gravity_torque(q)[0]) < 1e-12


def test_torque_scales_linearly_with_payload():
    q = POSES[3]
    base = gm.gravity_torque(q)
    one = gm.gravity_torque(q, payload=1.0) - base
    three = gm.gravity_torque(q, payload=3.0) - base
    np.testing.assert_allclose(three, 3.0 * one, rtol=1e-12, atol=1e-12)


def test_payload_cannot_load_the_wrist_roll_axis_it_sits_on():
    # the payload defaults to sitting on the wrist_roll axis, so it adds no moment
    # about that last joint itself
    q = POSES[5]
    delta = gm.gravity_torque(q, payload=2.0) - gm.gravity_torque(q)
    assert abs(delta[-1]) < 1e-12
    assert np.abs(delta[1:4]).max() > 0.1


def test_holding_torque_signs_match_a_hand_worked_pose():
    # arm folded so link2 reaches out horizontally: both torques are negative in
    # this URDF's sign convention
    q = np.zeros(gm.NUM_JOINTS)
    tau = gm.gravity_torque(q)
    assert tau[1] < 0 and tau[2] < 0
    assert np.all(np.abs(tau) < gm.EFFORT_LIMITS)


def test_gravity_torque_rejects_wrong_length_configurations():
    with pytest.raises(ValueError):
        gm.gravity_torque(np.zeros(gm.NUM_JOINTS + 1))


def test_link_table_is_a_chain_in_joint_order():
    assert gm.NUM_JOINTS == len(gm.JOINT_NAMES)
    for i, link in enumerate(gm.LINKS):
        assert link.parent < i, "parents must be resolved before their children"
    revolute = [link for link in gm.LINKS if link.axis is not None]
    assert len(revolute) == gm.NUM_JOINTS
    assert [link.name for link in gm.LINKS[: gm.NUM_JOINTS]] == [
        link.name for link in revolute
    ]
