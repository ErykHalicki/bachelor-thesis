import pytest

from thesis.utils.spec import action_entry, parse_index, window_lengths


def test_plain_list_passthrough():
    assert parse_index([0, 1, 2]) == [0, 1, 2]


def test_range_string():
    assert parse_index("0..3") == [0, 1, 2, 3]
    assert parse_index("-12..-10") == [-12, -11, -10]


def test_mixed_string():
    assert parse_index("-8..-6, 0, 2..3") == [-8, -7, -6, 0, 2, 3]


def test_list_with_range_entries():
    assert parse_index(["-2..-1", 4]) == [-2, -1, 4]


def test_window_lengths_parses_ranges():
    spec = {"action": {"index": "0..47"}, "state": {"index": [0]}}
    assert window_lengths(spec) == {"action": 48, "state": 1}


def test_action_entry_falls_back_to_the_action_field():
    """Every config predates the role key, so the old convention still has to resolve."""
    assert action_entry({"action": {"index": "0..47"}}) == ("action", "action")
    spec = {"chunk": {"from": "action", "index": "0..47"}}
    assert action_entry(spec) == ("chunk", "action")


def test_role_names_the_action_stream_whatever_it_is_called():
    """The point of the role: a stream reading a column named anything else is still
    the one eval executes."""
    spec = {"arm_cmd": {"from": "observation.joint_target", "role": "action", "index": "0..7"}}
    assert action_entry(spec) == ("arm_cmd", "observation.joint_target")


def test_role_outranks_the_naming_convention():
    spec = {
        "action": {"index": "0..47"},
        "arm_cmd": {"from": "cmd", "role": "action", "index": "0..7"},
    }
    assert action_entry(spec) == ("arm_cmd", "cmd")


def test_no_action_stream_is_not_an_error():
    """A world model predicts state; its callers decide what to do about that."""
    assert action_entry({"state": {"index": "0..3"}}) is None


def test_two_claimants_is_a_build_time_error():
    spec = {
        "left": {"from": "cmd_l", "role": "action", "index": [0]},
        "right": {"from": "cmd_r", "role": "action", "index": [0]},
    }
    with pytest.raises(ValueError, match="exactly one"):
        action_entry(spec)


def test_misspelled_role_is_rejected():
    """Silently ignoring it would leave the stream unclaimed and break eval later."""
    with pytest.raises(ValueError, match="unknown role"):
        action_entry({"arm_cmd": {"role": "actions", "index": [0]}})