"""resume_map/load_map: the per-run fallbacks behind multirun resume. A multirun shares
every override across its arms, so the maps are passed once and each arm picks its own
entry -- keyed by its wandb name or by the `run=` arm it was launched with. The scalar
keys always win over the maps."""

from omegaconf import OmegaConf

from main import resolve_run_refs


def cfg(**kwargs):
    return OmegaConf.create(kwargs)


def test_scalars_pass_through():
    resume, load = resolve_run_refs(cfg(name="a", resume="rid", load="/x.pt"))
    assert (resume, load) == ("rid", "/x.pt")


def test_maps_resolve_by_name():
    c = cfg(name="a", resume_map={"a": "rid_a", "b": "rid_b"},
            load_map={"a": "/a.pt", "b": "/b.pt"})
    assert resolve_run_refs(c) == ("rid_a", "/a.pt")


def test_missing_entry_means_fresh():
    c = cfg(name="c", resume_map={"a": "rid_a"}, load_map={"a": "/a.pt"})
    assert resolve_run_refs(c) == (None, None)


def test_scalar_wins_over_map():
    c = cfg(name="a", resume="explicit", resume_map={"a": "rid_a"})
    assert resolve_run_refs(c) == ("explicit", None)


def test_no_name_no_maps():
    assert resolve_run_refs(cfg(resume_map={"a": "rid_a"})) == (None, None)
    assert resolve_run_refs(cfg(name="a")) == (None, None)


def test_maps_resolve_by_arm_when_name_is_overridden():
    """A jobspec that relabels the run with `name=` still keys its map by the arm, because
    that is the only name its `run=` sweep line carries. Matching `name` alone resolved to
    None and the job trained from scratch with no error."""
    c = cfg(name="pnpt_act", resume_map={"act_b601_pnpt": "licb7s5e"},
            load_map={"act_b601_pnpt": "/tmp/ck_licb7s5e.pt"})
    assert resolve_run_refs(c, "act_b601_pnpt") == ("licb7s5e", "/tmp/ck_licb7s5e.pt")


def test_name_wins_over_arm():
    c = cfg(name="a", resume_map={"a": "rid_a", "arm": "rid_arm"})
    assert resolve_run_refs(c, "arm") == ("rid_a", None)


def test_arm_miss_still_means_fresh():
    c = cfg(name="a", resume_map={"b": "rid_b"})
    assert resolve_run_refs(c, "arm") == (None, None)
