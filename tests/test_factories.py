from omegaconf import OmegaConf

from thesis.experiments import build_experiment


def test_build_experiment_returns_registered_class():
    cfg = OmegaConf.create({"experiment": {"_name": "smoke"}})
    exp = build_experiment(cfg, output_dir="runs")
    assert exp.__class__.__name__ == "SmokeExperiment"


def test_every_experiment_yaml_is_registered():
    # a yaml in configs/experiment/ missing from exp_registry fails only at runtime
    # on the cluster, so pin the mapping here
    from pathlib import Path

    from thesis.experiments import exp_registry

    yamls = {p.stem for p in (Path(__file__).parent.parent / "configs" / "experiment").glob("*.yaml")}
    missing = yamls - set(exp_registry)
    assert not missing, f"experiment yamls without a registry entry: {sorted(missing)}"
