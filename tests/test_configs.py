"""Every shipped top-level config composes, and the values that must agree across groups
actually do. Composition only -- no models or datasets are built."""

import pathlib
from collections.abc import Mapping

import pytest
from hydra import compose, initialize_config_dir

from thesis.utils.spec import action_entry, parse_index

CONFIG_DIR = pathlib.Path(__file__).resolve().parents[1] / "configs"
# everything runnable is a member of `run/`, composed through `base`; the other
# top-level files are scaffolding
SCAFFOLD = ("base",)
TOP_LEVEL = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml") if p.stem not in SCAFFOLD)
RUNS = sorted(p.stem for p in (CONFIG_DIR / "run").glob("*.yaml"))


def _compose(param):
    """Compose one config the way `main.py` would.

    `base` derives `name` from the chosen run through the `hydra:` resolver, which reads a
    HydraConfig the runtime sets and plain composition does not -- so this sets one from the
    composed `hydra` node, then drops that node to leave an ordinary run config.
    """
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import open_dict

    name, run = param
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name=name,
            overrides=[f"run={run}"] if run else [],
            return_hydra_config=True,
        )
    HydraConfig.instance().set_config(cfg)
    with open_dict(cfg):
        del cfg["hydra"]
    return cfg


@pytest.fixture(
    params=[(n, None) for n in TOP_LEVEL] + [("base", r) for r in RUNS],
    ids=[*TOP_LEVEL, *RUNS],
)
def cfg(request):
    return _compose(request.param)


def _algorithm(cfg):
    """The algorithm section, or a skip for the one config shape that has none.

    `run/b601_irl` names no model and no dataset on purpose: it IS the eval, and what is
    being evaluated comes from the run `load=` points at, whose stored config supplies
    both. Nothing about a model can be asserted at composition time because there is no
    model until load time.
    """
    if "algorithm" not in cfg:
        pytest.skip("eval-only config: its algorithm is restored from `load=`")
    return cfg.algorithm


def test_config_composes(cfg):
    assert cfg.experiment.tasks
    if "algorithm" not in cfg:
        # only an eval-only run may arrive without a model; a TRAINING run missing its
        # algorithm group would compose happily here and die on the card
        assert "training" not in cfg.experiment.tasks, (
            "a run with no algorithm group can only be an eval that restores one from "
            f"`load=`, but this one trains: tasks={list(cfg.experiment.tasks)}"
        )
        return
    assert cfg.algorithm.name


def test_execute_len_fits_the_predicted_chunk(cfg):
    """`index` and `execute_len` are two statements of the chunk length: shrinking the
    chunk without the other is an assertion at build time, long after the run starts.
    """
    algorithm = _algorithm(cfg)
    execute_len = algorithm.get("execute_len")
    predict = algorithm.get("predict")
    if execute_len is None or not isinstance(predict, Mapping):
        return
    action = action_entry(predict)
    if action is None:
        return
    spec = algorithm.predict[action[0]]
    # a codec-latent action stream counts latent steps in `index`; `chunk_len`
    # restates the executed-chunk length
    chunk_len = int(spec.get("chunk_len") or len(parse_index(spec["index"])))
    assert 1 <= execute_len <= chunk_len, (
        f"execute_len {execute_len} outside the {chunk_len}-step chunk of '{action[0]}'"
    )


def test_lerobot_policies_get_their_normalization_from_the_dataset(cfg):
    """A lerobot policy has no Normalize layers of its own -- lerobot keeps them in the
    processor pipelines, which this repo's wrapper does not build. So the data layer is the
    only thing normalizing these arms, and turning it off there leaves the run training in
    raw units: silent in the loss, and fatal for diffusion, whose sampler clips every
    denoising step to [-1, 1].
    """
    algorithm = _algorithm(cfg)
    if algorithm.get("name") != "lerobot_policy":
        return
    normalize = cfg.dataset.get("normalize")
    assert normalize, (
        f"{cfg.dataset.repo_id} feeds a lerobot policy with `normalize:` off, so nothing "
        f"normalizes it"
    )
    keys = set(normalize["keys"])
    wanted = {"action"} | {
        name for name in algorithm.conditioning if not name.startswith("observation.images.")
    }
    assert wanted <= keys, f"unnormalized: {sorted(wanted - keys)}"


def test_robocasa_evaluates_the_tasks_it_trains_on(cfg):
    """A task is named by its RoboCasa task name, the one string dataset and simulator
    agree on (a dataset's task_index is an artifact of packing). That only protects
    anything while both sides read one list -- restating it in the eval config is how
    they silently diverge.
    """
    if cfg.eval.get("backend") != "robocasa":
        return
    tasks = list(cfg.dataset.get("tasks") or [])
    assert tasks, f"{cfg.dataset.repo_id} selects no tasks to train on"
    assert list(cfg.eval.tasks) == tasks, (
        "eval.tasks has drifted from dataset.tasks; it should interpolate ${dataset.tasks}"
    )


def test_effective_batch_is_a_positive_int(cfg):
    effective_batch = cfg.experiment.get("effective_batch")
    assert effective_batch is None or int(effective_batch) > 0


def _decoder_wiring(algo):
    """(entry name, source stream spec, decoder spec) for every pixel decoder a config
    wires up, however it names the link: a `latent_decoder` predict entry (`input:` +
    `decoder:`) or a `wam_decoder` `decode:` entry (`stream:` + `decoder:`).
    """
    groups = [algo.get(g) or {} for g in ("conditioning", "predict", "decode")]
    if not all(isinstance(g, Mapping) for g in groups):
        return
    conditioning, predict, decode = groups
    streams = {**dict(conditioning), **dict(predict)}
    encoders = algo.get("encoders") or {}
    for name, entry in {**dict(predict), **dict(decode)}.items():
        decoder = entry.get("decoder")
        source = entry.get("input") or entry.get("stream")
        if decoder in encoders and encoders[decoder].get("type") == "pixel_decoder":
            yield name, streams[source], encoders[decoder]


def test_pixel_decoders_match_the_streams_they_read(cfg):
    """A decoder restates its source stream's latent width and tokens per step. Both files
    are hand-copied from the run being probed, so this is where a drift shows up."""
    for name, stream, decoder in _decoder_wiring(_algorithm(cfg)):
        assert int(decoder["in_dim"]) == int(stream["dim"]), f"{name}: in_dim"
        grid = tuple(stream.get("grid", (1, 1)))
        assert int(decoder.get("tokens_per_step", 1)) == int(grid[0]) * int(grid[1]), (
            f"{name}: tokens_per_step vs grid {list(grid)}"
        )


def test_pixel_probes_read_pixels_only(cfg):
    """A `latent_decoder` probe declares no state streams, so every dataset column, slice
    and normalization entry would be unconsumed -- a build error for slices. The clears in
    the top-level config must be `null`: an empty dict merges as a no-op and leaves the
    dataset group's entries standing."""
    if _algorithm(cfg).name != "latent_decoder":
        return
    for key in ("columns", "slices", "normalize", "augment"):
        assert not cfg.dataset.get(key), f"dataset.{key} survives in a pixels-only probe"


def test_pixel_decoder_runs_freeze_what_they_measure(cfg):
    """A probe reads a fixed representation: an encoder feeding a decoder while trainable
    would be pulled around by the decoder's own gradient. `wam_decoder` enforces this in
    code (its spec is the probed run's, where those encoders were legitimately training);
    a `latent_decoder` config declares its own stack, so it has to say so here."""
    algo = _algorithm(cfg)
    if algo.name != "latent_decoder":
        return
    decoders = {d for _, _, d in _decoder_wiring(algo)}
    for name, encoder in algo.encoders.items():
        if encoder in decoders:
            continue
        assert encoder.get("frozen", encoder.get("type") in ("vjepa2", "smolvlm")), (
            f"encoder '{name}' of {algo._name} is trainable in a probe run"
        )
