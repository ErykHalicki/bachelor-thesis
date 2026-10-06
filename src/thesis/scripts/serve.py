#!/usr/bin/env python3
"""Remote inference server for on-robot rollouts.

A bare GPU box, nothing more. It starts up knowing nothing about the policy, the
embodiment, or the task: the rollout client (`experiments/eval/lerobot.py` with `server:`
set) connects, names the wandb run to serve and describes the observation columns its
robot produces, and this process downloads that run's checkpoint, rebuilds the model from
the config the run stored at train time, and answers action chunks until the client goes
away. Nothing about a robot is configured here, so the same server serves any embodiment.

Deliberately independent of lerobot's own async inference stack — no lerobot import at
all, so this runs on a machine with no hardware dependencies installed. The transport is
the length-prefixed pickle framing in utils/wire.py over one blocking TCP connection, and
preprocessing is the same ChunkDriver the in-process path uses, so a remote rollout and a
local one feed the model identical tensors.

    python src/thesis/scripts/serve.py                        # 0.0.0.0:8080
    python src/thesis/scripts/serve.py --port 9000 --device cuda:1

Then, on the robot:

    python main.py run=b601_irl load=<run_id> eval.server=<this-host>:8080

Models are cached per run for the life of the process, so reconnecting — or restarting the
client mid-session — costs one handshake, not another download and rebuild. One client at
a time; a second connection waits.

Requires the `serve` extra (`uv pip install -e ".[serve]"`) for wandb and the JPEG codec.
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from thesis.utils.wire import decode_jpeg, listen, recv_msg, send_msg  # noqa: E402

CHECKPOINT_DIR = Path("outputs/served")


OVERRIDES = []


def merge_model_overrides(stored, overrides):
    """Apply dotlist overrides (client handshake `model_overrides` + the server's
    --override) to a checkpoint's stored config. Only `algorithm.*` keys are accepted:
    everything else the server decides from the stored config, and a dataset/experiment
    override here would silently do nothing."""
    from omegaconf import OmegaConf

    clean = [str(o).lstrip("+") for o in overrides]
    bad = [o for o in clean if not o.startswith("algorithm.")]
    if bad:
        raise ValueError(f"model_overrides must target algorithm.*: {bad}")
    return OmegaConf.merge(stored, OmegaConf.from_dotlist(clean)) if clean else stored


def load_model(run_path, device, cache, overrides=()):
    """Build the model a wandb run trained and restore its latest checkpoint, or the
    version named by a `:<alias>` suffix (`<entity>/<project>/<id>:best`).

    The algorithm section comes from the config the run stored, never from a local yaml,
    so the rebuild cannot drift from what actually trained — the same rule main.py's
    post-hoc eval follows, and the reason this server needs no config of its own. A
    client may adjust inference-only algorithm keys per handshake via `model_overrides`
    (e.g. `algorithm.inference_streams=[action]`); models are cached per (run,
    overrides) pair.
    """
    key = (run_path, tuple(str(o) for o in overrides))
    if key in cache:
        return cache[key]

    from thesis.algorithms import build_algorithm
    from thesis.utils.checkpoint import load_checkpoint
    from thesis.utils.ckpt_utils import (config_from_checkpoint, download_latest_checkpoint,
                                        fetch_run_config, split_alias)

    print(f"loading {run_path} ...", flush=True)
    run, alias = split_alias(run_path)
    path = download_latest_checkpoint(run, CHECKPOINT_DIR, alias=alias)
    # embedded config first: it preserves the declaration order the model's token and
    # rope layouts were trained with; the wandb copy (alphabetized, reordered on fetch)
    # covers checkpoints from before configs were embedded
    stored = config_from_checkpoint(path)
    if stored is None:
        stored = fetch_run_config(run)
    stored = merge_model_overrides(stored, [*OVERRIDES, *overrides])
    model = build_algorithm(stored.algorithm)
    step = load_checkpoint(model, None, path)
    model = model.to(device).eval()
    print(f"serving {run_path} @ step {step} on {device} "
          f"(overrides: {list(overrides) or 'none'})", flush=True)
    cache[key] = (model, step)
    return cache[key]


def handle_init(msg, device, cache):
    """Client handshake -> (driver, ack). The client's own eval knobs (image_size,
    columns, slices, action_filter, and the sampling knobs) and its robot's observation
    schema arrive here, so every field and column decision is made once, on this side, and
    the client is told only which columns to buffer and at which offsets.

    The sampling knobs are applied per handshake, on a model this process caches across
    sessions, so each client gets what it asked for and an omitted knob is restored to the
    value the run trained with rather than inherited from the previous client.
    """
    from thesis.experiments.eval.chunking import SAMPLING_KNOBS, ChunkDriver, PlanDriver

    keys = ("image_size", "columns", "slices", "action_filter", "execute_len", "task",
            *SAMPLING_KNOBS)
    model, step = load_model(msg["run"], device, cache,
                             overrides=msg.get("model_overrides") or ())
    cfg = {k: msg.get(k) for k in keys}
    if model.action_stream is not None:
        driver = ChunkDriver(model, cfg, msg["obs_features"])
    else:
        # a world model plans toward the goal image the client sent with the handshake
        if not msg.get("goal"):
            raise ValueError(
                f"{msg['run']} is a world model and plans toward a goal: the client must "
                f"set eval.goal_image"
            )
        driver = PlanDriver(model, cfg, msg["obs_features"], decode_jpeg(msg["goal"]),
                            plan=msg.get("plan"))
    print(f"session: execute_len={driver.execute_len} "
          + " ".join(f"{k}={v}" for k, v in driver.sampling.items()), flush=True)
    return driver, {
        "ok": True,
        "run": msg["run"],
        "step": step,
        "columns": driver.columns,
        "offsets": driver.offsets,
        "execute_len": driver.execute_len,
        "sampling": driver.sampling,
        # only the client knows where an episode started, so it is told which offsets to
        # send rather than the server guessing
        "action_offsets": driver.action_offsets,
    }


def serve_client(conn, device, cache):
    driver = None
    while True:
        try:
            msg = recv_msg(conn)
        except (ConnectionError, EOFError, OSError):
            return
        try:
            kind = msg.get("type")
            if kind == "init":
                driver, reply = handle_init(msg, device, cache)
            elif kind == "predict":
                if driver is None:
                    raise RuntimeError("predict before init: the client must handshake first")
                # timed from before the decode: unpacking the frames is work this box does, and
                # billing it to the wire would send an operator debugging their network astray
                started = time.perf_counter()
                window = [
                    {col: decode_jpeg(v) if isinstance(v, bytes) else v for col, v in frame.items()}
                    for frame in msg["frames"]
                ]
                actions = driver.predict_window(window, msg.get("actions") or [])
                reply = {"actions": actions, "compute_s": time.perf_counter() - started}
            else:
                raise ValueError(f"unknown message type {kind!r}")
        except Exception:
            # a bad request costs the client an error, never the loaded model
            traceback.print_exc()
            reply = {"error": traceback.format_exc()}
        send_msg(conn, reply)


def main():
    p = argparse.ArgumentParser(description="Remote inference server")
    p.add_argument("--host", default="0.0.0.0", help="interface to bind (default: all)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    p.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                   help="dotlist override applied to every served checkpoint's stored "
                        "config before the model builds, e.g. "
                        "--override 'algorithm.inference_streams=[action]' to serve a "
                        "selfwam arm policy-only (its clean-action stream has no rows "
                        "at serve time)")
    args = p.parse_args()
    global OVERRIDES
    OVERRIDES = list(args.override)

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache = {}
    server = listen(args.host, args.port)
    print(f"inference server on {args.host}:{args.port}, device={device}", flush=True)
    try:
        while True:
            conn, addr = server.accept()
            print(f"client connected: {addr[0]}:{addr[1]}", flush=True)
            try:
                serve_client(conn, device, cache)
            finally:
                conn.close()
                print(f"client disconnected: {addr[0]}:{addr[1]}", flush=True)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        server.close()


if __name__ == "__main__":
    main()
