# scripts/

Entry points that don't belong in `main.py`'s Hydra flow. Each takes `--help`.

```bash
source .venv/bin/activate

# float the B601 follower on its own gravity model, so it can be pushed by hand
./src/thesis/scripts/b601/gravity_compensation.sh

# teleoperate the follower with the Arm 102 leader
./src/thesis/scripts/b601/teleop.sh

# record a LeRobot dataset, cameras attached; one config per dataset
./src/thesis/scripts/b601/record.sh --config-name=<dataset>

# answer action-chunk requests from a GPU box, for on-robot rollouts
./src/thesis/scripts/serve.sh
```

The b601 scripts need the `b601` extra (`uv pip install -e ".[b601]"`). They detect the
arms' USB ports on first run and cache them in `src/thesis/scripts/b601/.env` — delete it
to force re-detection after a cable swap.

Full operating reference, including camera configuration and the recording workflow:
**[docs/.claude/b601-scripts.md](../../../docs/.claude/b601-scripts.md)**.
