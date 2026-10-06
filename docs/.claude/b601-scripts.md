# b601 and serve scripts

Operating reference for the entry points in `src/thesis/scripts/`. Each script also takes `--help`.

## b601/gravity_compensation.sh

Holds the B601-DM follower against its own weight so it can be pushed around by
hand — for kinesthetic demos, or just to reposition the arm without fighting it.
The arm runs in MIT mode with `kp` dropped to zero and a feedforward torque
equal to its gravity load, `tau = g(q)`, recomputed each tick from the measured
joint angles.

**The follower already does this on every session, teleoperation included** —
see `gravity_compensation` in `RebotB601FollowerConfig`, which adds `tau = g(q)`
to every MIT command from the angles the follower process already reads each
tick (~200 µs of a 10 ms tick). Without it a joint has to sit `g(q)/kp` below
its target to make its own holding torque, which is 8° at this arm's elbow and
13–19 cm at the end effector — a standing tracking error under teleoperation,
and a pose-dependent gap between a recorded action and the state it produced.

This script only takes the *position gains* away on top of that. Same shape as
`record.sh`: the wrapper resolves ports (shared `common.sh`) and hands off to a
Hydra app, and homing / Ctrl-C handling / disconnect are the shared helpers in
`b601/common.py` rather than a second implementation.

`g(q)` lives in the lerobot fork at `robots/rebot_b601_follower/gravity_model.py`:
link masses and joint origins from the manufacturer's URDF, and the
potential-energy gradient in plain numpy. It agrees with pinocchio's
`computeGeneralizedGravity` to 1e-14 N·m, without the eigenpy/boost install on
the arm's Pi.

Check the model before letting the arm carry itself. `dry_run=true` reads the
live pose and prints the torques next to what the motors report, without ever
moving the arm:

```
./src/thesis/scripts/b601/gravity_compensation.sh dry_run=true
```

Joint angles are raw motor angles, so this assumes the motor zeros were set at
the URDF zero pose (where `lerobot-calibrate` homes). If the printed torques
don't match the direction the arm actually wants to fall, the zeros are wrong —
fix those (`--recalibrate`) before going further.

```
./src/thesis/scripts/b601/gravity_compensation.sh
./src/thesis/scripts/b601/gravity_compensation.sh hold=true
./src/thesis/scripts/b601/gravity_compensation.sh gain=0.95 '++scale={elbow_flex: 1.2}'
./src/thesis/scripts/b601/gravity_compensation.sh '++joints=[shoulder_lift]'
```

Controls, via the same keyboard backend as `record.py`:

| Key | Effect |
|---|---|
| Up / Right / `l` | Lock: hold the angles the arm is at right now |
| Down / Left / `f` | Unlock: float again |
| Esc / `q` | Stop and ramp home |

- **Locking and unlocking are instant.** Nothing is faded, because the follower
  carries the weight at every gain setting: removing the spring leaves a
  balanced arm, and restoring it engages against a goal already at the current
  pose. A joint only lurches when the two hand over unevenly, and here they
  never do.
- "Locked" means position control *plus* gravity feedforward, not position
  control alone — which is why a locked joint holds its angle *at* its angle
  instead of drooping `g(q)/kp` below it.
- Homing gets the same assist, at both ends of the session, so the arm no longer
  hangs behind the ramp on its way home.
- Default is free-floating (`kp=0`); `hold=true` adds a light spring so the arm
  drifts back to where it was released. `joints=[...]` floats a subset for
  per-joint checks — the rest stay stiff rather than going limp.
- The trim knobs (`gain`, `scale`, `payload_kg`, `payload_com`, `max_torque`)
  are the follower's own `gravity_*` config fields — set them there to affect
  every session, or pass them here to override for one run. Left unset they
  defer to the follower, so the arm behaves the same as it does under
  `record.py`.
- The model is a rigid-body ideal and knows nothing about joint friction or
  cable drag, so a rig may need trimming; it ships untrimmed. The live readout
  prints applied torque against measured, which is what tells you which joint
  needs it — a joint reading consistently heavier than it was told to pull wants
  its `gravity_scale` raised.
- Safety: torque is clamped to `min(motor rating, gravity_max_torque)`, so it
  only ever binds on the three 27 N·m joints — the wrists stay capped at their
  own 7 N·m regardless. It defaults to 20 because `shoulder_lift` needs up to
  15.5 N·m somewhere in its travel (23 carrying 1 kg); set lower than that and
  the shoulder sags at full reach rather than being protected.
- The session stops if any joint exceeds `max_vel_deg_s`. Requires MIT mode —
  pass `--update-mode` if a motor was left in another mode by an earlier run.

## b601/temperatures.sh

Prints the MOSFET and rotor temperature of every motor, and the follower's own
estimate of how long each has before it trips `temp_max_c` (65 °C, where the
thermal protection homes the arm and disconnects it).

```bash
./src/thesis/scripts/b601/temperatures.sh                          # one reading
./src/thesis/scripts/b601/temperatures.sh watch=true interval=5    # until Ctrl-C
./src/thesis/scripts/b601/temperatures.sh watch=true duration=600  # for 10 minutes
```

- Same source as the protection itself: every Damiao feedback frame carries both
  temperatures, `ThermalMonitor` in the follower process reads them here exactly
  as it does under teleop or a rollout, and the `overheat` row is that monitor's
  estimate rather than a second one. Its 57 °C / 61 °C warnings print on this
  terminal too.
- The arm is never commanded. Torque is dropped right after connecting, so the
  arm is as limp during the check as it was before it — rest it somewhere it can
  sit unpowered first.
- `overheat` is a fit over the last `temp_history_s` of 1 °C-quantised readings.
  On a motor that is actually heating it is worth acting on; on an idle arm it
  flickers between `-` and a spurious minute or two.
- The estimates need `temp_debug` on the follower, which this script sets. Under
  `gravity_compensation.sh` they cost `debug_temp=true`, and are off otherwise.

## b601/teleop.sh

Teleoperates the reBot B601-DM follower arm with the reBot Arm 102 leader via
lerobot's `lerobot-teleoperate` CLI. Requires the `b601` extra
(`uv pip install -e ".[b601]"`), which pulls in `motorbridge` and
`lerobot[hardware]`.

```
./src/thesis/scripts/b601/teleop.sh
```

- On first run (or whenever the saved USB ports no longer exist), it prompts
  you to plug in both arms, then runs `lerobot-find-port` once per arm,
  walking you through the unplug/replug prompt and parsing the detected port
  from its output.
- Detected ports are saved to `src/thesis/scripts/b601/.env` (gitignored,
  per-machine) so subsequent runs skip detection and go straight to
  teleoperation.
- Override without touching the saved file:
  ```
  FOLLOWER_PORT=/dev/cu.usbmodemXXXX LEADER_PORT=/dev/cu.usbserial-XXXX ./src/thesis/scripts/b601/teleop.sh
  ```
- Delete `src/thesis/scripts/b601/.env` to force re-detection (e.g. after
  swapping a cable/adapter).
- `CONTROL_MODE` selects the follower's arm control mode (`mit`, the
  lerobot default, or `pos_vel`), e.g.:
  ```
  CONTROL_MODE=pos_vel ./src/thesis/scripts/b601/teleop.sh
  ```

## b601/record.sh

Records a LeRobot dataset via the same teleoperation setup, with cameras
attached and the ability to pick a new task after each episode. Shares port
detection/control-mode reset with `teleop.sh` (see `b601/common.sh`).

The arm settings the episodes were recorded under land in the dataset's
`meta/info.json`, under `robot_config`, reachable as
`LeRobotDatasetMetadata.robot_config`. `robot_type` says which arm; this says
how it was driven — control mode, gains, gravity compensation, smoothing, joint
limits — which is what decides the state a given action actually produces. Two
datasets off the same arm are only interchangeable if these match.

Resuming a dataset compares the stored settings against the current ones and
asks before appending if any of them moved, since mixing control regimes in one
dataset teaches a policy that the same action has two different outcomes.
Settings that vary harmlessly between sessions (ports, log paths, camera
wiring) are not compared. Datasets recorded before this existed have no stored
config and append without a prompt.

Episode boundaries differ from `lerobot-record`'s: episodes and scene resets are
untimed and end on a keypress (right/`n` finish, left/`r` re-record, esc/`q`
stop), rather than being cut off by a countdown. Between episodes the follower
ramps back to its calibration zero pose while the previous episode encodes, and
the next episode does not start until the leader has been brought back to that
pose too — otherwise the first `send_action` snaps the arm across the workspace
to wherever the leader was left. Set `home_between_episodes=false` to skip that,
or `episode_time_s` / `reset_time_s` to put the clocks back.

`record.py` is a Hydra app (config: `src/thesis/scripts/b601/configs/`); pick a
dataset's config file with `--config-name`, or override fields directly with
`key=value`:

```
./src/thesis/scripts/b601/record.sh --config-name=<dataset>
./src/thesis/scripts/b601/record.sh repo_id=<hf_username>/<dataset_name> num_episodes=10
```

### Cameras

A dataset config's `cameras` block is the yaml form of lerobot's
`--robot.cameras='{...}'`. It defaults to empty (state-only recording); no
camera arrangement is baked into the script.

```yaml
cameras:
  zed_left:  {type: zed, side: left, serial_number: auto, width: 672, height: 376, fps: 30}
  wrist_cam: {type: opencv, index_or_path: auto, width: 640, height: 360, fps: 30, fourcc: MJPG}
```

Each key is the observation key it becomes in the dataset, so name it for what
the camera sees. `type` is any registered lerobot camera backend (`opencv`,
`zed`, `intelrealsense`, `zmq`, ...) and the remaining fields are that
backend's own, so anything lerobot supports works without code changes.

An arrangement used by more than one dataset belongs in its own file under
`b601/configs/cameras/`, selected by name in a dataset config's defaults so the
rig is described once:

```yaml
defaults:
  - record_schema
  - cameras: zed_c270
  - _self_
```

Each file describes one camera, and a rig with several is just their defaults
list — every setting still has exactly one home:

- `zed` — ZED-Mini left + right eyes.
- `c270` — a C270 on the wrist mount.
- `zed_c270` — `defaults: [zed, c270]`, nothing of its own.

A config can do both: an inline `cameras:` block merges on top of the selected
file, so one dataset can override a field, add a camera, or drop one
(`wrist_cam: null`) without copying the shared rig.

Devices can be pinned or auto-detected: `serial_number` (zed) and
`index_or_path` (opencv) may be `auto`, or omitted entirely. The ZED is then
looked up through its own SDK and its video nodes are kept out of the webcam
search, since a ZED also enumerates as a plain UVC device; several `auto`
webcams resolve to different nodes. Pin them once more than one camera of a
kind is attached, so which is which can't change between boots.

Overrides from the command line (`++` because these keys aren't in the base
config, quoted because of the braces):

```
./src/thesis/scripts/b601/record.sh --config-name=<dataset> \
    '++cameras.overhead={type: opencv, index_or_path: /dev/video4, fps: 30}'  # add one
./src/thesis/scripts/b601/record.sh --config-name=<dataset> '~cameras.wrist_cam'   # drop one
./src/thesis/scripts/b601/record.sh --config-name=<dataset> cameras.zed_left.width=1280
```

Adding, dropping, or renaming a camera changes the dataset's observation keys,
so treat the block as part of the dataset's identity rather than a run-time
knob — `~cameras.wrist_cam` is fine for a test, but a real dataset should have
its own config file.

## serve.sh

Remote inference for on-robot rollouts: a GPU box that answers action-chunk
requests while the arm's own machine does nothing but drive hardware. Use it
when the robot's computer can't run the policy at the control rate — the Pi on
the B601 can't — or when you want to test a checkpoint without copying it off
the training box.

The server holds no configuration of its own. It starts up knowing nothing about
the policy, the embodiment, or the task; the client's handshake names the wandb
run to serve and describes the observation columns its robot produces, and the
server downloads that run's checkpoint and rebuilds the model from the config the
run stored at train time. The same server therefore serves any embodiment, and
switching to a different checkpoint is a client-side flag, not a restart.

On the GPU box, one command from a fresh checkout — the wrapper creates `.venv`
if it's missing, installs the `serve` extra if the imports aren't there yet
(wandb and a JPEG codec; no lerobot, no hardware dependencies), and starts the
server:

```
./src/thesis/scripts/serve.sh                     # 0.0.0.0:8080
./src/thesis/scripts/serve.sh --port 9000 --device cuda:1
./src/thesis/scripts/serve.sh --reinstall         # after deps drift
```

Flags other than `--reinstall` pass straight through to `serve.py`, which is
also runnable directly (`python src/thesis/scripts/serve.py --help`) if the
environment is already set up. The wrapper warns up front when wandb looks
unauthenticated, since otherwise the first client handshake is where the
checkpoint download fails.

On the robot, the ordinary post-hoc eval command plus `eval.server`:

```
python main.py run=b601_irl load=<run_id> eval.server=<gpu-box>:8080
```

(`run/b601_irl` is the real-arm rollout of whatever run `load=` names: the rig, the `irl/*`
prefix and eval-only tasks come from it, and the model plus the data contract come from that
run's stored config. A training arm whose own config selects a b601 eval (e.g. `flow_wam_b601_pnpt` with
`eval=b601_pnpt`) also works, with `experiment.tasks=[validation]` spelled out.)

Everything else about the rollout is unchanged — same episode controls, same
human success tagging, same homing between episodes, same wandb write-back — and
the results attach to the source run exactly as a local eval's do. The only
difference is where `predict` happens.

- **The robot box builds and loads nothing.** With `eval.server` set, `main.py`
  skips the checkpoint download and the eval task passes no model at all, so
  that machine needs neither a GPU nor the run's algorithm config. It follows
  that `load=` must name a wandb run rather than a local `.pt` the server cannot
  fetch; point the server at a specific run with `eval.run=<entity>/<project>/<run_id>`
  if you want something other than what `load=` names.
- **Replans block.** The arm holds its last commanded pose for the round trip
  rather than executing actions predicted from a staler observation, so a chunk
  boundary is a visible pause of one RTT. Budget for it: `execute_len` decides
  how often it happens, and `profile_log=<path>` records the cost per tick. A
  round trip slower than `execute_len / fps` means the policy never really runs
  at `fps`, and the existing slow-tick warning will say so.
- **Only the conditioned frames cross the wire.** The server dictates which
  columns to buffer and at which offsets, so a sparse window like `-11, -5, 0`
  over a 12-frame history sends 3 frames per replan, not 12. Camera frames are
  JPEG-encoded (`eval.jpeg_quality`, default 95).
- **Preprocessing is server-side**, all of it: resizing, `slices:`, `columns:`,
  and normalization from the checkpoint's own stats. Local and remote rollouts
  feed the model identical tensors — the same `ChunkDriver`
  (`experiments/eval/chunking.py`) runs in both, which is what makes that
  guarantee testable rather than aspirational.
- Models are cached per run for the life of the process, so reconnecting — or
  restarting the client mid-session — costs a handshake, not another download.
  The server takes one client at a time and survives a client's bad config: the
  error goes back over the wire and the loaded model stays put.
- This is deliberately **not** lerobot's async inference stack. There is no
  lerobot import on the server side, no policy registry, and no shared queues —
  just length-prefixed pickle (`utils/wire.py`) over one blocking TCP
  connection. Pickle executes arbitrary code on load, so run it on a trusted
  network and never expose the port beyond one.
