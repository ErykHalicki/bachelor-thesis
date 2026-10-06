# B601 Teleop Jitter Investigation

Debugging log for the wrist jitter / freeze observed when teleoperating the
reBot B601-DM follower with the reBot Arm 102 leader through lerobot v0.6.0.

> **Historical log.** `record.py` has since become a Hydra app, so the `--flag value`
> options below are now `key=value` overrides (`smoothing_time_constant=0.08`,
> `image_format=jpg`, `control_fps=60`, ...). Line numbers into the lerobot fork were dropped
> because they drift; search the files by symbol name instead.

## Symptom

During lerobot teleop, the follower wrist visibly jittered, and periodically the
whole arm would hard-freeze for a fraction of a second then snap to catch up. It
did not happen during direct motor control. The perceived jitter was strongest
on the wrist, which is simply the joint being moved and watched most closely.

## Approach: tiered standalone reproduction

Rather than keep guessing inside lerobot, we built a standalone driver that
bypasses lerobot entirely and adds one layer of complexity at a time, driving
the follower directly through `motorbridge` and reading the leader directly
through `motorbridge_smart_servo`. Script: `src/thesis/scripts/b601/tiered_teleop.py`.

| Tier | Leader reads | Follower commands | Follower feedback | Result |
|------|--------------|-------------------|-------------------|--------|
| 1 | none (code triangle) | wrist only | no | smooth |
| 2 | servo 3 only | wrist only | no | smooth |
| 3 | all 7 | wrist only | no | smooth |
| 4 | all 7 | all 7 | no | smooth |
| 5 | all 7 | all 7 | yes | smooth, faster than real teleop |

Every tier was smooth. Tier 5 is a faithful replica of lerobot's loop shape
(read leader, command follower, read follower feedback), yet it was noticeably
faster and had none of the jitter. That isolated the problem to lerobot's
per-loop overhead, not the motors, the CAN bus, the leader signal, the command
traffic, or the feedback read.

## Finding 1: motor stored in the wrong control mode

While arming the follower, the wrist_flex motor was found stored in `POS_VEL`
(mode 2) instead of `MIT` (mode 1). lerobot's `ensure_mode()` writes the runtime
CTRL_MODE register but the change is not persisted, so it only works when the
motor is already in the target mode (the write is a no-op verify). A motor left
in the wrong mode by an earlier run makes `connect()` fail with
`control mode verify failed`, or worse, silently misinterpret command frames.

Fix: write CTRL_MODE directly and persist it with `store_parameters()`. This is
implemented as `src/thesis/scripts/b601/set_control_mode.py` and is now run automatically by
`src/thesis/scripts/b601/teleop.sh` before each teleop session.

## Finding 2: lerobot loop overhead (the real jitter cause)

Comparing lerobot's `teleop_loop` to our smooth standalone loop, three things in
lerobot's loop added latency and timing irregularity:

1. **`precise_sleep`** busy-spins the final ~6-10ms of every 60Hz cycle on macOS,
   burning a full CPU core. That spin competes with the serial reader threads, so
   leader/follower reads get delayed irregularly.
2. A blocking **`print(...)` + `move_cursor_up`** on every single tick. When the
   terminal buffer backs up the loop stalls.
3. Console logging at **DEBUG**, so `get_observation` / `get_action` /
   `send_action` each emit a line per tick (~240 terminal writes/sec).

Replacing `precise_sleep` with `time.sleep` and removing the per-tick terminal
output made lerobot teleop match the standalone: smooth, and faster. This is the
fix worth upstreaming, and it lives on the lerobot submodule branch (see below).

Two changes we tried that did **not** meaningfully help, and were dropped:
- Skipping `get_observation()` when `display_data` is off (a valid micro
  optimization, but not the cause).
- Moving the leader read into a background thread so a stalled read cannot freeze
  the control loop. It decoupled the stall from the loop timer but did not remove
  the visible freeze, because the freeze is missing leader data, not loop timing.

## Finding 3: leader comms stalls (hardware)

After the loop fix, a residual ~100ms freeze remained every few seconds. Timing
the leader read directly (inside the reader thread) showed the leader
`sync_monitor` call blocking for ~100-118ms whenever one or more servos failed to
respond that cycle. That ~100ms is a fixed internal serial read-timeout in the
compiled `motorbridge_smart_servo` core, and it is not configurable from Python
(no timeout parameter on `sync_monitor`, the constructor, or any setter).

Logging which servos were stale during each stall showed the misses were spread
across the back half of the leader chain (gripper always, wrist_yaw / wrist_roll
often, and after a reseat, intermittently across shoulder_lift / elbow_flex too).
That pattern is bus-wide marginal comms, not one bad connector. Likely root
causes, in order: the USB-to-UART adapter and its cable, the trunk cable /
connector to the first servo and leader power, or running 1 Mbaud on a marginal
harness.

This looked like a hardware issue on the leader at the time: the marginal comms
themselves are a hardware fact, and software cannot recover leader data that
never arrives, nor shorten the driver's internal read timeout from Python. But
see the update just below: the stalls were still avoidable in software, by
never sending a `sync_monitor()` batch large enough to trigger them in the
first place.

### Update: batch-size limit quantified, and the production-side fix

Follow-up session confirmed the threshold directly: a standalone `sync_monitor()`
loop querying all 7 servos in one call averaged 67.4ms/call (max 106.8ms) over
200 calls, with most servos reliable only 2-4% of the time (`shoulder_lift`,
`elbow_flex`, `gripper`); only `shoulder_pan` and `wrist_flex` came back
consistently. The same test querying one servo at a time stays fast and
reliable. So the limit isn't a specific bad servo, it's the *number of ids per
`sync_monitor()` call*: ≤4 stays in the fast (~3-5ms) regime, more than that
intermittently stalls ~100ms regardless of which ids are included.

`RebotArm102Leader._read_raw_positions()` (`external/lerobot/src/lerobot/teleoperators/rebot_102_leader/rebot_102_leader.py`)
already implements the fix: it chunks the 7 servo ids into `_MAX_SYNC_MONITOR_IDS
= 4`-sized batches (two `sync_monitor()` calls per tick instead of one) so
production `record.py`/`teleop.sh` sessions stay in the fast regime for every
servo. This is a real fix, not a workaround: reading ≤4 servos per call keeps
every call inside the ~3-5ms regime and avoids the ~100ms stall entirely, so
the "hardware issue" framing above only applies to the underlying marginal
comms, not to whether the resulting stall is avoidable, it is, from software,
by staying under the batch-size limit. `tiered_teleop.py`'s tiers 3-5 call
`leader.sync_monitor(read_ids)` with all 7 ids in a single batch and were never
updated with this chunking, so they reproduce this exact stall/unreliability by
design, and their failures are consistent with this finding, not a new
regression.

## Finding 4: follower enable-ordering fault (comm-timeout on connect)

A related but distinct bug, found while re-testing after Finding 3's chunking
fix: on connect, the follower motors' status LEDs would flash green very
briefly and then start blinking red, and the arm would not move at all.

`RebotB601Follower.configure()` (`external/lerobot/src/lerobot/robots/rebot_b601_follower/rebot_b601_follower.py`)
used to call `self.bus.enable_all()` *before* looping through the motors to
verify/set each one's mode via `ensure_mode()`. `enable_all()` makes every
motor live and expecting continuous commands immediately, but the per-motor
`ensure_mode()` loop that followed touched each motor one at a time (with up to
`_ENSURE_MODE_RETRIES + 1` attempts each on failure). Motors enabled early in
that loop sat idle, receiving no traffic, for however long it took to process
the rest, tripping their own Damiao communication-timeout fault (`RID_TIMEOUT`,
read as 600ms on this arm; status_code 13/0xD = comm-loss) before the control
loop ever sent a real command. This is the same class of bug as Finding 3's
`sync_monitor` batching, and the same class of bug independently found and
fixed in `tiered_teleop.py` (see below): something gets marked "live" and then
starves for commands while other work finishes.

First fix attempt: reorder `configure()` to verify every motor's mode first,
then call `self.bus.enable_all()` once, at the end. This changed the symptom
(only the last 2 of 7 motors ended up enabled, the rest faulted) but didn't
resolve it, revealing a second problem: `self.bus.enable_all()` itself is not
reliable on this serial-bridge hardware, independent of ordering.

Final fix: replace the single `self.bus.enable_all()` call with an explicit
per-motor loop, `for motor in self.motors.values(): motor.enable()`, mirroring
the pattern already confirmed working in `tiered_teleop.py`'s tier 4 (see
below). Confirmed working end-to-end via `teleop.sh`.

`tiered_teleop.py` needed the equivalent two-part fix on the follower-arming
side:
1. Leader connect/unlock/reset moved to *before* arming the follower (it's a
   multi-round-trip sequence on a separate serial port that sends the follower
   no traffic; running it after the follower was armed could outlast the
   follower's own comm-timeout).
2. `configure_motor()` split into `ensure_mit_mode()` (mode-setting only) and a
   separate final pass that calls `motor.enable()` on all armed motors in a
   tight loop, immediately before the control loop starts, instead of enabling
   each motor as it's individually configured.

## Finding 5: record.py's per-episode task prompt also faults the follower

A distinct comm-timeout fault, this time specific to `record.py` (not
`teleop.sh`): the follower's LEDs went green on connect, then blinked red
before the first episode ever started. The user diagnosed the likely cause
correctly: `pick_task()`'s blocking `input()`, called before every episode
(not just the first), leaves the follower enabled but silent for however
long the person takes to type a task, tripping the same `RID_TIMEOUT`
watchdog described in Finding 4.

First fix, wrapping the prompt in `robot.disable_torque()` /
`robot.enable_torque()`, was not sufficient on its own, the fault still
occurred. That means the watchdog fires even on a disabled motor, not just
an enabled-but-uncommanded one. Final fix: also raise the Damiao
`RID_TIMEOUT` register (9) from its default 600ms to 60000ms (60s) on all 7
follower motors, via
`motorbridge-cli damiao-write-param --param-id 9 --type u32 --value 60000 --verify 1 --store 1`,
giving `pick_task()`'s prompt effectively unlimited time without the motor
watchdog tripping. Confirmed via register read-back (`damiao-read-param`)
that all 7 motors persisted `60000`, and via a live `record.sh` session with
no fault. The torque disable/enable wrap was kept alongside the timeout
increase since it's harmless and correct regardless.

## Finding 5b: a flaky camera should degrade, not crash the session

Separately from the fault/timing findings above: a momentary USB webcam
glitch (`errno=19 ENODEV`-style read failure, camera not actually
disconnected) used to crash `record_loop()` outright, since both camera
backends' background read threads (`OpenCVCamera` and `ZedCamera`) gave up
and killed themselves after 10 consecutive read failures, so even a camera
that recovered a moment later would raise "read thread is not running" on
every subsequent read for the rest of the session.

Fixed in three places:

1. `OpenCVCamera._read_loop()` and `_ZedDevice._read_loop()`
   (`external/lerobot/src/lerobot/cameras/{opencv,zed}/...`): never die.
   Retry forever with a 0.1s backoff, logging a throttled warning (1st
   failure, then every 30th) instead of raising, plus one info line when
   reads resume.
2. `RebotB601Follower.get_observation()`
   (`external/lerobot/src/lerobot/robots/rebot_b601_follower/rebot_b601_follower.py`):
   added `_read_camera_or_last()`, which caches the last successfully-read
   frame per camera key and serves it on failure instead of propagating the
   exception, with the same throttled warning/recovery logging. Only
   re-raises if a camera has never produced a single good frame (nothing to
   fall back to).

Confirmed working on real hardware during the session that produced
Finding 6's profiling data: the webcam went stale for a stretch (logged
"webcam frame not read (90 consecutive failure(s))... reusing last known
frame" and later recovered), and the recording session continued
uninterrupted throughout.

## Finding 6: record.py's loop was 4-8x slower than teleop.sh's, and it is a
software problem, not a hardware ceiling

After Finding 5's fault fix, `record.sh` sessions were still far jitterier
than `teleop.sh`, dropping to 5-15Hz against a 30Hz target. Comparing
`--profile-log` output from a `teleop.sh` run and a `record.sh` run captured
back to back (same hardware, same motors, same arms, only add the dataset
being written), the `RebotB601Follower` CAN state read
(`get_observation()`'s `_present_pos()` call, logged as `read state: ...ms`)
told the whole story:

| | teleop.sh | record.sh |
|---|---|---|
| `read state` (CAN) median | 3.9ms | 65.6ms |
| `read state` (CAN) p90 | 5.4ms | 117.9ms |
| `total_loop` median | 15.9ms (~60Hz) | 119.0ms (~8Hz) |

Same code path, same bus, same motors, reading the same 7 registers, yet
16x slower under `record.sh`. That rules out a fixed CAN/serial hardware
ceiling as the (sole) explanation, since the hardware itself did not change
between the two runs, only whether a `LeRobotDataset` was attached and
writing.

The culprit: `record.py` created its dataset with
`image_writer_threads=4 * len(cameras)`, 12 threads for this rig's 3 camera
keys (`zed_left`, `zed_right`, `webcam`), each thread doing CPU-bound
PIL/PNG encode-and-write work per frame. This Pi has 4 physical cores
(`nproc`), so 12 writer threads plus the main control-loop thread (plus the
leader's own reader thread, plus each camera's background capture thread)
massively oversubscribes the CPU, and the main thread's CAN read gets
starved of scheduling time by the writer threads competing for the same
cores. `teleop.sh` has no dataset and no writer threads at all, so it never
hits this contention.

Confirmed directly on hardware: running `record.py` for a short session with
`--image-writer-threads 3` (one per camera, matching this Pi's 4 cores minus
the main loop) instead of the default 12 dropped `read state` from a 84.8ms
median / 128.7ms p90 down to a 6.7ms median / 44.3ms p90, and `total_loop`
from a 133.7ms median down to 27.7ms, essentially matching `teleop.sh`.
Fixed by changing `record.py`'s default from `4 * len(cameras)` to
`len(cameras)` (1 writer thread per camera), and adding
`--image-writer-threads` as an explicit override for tuning on other
hardware. This is separate from, and unaffected by, the leader's own
`sync_monitor` stalls covered in Finding 3, those were already fixed on the
production side by chunking reads to ≤4 servo ids per call, which keeps
every call in the fast regime and avoids the ~100ms stall rather than
tolerating it.

Between Finding 3's chunking fix and this one, both of `record.py`'s
software-side jitter sources (the leader's oversized `sync_monitor` batches,
and the writer-thread CPU oversubscription) are now resolved, on top of the
comm-timeout fault fixes in Findings 4 and 5. What's left of Finding 3's
original ~100ms stalls only reappears if something calls `sync_monitor()`
with more than 4 ids at once, e.g. `tiered_teleop.py`'s unpatched tiers,
never in the current production `record.py` / `teleop.sh` path.

**Update**: this finding's writer-thread-count fix was the right diagnosis
but an incomplete fix, it traded CAN-read speed for a write backlog paid at
`save_episode()` time. See Finding 8 for the real fix (a cheaper intermediate
image format) and why thread/process count alone can't resolve it on this
hardware.

## Finding 6b: stale motor fault not cleared on connect

A separate fault-LED symptom: red flashing on connect, even before the first
control tick, specifically when a motor had been left in a fault state by an
earlier session (e.g. an unclean exit). `teleop.sh` worked fine as long as no
motor was already faulted when the script started, which pointed at
connect-time state rather than anything about the control loop itself.

Root cause: `RebotB601Follower.configure()` and `enable_torque()` called
`motor.enable()` without first clearing any latched Damiao fault (the same
comm-timeout fault described in Findings 4 and 5). `enable()` alone does not
clear a fault, it just tries to resume normal operation, which immediately
faults again if the underlying condition (or a persisted fault flag) is still
set. `disconnect()` already called `motor.clear_error()` on the way out
(so a *clean* shutdown left motors fault-free for next time), but nothing
called it on the way in, so a fault surviving an unclean exit stayed stuck
until someone happened to run something that cleared it manually.

Fix: `motor.clear_error()` for every motor, before the mode-verify+enable
sequence, in both `configure()` and `enable_torque()`
(`external/lerobot/src/lerobot/robots/rebot_b601_follower/rebot_b601_follower.py`).
Confirmed on hardware: starting from a red-flashing state, `teleop.sh` now
connects cleanly (green LEDs, no fault) instead of requiring a manual clear
first.

## Finding 7: control_fps decoupling

`record_loop()` takes a `control_fps` (a whole multiple of `fps`; `record.py` defaults it to
60): the get_observation/get_action/send_action cycle runs every control tick, while dataset
frames (`build_dataset_frame` + `add_frame()`) are produced every `control_fps/fps`-th tick.
`None` keeps control and recording at the same rate. With the CPU contention of Finding 8
gone, it is no longer the binding constraint on the Pi; it stays as a knob.

## Finding 8: the actual root cause of record.py's residual slowness was a CPU
budget problem, not a scheduling problem, and the fix is a cheaper image format

After Finding 6's writer-thread-count fix, recording was still visibly
choppier than `teleop.sh`, and shutting down after Ctrl-C could hang for a
long time (sometimes bad enough that repeated impatient Ctrl-C presses left
a motor fault stuck, see Finding 9). Direct measurement on this Pi (`nproc`
= 4 physical cores):

- A single 1280x720 PNG write (`compress_level=1`, the same setting
  `record.py`'s "video" dtype images use) takes **~150ms**.
- 3 cameras at 30fps therefore need `3 * 30 * 0.150s` = **~13.5 CPU-seconds
  of encode throughput per second of wall-clock time**, more than 3x this
  machine's entire 4-core budget, before the control loop gets a single
  scheduling slice.

This means the tradeoff Finding 6 was tuning (thread/process count) can
never actually be resolved by scheduling alone:

| Writer config | CAN read (median) | Effect |
|---|---|---|
| 1 thread/camera | ~6.7ms | Fast control loop, but can only drain ~6.7fps/camera, so a long recording backs up an enormous queue that `save_episode()`'s `image_writer.wait_until_done()` must drain before it can even start encoding, sometimes minutes. |
| 3 processes x 4 threads (multiprocessing, isolates writer CPU work from the main process's GIL) | ~12.8ms | Better than 12 in-process threads, still meaningfully slower than 1 thread/camera; not enough throughput headroom on only 4 cores regardless of GIL isolation. |
| 12 in-process threads (Finding 6's original bug) | ~84.8ms | Drains fast enough to avoid backlog, but the CAN read is starved to the point of being nearly unusable. |

Also tried `--streaming-encoding` (skips per-frame images entirely, encodes
video in real time as frames arrive) as an alternative direction:

| Streaming vcodec | `total_loop` median | Notes |
|---|---|---|
| `libsvtav1` (lerobot's default) | 144ms, spikes to 772ms | Far worse than any PNG-thread configuration; AV1 is simply too CPU-expensive to encode in real time on 4 ARM cores. |
| `h264` | 85ms | Much more stable than AV1 (no runaway spikes), still ~3x worse than `teleop.sh`. |

**The actual fix**: benchmarked PNG vs JPEG at identical resolution:

| Format | ms/frame (1280x720) | CPU-sec/sec needed (3 cams @ 30fps) |
|---|---|---|
| PNG, `compress_level=1` | ~148.9 | ~13.4 (over the 4-core budget) |
| JPEG, `quality=85` | ~11.2 | ~1.0 (comfortably under budget) |

JPEG is roughly 13x cheaper to encode than PNG at this resolution, in budget
even with only 1 writer thread per camera, so there's no longer a
backlog/CAN-read tradeoff to make at all. Added JPEG as a selectable
intermediate image format across the stack (all upstreamable, in the lerobot
submodule):

- `datasets/image_writer.py`: `save_kwargs_for_path()`/`write_image()` accept
  `.jpg`/`.jpeg`, mapping the existing `compress_level` parameter to PIL's
  `quality` kwarg for that format.
- `datasets/dataset_writer.py`: `DatasetWriter` gained `image_suffix`
  (default `.png`, unchanged) and `jpeg_quality` (default 90) constructor
  parameters, threaded through `_get_image_file_path()` (RGB frames only,
  depth frames are unaffected and always `.tiff`) and `add_frame()`'s
  save-quality calculation. `_encode_video_worker()` and
  `_encode_temporary_episode_video()` pass the configured suffix through to
  encoding.
- `datasets/video_utils.py`: `encode_video_frames()` accepts an
  `image_suffix` override for its frame glob (was hardcoded `.png`), and
  `VideoEncodingManager`'s empty-directory cleanup check also globs for
  `.jpg`/`.jpeg` now.
- `datasets/lerobot_dataset.py`: `LeRobotDataset.create()`/`.resume()` both
  gained `image_suffix`/`jpeg_quality` passthrough parameters.
- `record.py`: new `--image-format {png,jpg}` (default **jpg**) and
  `--jpeg-quality` (default 90) flags. Also defaulted `--vcodec` to
  **h264** instead of lerobot's `libsvtav1` default, since the streaming
  benchmark above showed AV1 is far more expensive to encode on this
  hardware even for the final (non-streaming) batch encode at episode end.

Confirmed on hardware with the new defaults (`--image-format jpg
--vcodec h264`, 1 writer thread/camera, no `--streaming-encoding`):
`total_loop` median **14.4ms** and `read_state` median **6.1ms**, matching
or slightly beating `teleop.sh`'s own baseline (15.9ms / 3.9ms respectively). `add_frame` (which now just
enqueues a JPEG write instead of a PNG write) dropped to a median of
**0.3ms**. The batch h264 encode at episode end still takes real time
(~25-30s for a 20s episode on this hardware), which is the accepted
tradeoff: recording-time smoothness was prioritized over between-episode
save latency.

## Finding 9: Ctrl-C could corrupt an in-flight CAN transaction and leave a
motor fault stuck, especially under repeated/impatient Ctrl-C

`record.py` runs on a headless Pi where `TerminalKeyListener` puts the
terminal in cbreak mode (not raw mode), so Ctrl-C still delivers a normal
`SIGINT`/`KeyboardInterrupt` rather than being consumed as a discrete key
event. Python's default `KeyboardInterrupt` can unwind at literally any
bytecode boundary, including in the middle of `robot.send_action()`'s
`motor.send_mit(...)` CAN write. Interrupting a command mid-transaction can
leave the follower in the same comm-fault state described in Findings 4/5/6b
(red flashing). Confirmed via traceback: a `KeyboardInterrupt` raised inside
`_abi.lib.motor_handle_send_mit(...)` correlated with exactly this symptom.

Reported behavior matched this mechanism precisely: a single Ctrl-C
sometimes left the LEDs flashing red for a while (mid-transaction hit) or
solid red (a clean-ish stop that still left torque disabled); repeatedly
"spamming" Ctrl-C to force an unclean exit reliably left the fault stuck,
because a second/third interrupt could also land inside `disconnect()`'s own
`disable()`/`clear_error()`/`close()` cleanup calls, aborting the very code
meant to leave the motors safe.

Fix, in `record.py`:

1. `install_graceful_sigint_handler(events)`: installs a `SIGINT` handler
   that, on the *first* Ctrl-C, sets the same `events["exit_early"]` /
   `events["stop_recording"]` flags the `esc` key already sets, instead of
   raising. `record_loop()` already checks `exit_early` at the very top of
   every iteration (before touching any hardware that tick), so this stops
   within one tick without ever interrupting a live transaction. The
   handler then restores Python's default `SIGINT` behavior, so a *second*
   Ctrl-C still force-exits as before, an escape hatch if something is
   genuinely stuck.
2. The `finally:` block's `robot.disconnect()` / `teleop.disconnect()` calls
   (the ones that actually touch motor state: disable, clear-error, close)
   are now wrapped with `signal.signal(signal.SIGINT, signal.SIG_IGN)`,
   restored immediately after. Even an impatient repeated Ctrl-C during this
   narrow cleanup window can no longer interrupt it.

Confirmed on hardware: a single Ctrl-C during active recording now stops
cleanly with no fault (previously this could go either way depending on
exactly when the signal landed).

## Finding 11: residual jitter after Finding 8 traced to GIL contention, fixed by
moving the motor connection to its own OS process

Even after Finding 8 (image format) resolved the CPU-budget contention,
some jitter remained on `send_action`
timing. Root cause: `send_action()`/the control loop and the camera/image-writer
threads all ran in the same Python process, so they share one GIL. CPU affinity
only controls which physical cores a process's threads may run on; it doesn't
stop one thread from holding the GIL (e.g. during a CPU-bound JPEG encode or a
numpy op) and blocking every other thread in that process from executing
Python bytecode, including the one timing motor sends.

Fix: `RebotB601Follower` now owns the motor connection from a dedicated child
process (`multiprocessing`, spawn context), not a thread. `connect()` opens a
short-lived setup connection to calibrate/configure the motors, closes it, and
hands off to `_follower_process_main()`
(`external/lerobot/src/lerobot/robots/rebot_b601_follower/rebot_b601_follower.py`),
which becomes the sole owner of the real motor connection and runs its own
read/smooth/send loop at `config.send_rate_hz` (100Hz default). Since it's a
separate OS process with its own interpreter and GIL, nothing in the caller's
process — camera reads, image writer threads, dataset writes — can delay it.

The two processes talk only through shared memory (`multiprocessing.Array`/
`Value`, `lock=False`): `goal_pos_shared` (target position, written by
`send_action()`), `present_pos_shared`/`present_pos_ts` (latest reading),
`last_sent_shared` (what was actually sent, for `send_action()`'s return
value). A `command_queue` carries the rare disable/enable/clear_error
messages. `RebotB601Follower.__init__`
(`rebot_b601_follower.py`) sets these up; `_start_follower_process()`/
`_stop_follower_process()` manage the child's lifecycle around `connect()`/
`disconnect()`.

## Finding 13: raw teleop targets sent straight to MIT-mode joints were
audibly jerky, fixed with S-curve smoothing

Independent of the timing/jitter findings above, forwarding the leader's raw
position every send tick was audible as jerk on MIT-mode joints: MIT mode
tracks a position/velocity/kp/kd setpoint directly, so a discontinuous jump in
target between ticks produces a discontinuous commanded velocity.

Fix: `_SCurveAxis` (`rebot_b601_follower.py`), 3 cascaded first-order
(exponential) filters per joint, applied to the goal position inside the
follower process before every send. An explicit accel/jerk-clamped trajectory
was tried first but chatters once the clamp saturates (a bang-bang/anti-windup
problem); cascaded first-order filters are monotonic and non-overshooting by
construction for any tick interval, at the cost of no hard-clamped peak
acceleration. Tunable via `smoothing_time_constant_s` (config default 0.08s
per joint) and disable-able via `enable_trajectory_smoothing=False` for direct
motor testing. Wired to `record.py`'s `--smoothing-time-constant` /
`--no-smoothing` flags.

## Finding 14: disconnect() used to stop the arm instantly; it now ramps home
first

`RebotB601Follower.disconnect()` used to stop the follower process (and, per
`disable_torque_on_disconnect`, cut torque) immediately wherever the arm
happened to be, e.g. mid-episode after a Ctrl-C. Since `record.py` already
ignores Ctrl-C during `disconnect()` (Finding 9), there was headroom to make
that call slower and safer instead of instant.

Fix: `disconnect()` now calls `go_home()`
(`rebot_b601_follower.py`) before stopping the follower process,
when `config.return_home_on_disconnect` is set (default true). It ramps every
joint from wherever it currently is to 0° (the calibration zero pose, the
same pose `calibrate()` asks the user to manually set with the gripper
closed) over a fixed `home_duration_s` (default 5s), using a smoothstep easing
curve so velocity is zero at both the start and end of the move. The target
sent each tick is an interpolation between the starting position and 0°, not
0° itself, so a joint that starts far from home doesn't move any faster than
one that starts close — every joint arrives in the same fixed time regardless
of distance. This still goes through the normal `send_action()` path (shared
memory) and the follower process's own S-curve smoothing (Finding 13) on top,
so it doesn't bypass either.

## Tooling produced

- `src/thesis/scripts/b601/tiered_teleop.py`: standalone tiered teleop reproduction, bypasses
  lerobot. `--tier 1..5`, `--seconds`, `--readback`.
- `src/thesis/scripts/b601/set_control_mode.py`: force + persist Damiao CTRL_MODE on all
  follower motors. Run automatically by `teleop.sh`.

## Where the fix lives

The minimal loop-speedup fix is applied on a branch of a lerobot submodule pinned
at `v0.6.0`, so it is version tracked and can be filed upstream. The fix is on the fork's
`b601-teleop-loop-fixes` branch, which is merged into `b601-dataset-robot-config`, the
branch the `external/lerobot` submodule tracks. See also the accompanying GitHub issue.
