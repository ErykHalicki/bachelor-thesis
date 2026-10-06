# `backend: lerobot` eval knobs

Every key an `eval/` config may set for a real-robot rollout
(`src/thesis/experiments/eval/lerobot.py`).

## Rollout

| key | meaning |
| --- | --- |
| `robot` | lerobot `RobotConfig` as data (`defaults: - /embodiment@robot: b601`) |
| `episodes` | rollout count (default 10) |
| `fps` | control-loop rate (default 30) |
| `max_episode_steps` | per-episode step cap (default 600) |
| `home_between_episodes` | ramp a homing-capable robot to its zero pose between episodes (default true; ignored on robots without a homing ramp) |
| `quiet` | disable spoken announcements (default false) |
| `profile_log` | write a per-tick timing breakdown (obs/predict/send) to this path (default null) |
| `robot_recovery_attempts` | reconnect this many times when the robot drops out mid-episode, discarding and redoing the partial episode (default 3; 0 ends the session on the first drop) |
| `robot_recovery_delay_s` | wait before each of those attempts (default 2.0) |

## Model inputs and sampling

| key | meaning |
| --- | --- |
| `execute_len` | steps of each predicted chunk to execute before replanning; null keeps the trained horizon |
| `num_flow_steps` | Euler steps per replan; null keeps the trained value |
| `cfg_scale` | classifier-free guidance scale; null keeps the trained value |
| `image_size` | `[H, W]` resize for the model's visual inputs, which must match the dataset config the model trained on; null keeps native size |
| `columns` | `{model_field: dataset column}` remap, in the dataset backend's vocabulary. Default identity |
| `slices` | `{model_field: name patterns}` keeping only some `observation.state` dims, matching the dataset backend's `slices` |

## Rollout recording

Every kept episode is written to a LeRobotDataset under `record_root`; the per-camera mp4s
are concatenated into the video wandb gets. Encoder keys mirror `src/thesis/scripts/b601/record.py` --
lerobot's own libsvtav1 default costs far more CPU on the Pi.

| key | meaning |
| --- | --- |
| `record_dataset` | record the rollouts at all (default true) |
| `record_root` | where they land (default `outputs/rollouts/<timestamp>`) |
| `record_task` | task string stored per frame (default "eval rollout") |
| `resume` | a rollout dir to continue instead of starting a session: its scored episodes count toward `episodes`, new ones append to its dataset, and its wandb run is reopened. Overrides `record_root` |
| `record_vcodec` | camera codec (default h264) |
| `record_streaming_encoding` | encode as frames arrive rather than at episode end, which keeps `save_episode` near-instant at the cost of CPU during the episode (default true) |
| `record_encoder_threads` | threads per encoder; null lets the codec choose |
| `record_image_format` | `png` or `jpg` intermediates (default jpg); unused while streaming encoding is on |
| `record_jpeg_quality` | 0-100 for jpg intermediates (default 90) |
| `record_image_writer_threads` | null means one per camera |
| `record_image_writer_processes` | default 0 |

Resuming after a cooldown or a crash:

```
python main.py run=b601_irl load=<run_id> eval.resume=outputs/rollouts/20260819_111209
```

`<record_root>/eval_session.json` is rewritten after every kept episode, so a session
killed mid-episode still resumes with everything scored before it. It also holds the wandb
run id, and the resumed session reattaches to that run -- one run scores all `episodes`,
not one per sitting.

## Live stream

| key | meaning |
| --- | --- |
| `stream_cameras` | serve the cameras as one merged MJPEG stream during the rollout (default true). Costs a downscale + JPEG encode per streamed frame; failing to start one never stops an eval |
| `stream_port` | port the stream listens on (default 8090) |
| `stream_fps` | streamed frames per second, independent of the control fps (default 15) |
| `stream_height` | per-camera height before the panels are stacked (default 360) |
| `stream_quality` | stream JPEG quality, 0-100 (default 95) |
| `stream_metrics` | plot the executed action and observed state under the camera feed (default true). One plot per motor field (pos, torq, vel), derived from the robot's own feature names |
| `stream_metrics_fps` | rows pushed per second, subsampled off the control loop (default 10) |
| `stream_metrics_window` | rows kept, i.e. how much history a plot shows (default 300, so 30 s at 10 Hz) |
| `stream_latency` | chart what each replan cost in ms (default true): against a server, the local JPEG encode, the wire and the server's own compute as three lines; in-process, one compute line |

## Inference server

| key | meaning |
| --- | --- |
| `server` | `host:port` of an inference server (`src/thesis/scripts/serve.py`); null runs the model in-process |
| `run` | `<entity>/<project>/<run_id>` the server should serve. Defaults to the run named by `load=`; server mode only |
| `server_timeout` | socket timeout in seconds (default 120), covering the first request, which waits on the server's checkpoint download |
| `jpeg_quality` | camera-frame JPEG quality on the wire (default 95) |

## Session controls

Mirror `record.py` — the same keyboard backend, so arrow keys or `n`/`r`/`q` work over SSH,
and Ctrl-C routes to a graceful stop rather than unwinding mid-CAN-write.

| key | effect |
| --- | --- |
| Right / `n` | end the episode early (e.g. on success) and drop to the tagging prompt |
| Left / `r` | discard the episode and redo it (also available at the tagging prompt) |
| Esc / `q` | stop the eval after tagging the current episode |
