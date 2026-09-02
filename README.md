# SimToolReal AnimRL

Minimal UR5e + Tesollo DG5F motion-imitation environment using the training
structure and Python configuration style of `cmm-25-a3-animrl`.

This repository intentionally does not depend on `rl_games`. It contains a
small AnimRL-compatible PPO stack plus an independently testable Isaac Gym
environment.

## Current training status

Object tracking, configurable object-distance termination, pre-grasp RSI and
optional fingertip-contact shaping are implemented and covered by software
tests. The grasp-training experiments run so far still do not learn a reliable
grasp, so this training strategy remains experimental and needs further tuning.

## Headless environment test

From the repository root:

```bash
/home/simone/.venv/bin/python scripts/test_headless_env.py --num-envs 16
```

The production configuration uses 4096 environments, matching AnimRL's Walk
and Cartwheel configurations:

```bash
/home/simone/.venv/bin/python scripts/test_headless_env.py
```

The test loads the local robot asset and 60 Hz demonstration, applies the
corrected arm/DG5F gains, resets the robot and physical cuboid with the
configured early/pre-grasp RSI mixture, executes ideal AnimRL residual actions, and validates
tracking, reward, early termination, object physics, collision filtering, and
the Cartwheel-style reference-end timeout.

The policy observation has 108 values. The first 79 are 26 normalized joint
positions, 26 previously applied physical joint targets, 26 joint velocities,
and phase. They are followed by palm pose in the robot-base frame (7), the five
fingertip positions relative to the palm center in the palm frame (15), cube
orientation relative to the palm (4), and cube center relative to the palm (3).
The 26 actions use AnimRL's unbounded residual parameterization around the
first pose of the demonstration, with separate arm and hand residual scales.

The Gaussian reward weights and widths and the arm/hand early terminations match
the `2026-08-28_173516_no_object_reward/model_7500.pt` run. The action-delta
terms differ intentionally: they track the demonstrated displacement using
`(a_t-a_{t-1}) - dq_demo*dt/action_scale`, instead of rewarding zero action
change. Cube-pose, fingertip-proximity and contact rewards are disabled, as is
cube-distance early termination. The cube remains part of the observation.

The 108D network input is checkpoint-incompatible with both the earlier 79D
policy and the 114D `model_7500.pt`; this experiment must start from scratch.

### Robustness and object-reward experiments

`scripts/run_robustness_object_reward.py` launches seven independent scratch
trainings of the current 108D experiment, with 12000 PPO updates each. The first
group runs the no-object-reward baseline with seeds `42, 43, 44`. The second
group uses seed `42` for all four runs and multiplies the cube-position and
cube-orientation reward weights by `1.0, 0.8, 0.6, 0.4`. Keeping one seed fixed
in the second group isolates the reward-scale comparison. Fingertip proximity
and contact rewards remain disabled in both groups.

Runs are stored below one timestamped `*_robustness_object_reward` directory
and summarized in `experiment_sweep.json`. A failed or diverged run is recorded
but does not stop the remaining experiments.

```bash
/home/simone/.venv/bin/python scripts/run_robustness_object_reward.py
```

Use `--dry-run` to inspect the seven generated `train.py` commands without
starting Isaac Gym. `--robustness-seeds`, `--object-scales`, `--object-seed`,
`--iterations`, `--num-envs`, video/evaluation switches, and repeated
`--set PATH=VALUE` overrides are also available.

### No-object entropy sweep

`scripts/run_no_object_entropy_sweep.py` launches five independent scratch
replicas of the no-object-reward robustness training. Every run uses seed `42`
and 12000 PPO updates; the only varied setting is `entropy_coef`, tested at
`0.005, 0.002, 0.001, 0.0005, 0.0`. The already-tested divergent
`entropy_coef=0.01` case is intentionally omitted. Cube position, cube
orientation, fingertip proximity, and contact rewards are all explicitly
disabled.

```bash
/home/simone/.venv/bin/python scripts/run_no_object_entropy_sweep.py
```

Runs are stored below a timestamped `*_no_object_entropy_sweep_seed_42`
directory and summarized in `entropy_sweep.json`. A failed or diverged run does
not stop the remaining independent experiments. Use `--dry-run` to inspect all
five commands without starting Isaac Gym.

### Pre-grasp proximity training

`scripts/run_pregrasp_proximity.py` launches one independent scratch training
whose RSI starts are uniformly restricted to the pre-grasp frames `740..830`.
It uses the current 108D observation and demonstration-aware action-delta
reward, restores the parent `object` branch's cube-position (`0.8`),
cube-orientation (`0.2`), and thumb/index/middle proximity (`0.2`) reward
weights, and keeps contact shaping off. Cube-distance early termination is
enabled at the configured `0.05 m` threshold with a five-step grace period.

```bash
/home/simone/.venv/bin/python scripts/run_pregrasp_proximity.py
```

The default run has 12000 PPO updates, 200-step episodes, and uses seed `43`,
the stable seed from the preceding robustness experiment. Use `--dry-run` to
inspect the exact command; the iteration count, seed, reward weights,
evaluation/video options, number of environments, and output directory are
configurable.

`scripts/run_pregrasp_entropy_sweep.py` repeats that exact scratch experiment
five times with seed `43`, changing only `entropy_coef` across `0.005, 0.002,
0.001, 0.0005, 0.0`. The already-tested `0.01` run is omitted. Every run is
independent, and a failure or divergence does not stop the remaining values:

```bash
/home/simone/.venv/bin/python scripts/run_pregrasp_entropy_sweep.py
```

The timestamped output directory contains one subdirectory per coefficient and
an `entropy_sweep.json` manifest. Use `--dry-run` to inspect the commands.

### Object-reward hyperparameter sweep

`scripts/run_object_reward_sweep.py` runs independent experiments with the
contact reward disabled. It multiplies all current object-related reward
weights (position, orientation, and fingertip proximity) by
`0.1, 0.2, ..., 1.0`. Every `warm` experiment starts directly from the same
known-good no-object `model_7500.pt`; experiments never load checkpoints from
other scales. With the default policy warm start, only the actor and observation
normalizers are loaded, while the critic and optimizer start fresh. Every
`scratch` experiment starts with new random networks. A failed or diverged run
is recorded but does not stop or initialize any of the remaining experiments.
The generated `sweep.json` records commands, weights, checkpoints, and outcomes.

Run both initialization variants, with 1000 PPO updates at each of the ten
scales:

```bash
/home/simone/.venv/bin/python scripts/run_object_reward_sweep.py
```

Use `--dry-run` to inspect every command without starting Isaac Gym. The run
length, scale list, initialization variants, environment count, seed, devices,
and output root are command-line options of the sweep launcher; run it with
`--help` for the complete list. Pass `--warm-start-mode full` when every warm
experiment must independently resume actor, critic, optimizer, normalizers,
and counters from the source checkpoint rather than performing the default
policy-only initialization.

Episodes last up to 300 control steps. Automatic RSI is uniform over
`0..830`, so every reachable start frame is equally likely. No automatic
training or periodic-evaluation reset starts after frame 830; successful
policies can still reach the final frame 1107 by continuing the episode.
Setting `env.reference_init_distribution` to `pregrasp_mixture` selects the
skewed alternative instead, where `env.rsi_early_probability` of the resets
sample the early approach (`0..739`) and the rest sample the pre-grasp window
(`env.rsi_pregrasp_start_index..env.rsi_max_start_index`). That mixture
concentrates 80% of resets on 91 of the 831 reachable frames, which trains the
grasp from an RSI reset but leaves the approach too rarely visited to be
chained from frame 0. Both distributions obey `env.rsi_max_start_index`.

Cube lifting is diagnostic rather than rewarded. TensorBoard's `Episode/`
group reports the mean and maximum of the per-episode peak cube centre-of-mass
height, plus the corresponding lift relative to each episode's RSI reset
height. Periodic deterministic evaluation exposes the same peak metrics under
`Evaluation/`. The standalone evaluator stores the raw height/lift time series
in `evaluation_log.npz` and writes `object_com_height.png`, comparing the
physical cube with the demonstrated cube trajectory.

### Fingertip-to-cube proximity

`scripts/evaluate.py` writes `fingertip_proximity.png`, the term the proximity
reward actually consumes. The upper panel shows the point-to-box surface
distance of every finger in `rewards.fingertip_object_distance_names`, with the
mean, the 1sigma and 2sigma lines of
`rewards.fingertip_object_distance_std_m`, and a grey band over the steps where
the term is gated off before `env.rsi_pregrasp_start_index`. The lower panel
shows each finger's Gaussian next to the gated term and its weighted
contribution.

The reward averages one Gaussian per selected finger, so its scalar cannot
distinguish a hand closing evenly from one finger reaching the cube while the
others trail. The per-finger curves can. Distance zero means touching the
cuboid surface, not the centre. The raw `(steps, F)` array is stored in
`evaluation_log.npz` as `fingertip_object_distance_per_finger_m`, with
`proximity_fingertip_names` and `proximity_std_m`.

Unlike the contact figure below, this one needs no PhysX contact reporting: it
is pure geometry and is always produced.

### Fingertip contact forces

`scripts/evaluate.py` writes `fingertip_forces.png`: the net contact force of
every fingertip over the episode, with the `contact.force_threshold_n` line and
a raster of which rewarded fingers are above it. Fingers outside
`contact.fingertip_names` are drawn dashed and labelled "not rewarded". The raw
`(steps, 5)` array is saved in `evaluation_log.npz` as `fingertip_force_n`,
alongside `fingertip_names` and `contact_force_threshold_n`.

This separates what `Contact/mean_fingertip_force_n` averages away: that scalar
divides by every selected finger, those touching nothing included, so one
finger pressing at 3 N reads the same as three pressing at 1 N.

Isaac Gym reports one net force vector per rigid body, so each curve aggregates
every contact on that fingertip — cube and table alike. It is not a per-pair
fingertip-to-cube force.

The figure needs PhysX contact reporting, which is a construction-time
decision. When a run was trained with `contact.enabled=false` the tensor is
never acquired and the figure is skipped rather than drawn flat at zero. To
inspect the forces anyway:

```bash
/home/simone/.venv/bin/python scripts/evaluate.py \
  --checkpoint logs/simtoolreal/<run>/model_<n>.pt \
  --contact-forces
```

That flag enables reporting and sets `contact.reward_per_finger` to zero, so
the measured return stays comparable with training.

The principal TensorBoard series are
`Episode/mean_peak_object_com_height_m`,
`Episode/max_peak_object_com_height_m`,
`Episode/mean_peak_object_com_lift_m`, and
`Episode/max_peak_object_com_lift_m`. Evaluation series have the same suffixes,
prefixed by `Evaluation/fixed_` or `Evaluation/uniform_`.

The environment follows AnimRL's five-value vectorized step contract:

```python
observations, privileged_observations, rewards, dones, infos = env.step(actions)
```

External collision filtering keeps robot-table contacts disabled everywhere,
and UR5e-arm-to-cube contacts disabled up to `wrist_2_link`. Cube contacts
remain enabled for the whole hand assembly and for the table. Because fixed
joints are collapsed, the static DG5F mount/base/palm collision shapes belong
to `wrist_3_link`, so that body is grouped with the articulated `rl_dg_*`
fingers rather than with the arm: otherwise the palm could not support a grasp.
Wrist 3 therefore also contacts the cube through its own mesh, which sits about
0.099 m behind the flange while the palm sits about 0.074 m in front of it.

`infos["time_outs"]` is true for the configured horizon and for the end of the
reference motion, allowing PPO to bootstrap those transitions. It stays false
for imitation early termination. When one or more environments finish,
`infos["episode"]` contains scalar aggregate return, reward/error, episode
length, and termination-type statistics for AnimRL logging.

## Controlled PPO update

The repository includes the checkpoint-compatible AnimRL policy, value,
normalizer, diagonal Gaussian distribution, rollout storage, and PPO training
loop. Run one complete 24-step rollout followed by the configured five PPO
epochs and four minibatches with:

```bash
/home/simone/.venv/bin/python scripts/test_ppo_update.py --num-envs 64
```

The test checks finite rollout tensors and losses, normalized GAE advantages,
nonzero actor/critic parameter updates, fixed learning rate, and deterministic
save/reload using AnimRL's checkpoint keys.

Start production training with the AnimRL Cartwheel PPO settings (4096 envs,
24-step rollouts, five learning epochs, four minibatches, fixed `1e-4` learning
rate) using:

```bash
/home/simone/.venv/bin/python scripts/train.py
```

For a short smoke run:

```bash
/home/simone/.venv/bin/python scripts/train.py \
  --num-envs 64 \
  --iterations 2 \
  --save-interval 1 \
  --run-name smoke
```

The optional thumb/index/middle contact reward is disabled by default. Enable
the GPU contact tensor and add `0.05` per contacting fingertip at each step
with:

```bash
/home/simone/.venv/bin/python scripts/train.py \
  --set contact.enabled=true
```

Its force threshold, reward and PhysX collection mode remain independently
overridable, for example:

```bash
--set contact.force_threshold_n=1.0 \
--set contact.reward_per_finger=0.02 \
--set contact.collection=1
```

Set `contact.enabled=false` to restore `CC_NEVER`; the contact tensor is then
neither acquired nor refreshed.

Checkpoints use AnimRL's `model_<iteration>.pt` schema. Resume for an additional
number of PPO updates with:

```bash
/home/simone/.venv/bin/python scripts/train.py \
  --num-envs 64 \
  --iterations 2 \
  --resume logs/simtoolreal/<run>/model_2.pt
```

Each run stores `config.json`, `metrics.jsonl`, TensorBoard event files, and
AnimRL-compatible models. Follow a running experiment locally with:

```bash
/home/simone/.venv/bin/tensorboard --logdir logs/simtoolreal
```

TensorBoard records reward components, actor/critic losses, policy standard
deviation, action clipping, arm tracking errors, termination fractions, and
throughput, plus fingertip contact count/fraction/force when enabled and
periodic deterministic evaluation. The data is visible here:
`http://localhost:6006`

Training video is opt-in because an off-screen Isaac Gym camera still requires
a graphics-capable device. Add `--record-video` to `train.py` or to the object
reward sweep launcher to point a 640x480 camera at environment 0. Every 500 PPO
iterations it records 600 consecutive control frames (10 seconds at 60 Hz),
spanning 25 standard 24-step rollouts, into
`videos/training_env_00_iteration_<n>.mp4`. This is the actual stochastic
training environment: natural early resets may therefore appear in the video.
TensorBoard's `Video/training_environment` text series records the saved path.

Video is disabled by default. `--no-record-video` explicitly forces the
original compute-only headless path with `graphics_device_id=-1`, so servers
without display/graphics support do not initialize a graphics context.
The optional encoder dependencies can be installed with
`pip install -e '.[video]'`; they are imported only when recording starts.

Periodic evaluation runs in a separate headless environment every 100 PPO
updates by default. It evaluates the deterministic policy mean on both fixed
RSI phases (`0`, `0.25`, `0.5`, `0.75`) and a uniformly sampled RSI cohort
that is reconstructed from the same dedicated seed at every evaluation.
`best_model.pt` is selected using the mean fixed/sampled position score, with early-terminated
episodes penalized by subtracting their fraction from the mean position reward.
This preserves a useful ranking while all early policies still fail, and makes
any fully successful cohort outrank a fully failed one. Periodic
`model_<iteration>.pt` checkpoints are retained independently.

Override the evaluation cadence and size without editing configs:

```bash
python scripts/train.py --eval-interval 50 --eval-num-envs 128
```

Use `--no-periodic-eval` for short profiling/debug runs.

## Deterministic checkpoint evaluation

Evaluate the policy mean (the deterministic action used by original AnimRL)
from reference sample zero with one headless environment:

```bash
/home/simone/.venv/bin/python scripts/evaluate.py \
  --checkpoint logs/simtoolreal/<run>/model_3000.pt
```

The evaluator automatically loads `config.json` from the checkpoint directory,
validates the 108D/26D environment contract, and writes
`eval_model_3000.json`. To display the same rollout in Isaac Gym:

```bash
/home/simone/.venv/bin/python scripts/evaluate.py \
  --checkpoint logs/simtoolreal/<run>/model_3000.pt \
  --viewer
```

Use `--sampled-rsi --seed <N>` to evaluate with the training RSI distribution,
or `--rsi-index <sample>` for a fixed reproducible initial state.

## Isaac Gym demonstration viewer

To open the exact same environment configuration used by training and replace
the six policy outputs with the ideal AnimRL residual action for the next
demonstration sample (the hand remains reference-driven), run:

```bash
/home/simone/.venv/bin/python scripts/demo_viewer_isaacgym.py
```

It uses one environment and the configured RSI mixture by default. For deterministic playback
from a particular demonstration sample and for one episode only:

```bash
/home/simone/.venv/bin/python scripts/demo_viewer_isaacgym.py \
  --rsi-index 732 \
  --episodes 1
```
