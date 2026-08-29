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

The policy observation has 114 values. The first 79 are 26 normalized joint
positions, 26 previously applied physical joint targets, 26 joint velocities,
and phase. They are followed by palm pose in the robot-base frame (7), cube
pose relative to the palm (7), cube-center vectors from the five fingertips in
the palm frame (15), and cube linear/angular velocity relative to the palm (6).
The 26 actions use AnimRL's unbounded residual parameterization around the
first pose of the demonstration, with separate arm and hand residual scales.

The Gaussian joint imitation terms cover arm and hand separately. The object
is tracked at every reference sample with Gaussian center-position and
orientation rewards; object velocity is observed but is not rewarded. Early
termination is based on arm and hand joint drift and on the cube center staying
within `termination.object_position_threshold_m` of its reference target. All
three conditions use the configured consecutive-step grace period. Set
`termination.object_position_enabled=false` to disable only the cube-distance
condition while retaining the arm and hand early terminations.

The 114D network input is checkpoint-incompatible with the earlier 79D policy.
Those checkpoints must not be resumed for this object-training family.

Episodes last up to 300 control steps. Automatic RSI uses a configurable
mixture: 20% of resets sample the early approach (`0..649`) and 80% sample the
pre-grasp window (`650..830`). No automatic training or periodic-evaluation
reset starts after frame 830; successful policies can still reach the final
frame 1107 by continuing the episode. The relevant parameters are
`env.rsi_early_probability`, `env.rsi_pregrasp_start_index`, and
`env.rsi_max_start_index`.

The environment follows AnimRL's five-value vectorized step contract:

```python
observations, privileged_observations, rewards, dones, infos = env.step(actions)
```

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
periodic deterministic evaluation. W&B and video capture remain deliberately
deferred. The data is visible here:
`http://localhost:6006`

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
validates the 114D/26D environment contract, and writes
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
