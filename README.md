# SimToolReal AnimRL

Minimal UR5e + Tesollo DG5F motion-imitation environment using the training
structure and Python configuration style of `cmm-25-a3-animrl`.

This repository intentionally does not depend on `rl_games`.  The first
milestone is a headless reference-action test; PPO will be added only after the
environment is verified independently.

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
corrected DG5F gains, resets uniformly over the complete motion, executes ideal
normalized arm reference actions while driving the hand directly from the demo,
and validates tracking, arm-only reward, arm-only early termination, and the
Cartwheel-style reference-end timeout.

The arm-only policy observation has 19 values: 6 normalized arm positions, 6
previously applied physical arm targets, 6 arm velocities, and the normalized
reference phase. Its 6 normalized actions are absolute UR5e joint targets. The
20 hand targets are read directly from the next demonstration sample inside the
environment, so the policy neither observes nor controls the hand.

The Gaussian position/velocity imitation reward and early termination use only
the six arm joints. Hand tracking errors remain available as diagnostics but do
not affect reward or termination.

This replaces the former 79D/26D full-robot policy contract. Checkpoints made
with that earlier architecture cannot be loaded by the 19D/6D networks and must
not be resumed for this training family.

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

Checkpoints use AnimRL's `model_<iteration>.pt` schema. Resume for an additional
number of PPO updates with:

```bash
/home/simone/.venv/bin/python scripts/train.py \
  --num-envs 64 \
  --iterations 2 \
  --resume logs/simtoolreal/<run>/model_2.pt
```

Each run stores `config.json`, `metrics.jsonl`, and AnimRL-compatible models.
TensorBoard/WandB, periodic evaluation, and video capture remain deliberately
deferred.

## Deterministic checkpoint evaluation

Evaluate the policy mean (the deterministic action used by original AnimRL)
from reference sample zero with one headless environment:

```bash
/home/simone/.venv/bin/python scripts/evaluate.py \
  --checkpoint logs/simtoolreal/<run>/model_3000.pt
```

The evaluator automatically loads `config.json` from the checkpoint directory,
validates the 19D/6D arm-only environment contract, and writes
`eval_model_3000.json`. To display the same rollout in Isaac Gym:

```bash
/home/simone/.venv/bin/python scripts/evaluate.py \
  --checkpoint logs/simtoolreal/<run>/model_3000.pt \
  --viewer
```

Use `--uniform-rsi --seed <N>` to evaluate with the training RSI distribution,
or `--rsi-index <sample>` for a fixed reproducible initial state.

## Isaac Gym demonstration viewer

To open the exact same environment configuration used by training and replace
the six policy outputs with the normalized arm action from the next
demonstration sample (the hand remains reference-driven), run:

```bash
/home/simone/.venv/bin/python scripts/demo_viewer_isaacgym.py
```

It uses one environment and uniform RSI by default. For deterministic playback
from a particular demonstration sample and for one episode only:

```bash
/home/simone/.venv/bin/python scripts/demo_viewer_isaacgym.py \
  --rsi-index 732 \
  --episodes 1
```
