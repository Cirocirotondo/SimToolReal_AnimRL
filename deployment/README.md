# Real-robot deployment of AnimRL policies

Runs a trained AnimRL motion-imitation policy on the physical UR5e + Tesollo
DG5F. Simulation is the default; the arm and the hand are armed separately and
explicitly, so each rung of the ladder below is a flag change, never an edit.

The policy contract is **not** restated here. `build_observation` and
`actions_to_position_targets` are imported from `simtoolreal_animrl.sim2sim`,
so a policy sees the same observation on the robot that it saw in sim2sim.
MuJoCo is loaded only as a forward-kinematics engine for the palm and fingertip
poses that no encoder reports; it never integrates physics while the robot moves.

```
deployment/
  run_policy_real.py        the staged controller (this is what you run)
  animrl_deploy/
    kinematics.py           MuJoCo driven by measured joint angles, mj_forward only
    arm_client.py           ZMQ to impedance_controller (UR5e)
    hand_client.py          UDP peer of dg5f_policy_ros_bridge.py (DG5F)
    cube_source.py          demonstration / frozen / pose-estimation cube pose
    safety.py               rate limits, discontinuity detection, arming prompt
```

---

## 0. Two blockers to clear before anything is powered

### 0a. The observation contract must match the checkpoint

The repository migrated the palm/cube rotations from quaternions (**108-D**) to
the 6-D encoding (**112-D**). A checkpoint must be deployed against the
observation it was trained on; there is no adapter, and the two are not
interchangeable. `run_policy_real.py` refuses to start on a mismatch and prints
what to do.

`best_models/2026-09-07_003258_pg830_blind512_n256` is a **108-D** policy. The
in-progress 112-D work is uncommitted in the working tree, so deploying this
policy needs the committed (108-D) sim2sim checked out — `git stash` the 112-D
work, or use a worktree at that revision.

### 0b. MuJoCo must be pinned to 3.2.0–3.2.3

`simtoolreal_animrl/sim2sim/mujoco_sim.py` targets the MuJoCo 3.2.0–3.2.3 spec
API. Newer releases moved three things, and two of them fail *silently or
confusingly* rather than loudly:

| API | 3.2.0–3.2.3 | 3.3+ |
| --- | --- | --- |
| `MjSpec.from_file` | instance method, loads in place | **static factory that returns a new spec** |
| `discardvisual` | `spec.discardvisual` | `spec.compiler.discardvisual` |
| directional light | `light.directional = True` | `light.type = mjLIGHT_DIRECTIONAL` |

The `from_file` change is the dangerous one: `spec = MjSpec(); spec.from_file(p)`
leaves `spec` **empty**, so the model compiles with no joints and the failure
surfaces later as `unknown transmission target 'shoulder_pan_joint'`.

Neither interpreter on this machine works as-is (`simtoolreal_real/.venv` has
MuJoCo 3.9.0, `robohand/.venv` has 3.3.7), so sim2sim has never run here — the
committed `sim2sim_plots` came from the training machine. Create a pinned venv:

```bash
python3 -m venv ~/venvs/animrl-deploy
~/venvs/animrl-deploy/bin/pip install "mujoco==3.2.3" numpy pyzmq matplotlib tensorboard
~/venvs/animrl-deploy/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
```

CPU torch is enough — one 26-action forward pass per control step.

Pinning is deliberately preferred over porting the scene builder: that code
defines the kinematic chain, the palm offset and the joint limits that produce
the observation, so changing it changes the observation. With 3.2.3 a local
sim2sim rollout reproduces the committed training-machine rollout exactly (same
worst action jump, 2.4195 on `rj_dg_3_4`, at the same step).

---

## 1. Known hazard in the 108-D checkpoint

The 108-D observation canonicalizes quaternions to `w >= 0`, which cuts the
double cover at `w = 0`: a palm rotating smoothly through that plane negates all
four components at once and steps the observation by 2.0 while nothing physical
moves. This is measured, not theoretical, on
`2026-09-07_003258_pg830_blind512_n256`:

```
  ref_idx   palm_w   max|dAction|
      452   0.0032         0.0121
      453   0.0021         0.0121
      454   0.0010         0.0121
      455   0.0001         2.4195   <-- w reaches zero
      456   0.0136         1.1746
      457   0.0203         0.7136
      458   0.0229         0.9681
```

2.4195 action units on `rj_dg_3_4` is **0.363 rad — about 21° in one 60 Hz
tick** — followed by several steps of ringing. It recurs at every RSI index
tested (0, 60, 740), 14–25 steps per rollout exceed 1.0 action units, and the
p99.9 of all other steps is ~0.72.

The 6-D rotation encoding (112-D) is the actual fix. Until a 112-D policy is
available, this checkpoint should be run with `--spike-mode stop` (the default)
and the step limiter left on. Do not raise `--control-hz` to 60 with the
limiter relaxed.

---

## 2. Hardware bring-up

**Hand.** Bench supply 24 V / 10 A, ethernet, `tesollo` network profile
(169.254.186.0). Then the driver — note the topic requirement below:

```bash
cd /home/duplo/git/tesollo_ros2
source /opt/ros/humble/setup.bash && source install_dg5f/setup.bash
ros2 launch dg5f_driver dg5f_right_driver.launch.py
```

> Use `dg5f_right_driver.launch.py`, **not** `dg5f_right_pid_all_controller.launch.py`.
> The `pid_all` launch spawns `rj_dg_pospid`, which takes
> `control_msgs/MultiDOFCommand` on `/dg5f_right/rj_dg_pospid/reference`, while
> `dg5f_policy_ros_bridge.py` publishes `trajectory_msgs/JointTrajectory` on
> `/dg5f_right_controller/joint_trajectory` and subscribes to `/joint_states`.
> Wrong topic *and* wrong message type. With `pid_all` you must override
> `--joint-state-topic` / `--command-topic` and teach the bridge the other
> message type first.

**Hand bridge** (ROS 2 Humble's Python 3.10, a separate process from the policy):

```bash
cd /home/duplo/simone/SimToolReal/deployment/simtoolreal_real
source /opt/ros/humble/setup.bash && source /home/duplo/git/tesollo_ros2/install_dg5f/setup.bash
python3 dg5f_policy_ros_bridge.py
```

The bridge is reused unmodified and owns the last line of hand safety: joint
limit clipping, a 0.12 rad per-command step clamp, rejection of commands while
`/joint_states` is stale, and a measured-position hold when the policy stops.

**Arm.** Ethernet, `ur5` profile (192.168.1.10), Remote Control on the tablet:

```bash
cd /home/duplo/simone/SimToolReal/deployment/simtoolreal_real
./impedance_controller pc_ur_new.json
```

**Cube pose.** Not needed until stage 6 — stages 1–5 take the cube from the
demonstration.

---

## 3. The commissioning ladder

```bash
cd /home/duplo/simone/SimToolReal_AnimRL
PY=~/venvs/animrl-deploy/bin/python
CKPT=best_models/2026-09-07_003258_pg830_blind512_n256/best_model.pt
```

**Stage 1 — sim2sim with physics.** The real gate. Not this script: this script
has no physics, so a full-simulation run here is strictly weaker.

```bash
$PY scripts/run_mujoco_sim2sim.py --checkpoint $CKPT --rsi-index 0
```

**Stage 2 — dry run, no hardware.** Confirms the policy loads, the observation
builds and the monitors behave. Nothing is commanded.

```bash
$PY deployment/run_policy_real.py --checkpoint $CKPT --no-viewer --no-realtime
```

**Stage 3 — read hardware, command nothing.** Needs the low-level controller,
the DG5F driver and the bridge. Verifies both state streams and the FK.

```bash
$PY deployment/run_policy_real.py --checkpoint $CKPT \
  --use-real-arm-state --use-real-hand-state --max-steps 200
```

**Stage 4 — hand alone, stepped and slow.** The arm stays simulated on the
demonstration trajectory. Two confirmations: a typed `SEND`, then Space.

```bash
$PY deployment/run_policy_real.py --checkpoint $CKPT \
  --send-to-hand --debug-step --control-hz 1 --hand-action-scale 0.2 --max-steps 10
```

**Stage 5 — arm alone**, hand simulated:

```bash
$PY deployment/run_policy_real.py --checkpoint $CKPT \
  --send-to-arm --debug-step --control-hz 1 --arm-action-scale 0.2 --max-steps 10
```

**Stage 6 — both, stepped**, then both free-running slow, then raise the rate:

```bash
# stepped
$PY deployment/run_policy_real.py --checkpoint $CKPT --send-to-arm --send-to-hand \
  --debug-step --control-hz 1 --arm-action-scale 0.2 --hand-action-scale 0.2 --max-steps 10

# free-running, slow, smoothed, short
$PY deployment/run_policy_real.py --checkpoint $CKPT --send-to-arm --send-to-hand \
  --control-hz 5 --arm-action-scale 0.3 --hand-action-scale 0.3 \
  --target-smoothing 0.6 --max-steps 100

# then, one change at a time: control-hz 5 -> 10 -> 20 -> 60,
# then action scales 0.3 -> 0.5 -> 1.0, then smoothing 0.6 -> 0.
```

**Stage 7 — live cube.** Verify the estimator frame *before* trusting it. The
demonstration and the estimator must agree; a sign flip means they do not.

```bash
# terminal A -- the estimator, in its own environment
cd /home/duplo/git/robohand/src/tag-pose-estimation
/home/duplo/git/robohand-robohand2/.venv/bin/python scripts/run_pose_estimation.py \
  --config config/pose_estimation_configs/parallelepiped_5x5x15cm_robot_frame.json

# terminal B -- check the frame agrees with the demonstration, then run
$PY deployment/run_policy_real.py --checkpoint $CKPT --check-cube-frame --rsi-index 0
$PY deployment/run_policy_real.py --checkpoint $CKPT --send-to-arm --send-to-hand \
  --cube-source pose-estimation
```

---

## 4. Safety monitors

| Monitor | Default | What it does |
| --- | --- | --- |
| `--spike-mode` | `stop` | Aborts on a single-step action jump above `--max-action-step` (1.0). This is the quaternion-crossing detector of section 1. |
| `--spike-grace-steps` | `2` | Reports but does not abort during the policy's settling transient, which legitimately exceeds 1.0 at step 1. |
| `--reference-error-mode` | `stop` | Aborts when the robot leaves the demonstration by more than training's own termination thresholds (arm 0.35 rad, hand 1.35 rad, read from the run config). Measured against the *reference*, not the target: a soft-PD finger sits up to 1.4 rad from its target quite legitimately, so target error says little. |
| `--max-arm-step-rad` / `--max-hand-step-rad` | `0.02` / `0.05` | Hard per-tick clamp on the commanded target. The reference rollout's p95 step is 0.007 (arm) / 0.004 (hand), so normal motion passes and only outliers are clipped. |
| `--target-smoothing` | `0` | EMA on the target. Use 0.5–0.7 on the first free-running rungs, then return to 0. |
| `--current-warning-ma` | `170` | Reports DG5F motor current above the driver's own limiter threshold. |

Ctrl+C, any abort, and any exception all brake the arm with a zero-velocity
hold and park the hand at its measured position.

Two flags change what the policy *sees* and are worth understanding:

- `--simulated-state-source` (default `demonstration`) — state fed back for a
  subsystem not read from hardware. `demonstration` replays the reference and
  its recorded velocities, reproducing the training observation. `target`
  assumes the subsystem tracks its own command; with no physics behind it a
  joint would traverse a whole step per tick and report tens of rad/s, far
  outside anything training showed the network, so it is velocity-clamped. Even
  clamped it drifts open-loop over a long rollout and will trip the monitors;
  it is a short-run diagnostic, not a substitute for `demonstration`.
- `--previous-target-source` (default `raw`) — `raw` reproduces sim2sim exactly;
  `applied` reports what the safety limiter actually sent. They differ whenever
  the limiter is active, which on the early rungs is most of the time.

---

## 5. What has and has not been tested

Verified on this machine, without robots:

- sim2sim under MuJoCo 3.2.3 reproduces the committed training-machine rollout.
- Full 1047-step deployment rollout, demonstration-driven state, monitors live.
- Arm and hand command paths against fake endpoints: homing trajectory, 100 Hz
  target streaming, UDP hand targets, state feedback, braking and hold on exit.
- The contract guard, the spike monitor and the reference-deviation monitor
  each firing on the conditions they are meant to catch.

**Not** tested: anything against real hardware, and `--cube-source
pose-estimation`, whose frame equivalence with the demonstration is assumed and
must be confirmed with `--check-cube-frame` before use.
