# MuJoCo sim2sim

This package runs the blind 108-observation, 26-action AnimRL policy in a
standalone MuJoCo model of the UR5e, right DG5F hand, table, and cuboid. It does
not import the sibling `simtoolreal` repository and it does not require Isaac
Gym at runtime.

The robot uses gravity compensation, matching `asset.disable_gravity = true`
during training. The cuboid retains normal gravity and no object-assist wrench
is applied, matching the saved blind run.

Run the default checkpoint with a viewer:

```bash
/home/simone/.venv/bin/python scripts/run_mujoco_sim2sim.py
```

Run headless without real-time pacing:

```bash
/home/simone/.venv/bin/python scripts/run_mujoco_sim2sim.py \
  --headless --no-realtime
```

The runner accepts `--rsi-index`, `--max-steps`, and MuJoCo-specific arm/hand
PD overrides. An RSI state that already contains hand/cube contacts receives a
0.1 s fixed-cube/fixed-arm contact warm start by default; pass
`--contact-settle-seconds 0` to inspect the raw reset. Use `--help` for the
complete list.

With the viewer enabled, the scene pauses for two seconds before movement and
shows a green, collision-free robot 0.8 m to the side. That robot is updated
kinematically from the demonstration while the original robot is controlled by
the policy. Use `--start-delay-seconds` to change the pause or `--no-ghost` to
hide the reference robot.

At the end of a rollout, four figures compare policy/reference/action-delta
signals and measured/target/reference joint angles, separately for the arm and
hand. The plots and their raw `rollout_data.npz` are saved under
`sim2sim_plots/rsi_NNNN` beside the checkpoint. With the viewer enabled the
figures also open interactively after the MuJoCo window closes. Pass
`--plot-dir` to select another folder, `--no-show-plots` to save without opening
plot windows, or `--no-plots` to disable both saving and display.

Collision filtering matches the Isaac Gym task: robot self-collision and
robot/table collision are disabled, the arm through `wrist_2_link` is filtered
from the cube, the `wrist_3_link`/hand assembly can contact the cube, and the
cube can contact the table. The ground plane is visual only.

Run the focused tests with:

```bash
/home/simone/.venv/bin/python -m unittest tests.test_sim2sim -v
```
