# blind_quiet2 — best blind policy for the real robot

**Use `model_1600.pt`.** Chosen by Simone from the evaluation videos and the
per-joint action plots.

**Not `best_model.pt`**: that file holds the *iteration-0* weights. Its
`evaluation_score` peaked at iteration 0 and the run never beat it, so the file
never updated. `evaluation_score` is computed on the fixed evaluation cohort,
three of whose four start phases lie outside the window `pregrasp_mixture`
trains, and it rescales whenever a reward sigma changes -- so it did not track
this run's real progress. See `simtoolreal_animrl/runners/deployment_score.py`
for the metric that does, and `best_deployment_model.pt` in newer runs.

## Where iteration 1600 sits

Periodic evaluations run every 500 iterations, so 1600 has none of its own.
Its neighbours, ranked by `deployment_score`:

| iteration | deployment | uniform ET | uniform lift | arm err | hand err | rms action rate |
|-----------|-----------|-----------|--------------|---------|----------|-----------------|
| 0         | 0.277     | 0.047     | 0.185        | 0.0335  | 0.0830   | 0.1845          |
| 500       | 0.175     | 0.109     | 0.192        | 0.0670  | 0.1091   | 0.2519          |
| 1000      | 0.288     | 0.016     | 0.198        | 0.0725  | 0.0906   | 0.0792          |
| **1500**  | **0.332** | **0.000** | 0.195        | 0.0365  | 0.0975   | 0.1010          |
| 2000      | 0.259     | 0.062     | 0.194        | 0.0761  | 0.0996   | 0.0862          |
| 2500      | 0.235     | 0.141     | 0.185        | 0.0705  | 0.0970   | 0.0744          |

1500 is the scored peak; 1600 is 100 iterations past it and before the decline
at 2000. The deterministic replay in `eval_videos/` is of `model_1500`.

## Deterministic replay (`eval_videos/eval_model_1500_rsi_0.mp4`)

- peak cube lift **0.246 m** (the demonstration lifts 0.24)
- peak arm joint error **0.168 rad**, threshold 0.35
- peak hand joint error **0.459 rad**, threshold 1.35

Both limbs sit at roughly half their termination margins.

## Before putting this on hardware

Trained with `control.hand_stiffness_scale=0.5`, so the hand PD gains in
`simtoolreal_animrl/envs/pd_gains.py` are **halved** relative to what the real
hand ships with. Halve the hardware's hand gains to match, or the policy will
command into a drive twice as stiff as the one it learned on.

## What is in this folder

`model_1600.pt` and `config.json` are what you need to run the policy.
`evaluations.jsonl` holds the six unassisted evaluation rows from the run (the
full 2931-line `metrics.jsonl`, the tensorboard events and the raw per-step
arrays are deliberately not committed -- git keeps them forever and the plots in
`episode_00/` are their readable form). `checkpoint_source.json` records which
run these weights were warm-started from.

Lineage: `blind_quiet2 <- blind_soft <- blind_sharp <- blind_track2 <- blind_track
<- pg830_blind512`. Blind throughout: 108D observation, no contact sensing.
