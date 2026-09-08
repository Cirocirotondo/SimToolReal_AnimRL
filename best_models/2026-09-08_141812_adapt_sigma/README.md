# adapt_sigma — best policy by deployment score (0.4058)

**Use `best_deployment_model.pt`** (iteration 1000). It is selected by
`deployment_score`, not by `evaluation_score`, and it is the first checkpoint in
this project chosen automatically rather than reconstructed by hand afterwards.

## Why this one

| | base `blind512` | `blind_quiet2` @1600 | **this** |
|---|---|---|---|
| deployment score | 0.000 | 0.332 | **0.406** |
| rms action rate | 2.5275 | 0.1010 | **0.0537** |
| arm position error | 0.0692 | 0.0365 | 0.0360 |
| hand position error | 0.2124 | 0.0975 | 0.1220 |
| early termination (trained starts) | 0.000 | 0.000 | **0.000** |
| cube lift (trained starts) | 0.2137 | 0.1954 | **0.2170** |

**47x smoother than the base blind run**, with zero early termination and the
highest cube lift since the base run. The hand tracks slightly worse than
blind_quiet2; everything else is better.

## Deterministic replay (`eval_videos/`)

18.4 s, peak arm error **0.121 rad** (threshold 0.35), peak hand error
**0.232 rad** (threshold 1.35). Both limbs at roughly a third of their margins,
against blind_quiet2's half.

Mean rms action rate over the replay: **0.027 arm**, 0.163 hand. The hand is
where the remaining vibration lives, concentrated after reference frame ~690 --
see the note on the RSI seam below.

## What produced it

Four things stacked, each calibrated to the policy's own operating point rather
than copied:

1. **Asymmetric actor-critic** — the actor is blind (108D), the critic reads
   117D including fingertip contact forces. The critic is discarded at
   deployment, so this policy is genuinely blind.
2. **Softer hand drive** — `control.hand_stiffness_scale=0.5`. The largest
   single smoothness lever found.
3. **The sigma ladder** — position widths tightened in stages, recalibrated at
   each rung against measured error.
4. **Adaptive reward widths** — `rewards.adaptive_sigma_enabled=true`. Each
   Gaussian width follows a slow average of its own MSE and holds the term near
   reward 0.6, so it never saturates and never stops paying. This is what made
   the 21% jump over the previous best.

## Known limitations

- **First 2-3 seconds** (reference frames 1-179): arm error ~0.15 rad, 3-8x
  worse than the rest of the episode. `pregrasp_mixture` gives those frames
  about 4% of episode starts, so the policy barely trains there.
- **Jitter from frame ~690** (11.5 s): hand action rate jumps 54x at exactly
  the point where the RSI distribution's mass begins, and before any contact
  (correlation with contact is 0.08). It is a coverage seam, not a contact
  effect -- the policy stitches a barely-trained approach onto a heavily-trained
  grasp. Widening `rsi_pregrasp_start_index` below 740 is the untried fix.
- This run decayed after iteration 1000 (deployment 0.406 -> 0.209 by 2500)
  because the adaptive ratchet pinned a width to its all-time best and left that
  term unsatisfiable. Fixed by `adaptive_sigma_slack`; this checkpoint predates
  the decay.

## Before putting this on hardware

Trained with `control.hand_stiffness_scale=0.5`, so the hand PD gains in
`simtoolreal_animrl/envs/pd_gains.py` are **halved** relative to what the real
hand ships with. Halve the hardware's hand gains to match.

Lineage: `adapt_sigma <- asym_ladder <- blind_asym` (scratch, asymmetric critic).

`evaluations.jsonl` holds the run's unassisted evaluation rows; the full
per-iteration metrics, tensorboard events and intermediate checkpoints are
deliberately not committed.
