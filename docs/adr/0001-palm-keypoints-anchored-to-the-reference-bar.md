# Palm keypoints are anchored to the reference bar, fingertips to the measured bar

The palm keypoint reward (weight 0.80, the heaviest term) measured the palm in
the frame of the bar *as measured*, which makes it a statement about hand-bar
relative geometry alone. During Transport the grasp fixes that geometry, so the
term became blind to whether the bar was being lifted at all and simultaneously
emitted a constant, unsatisfiable gradient for whatever grasp offset the episode
happened to end up with -- at a weight that drowned out every term asking for
the lift. We therefore anchor the palm half to the bar's *reference* pose, so
its target rises with the reference bar, while the fingertip half (weight 0.48)
stays anchored to the measured bar.

This deliberately overrides the reasoning recorded in `_hand_keypoints_cube_frame`'s
docstring ("anchored on the cuboid's measured pose [...] so the term still points
the right way when the bar has been nudged"). That property is real and we are
giving it up for the palm only: the fingertips retain it, and they are the half
for which relative geometry is the actual subject matter.

## Considered options

**Hand-truth (chosen).** Reward the hand for being where the demonstration's
hand was. The target is known offline from the transform bank, so this is a
change of argument at one call site.

**Cube-truth (rejected).** Reward the hand for putting the bar where the
demonstration's bar was. More correct in principle: it is the task objective,
and it is what the project cares about. Rejected on cost. Expressing it as a
palm target requires knowing the grasp offset, which exists only at runtime and
differs per episode, so it needs either an event plus per-environment state
(freeze the offset at lift-off) or a new reward term recomputing the rigid
correction every step. The two differ only by the grasp offset itself -- a
couple of centimetres -- against a success criterion that tolerates 0.07 m.

**Drop the palm term during Transport (rejected).** The cheap route to
cube-truth: gate the misleading term off and let the existing bar pose rewards
(0.8 and 0.4) carry the lift. Rejected because it cures the bad gradient
without supplying a good one, and those two terms were measured weak in exactly
that phase (see the calibration comments on `object_position_std_m`).

## Consequences

- `termination.palm_keypoint_threshold_m` (0.20 m) now fires on absolute
  deviation from the demonstrated palm trajectory rather than on relative
  geometry. This is load-bearing and welcome: under the old anchor a policy that
  never lifted kept a *low* palm error and so was never terminated, which made
  "survive without lifting" a viable strategy. It no longer is.
- Absolute return is not comparable across this change.
- The symmetry element (`symmetry_index`) must be applied consistently across
  the two anchors. The reference orientation already arrives in the
  demonstration's labelling, so it likely needs no symmetry element at all,
  where the measured one does. This is the single most likely silent bug in the
  change and is what the open-loop playback test exists to catch.
