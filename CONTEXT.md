# SimToolReal AnimRL

Training a UR5e arm plus Tesollo DG5F hand in Isaac Gym to imitate a
demonstration: approach a bar resting on a table, grasp it, and carry it
along the demonstrated trajectory.

## Language

### Episode phases

**Approach**:
The part of an episode before the bar leaves the table. The hand is free to
move relative to a stationary bar, so relative geometry is fully correctable.

**Transport**:
The part of an episode after the bar leaves the table. The grasp constrains
the hand-bar relative pose, so what remains to be controlled is where the
bar goes.
_Avoid_: lift, carry phase

### Reward geometry

**Anchor**:
The frame in which keypoint tracking error is measured. Choosing an anchor
is choosing what a keypoint reward is a statement about.

**Measured-cube anchor**:
Keypoints expressed in the frame of the bar as it actually is. States a fact
about the hand's pose relative to the bar.

**Reference-cube anchor**:
Keypoints expressed in the frame of the bar as the reference says it should
be. States a fact about the hand's pose in the world.

**Frozen error**:
Tracking error a reward keeps charging for after the system has made it
uncorrectable. A measured-cube-anchored palm term during Transport is the
canonical case: the grasp fixes the relative pose, so the term emits a
constant gradient the policy cannot satisfy without releasing the bar.

**Hand-truth**:
The resolution of an imperfect grasp that sends the hand where the reference
hand went, letting the bar land wherever the grasp offset carries it. The
project's chosen resolution: it costs only the grasp offset itself, which is
smaller than the tolerance the success criterion allows.

**Cube-truth**:
The opposite resolution: the bar goes where the reference bar went, and the
hand must then be somewhere the reference never was. Rejected, because
expressing it as a palm target requires measuring the grasp offset at
runtime.

**Grasp offset**:
The rigid hand-bar relative pose actually achieved, as opposed to the one the
reference demonstrates.

### Division of labour

**Fingertip keypoints price the grasp**:
The five fingertip keypoints stay measured-cube-anchored in every phase.
Relative geometry is their subject matter, and the fingers can still adjust
it during Transport.

**Palm keypoints price the transport**:
The four palm keypoints describe where the bar is being taken. Their
measured-cube anchor is correct during Approach and misleading during
Transport.
