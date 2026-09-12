#!/usr/bin/env python3
"""Deploy an AnimRL motion-imitation policy on the UR5e + Tesollo DG5F.

Simulation is the default. Physical arm output and physical hand output are
armed separately and explicitly, so every rung of the commissioning ladder in
``deployment/README.md`` is a flag change rather than an edit.

The policy contract is imported from ``simtoolreal_animrl.sim2sim`` rather than
restated here: the same ``build_observation`` and ``actions_to_position_targets``
that run in sim2sim run on the robot. MuJoCo is present only as a
forward-kinematics engine for the palm and fingertip poses; it never integrates
physics while the robot is moving.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simtoolreal_animrl.envs.demonstration import JointDemonstration60Hz
from simtoolreal_animrl.sim2sim.constants import (
    ACTION_DIM,
    ARM_JOINT_NAMES,
    BASE_OBSERVATION_DIM,
    HAND_JOINT_NAMES,
    JOINT_NAMES,
)
from simtoolreal_animrl.sim2sim.observation import (
    QuaternionContinuity,
    actions_to_position_targets,
    build_observation,
    smoothstep01,
)
from simtoolreal_animrl.sim2sim.policy import AnimRLInferencePolicy, load_saved_run

from deployment.animrl_deploy import (
    ArmClient,
    ArmClientError,
    CubeSourceError,
    DemonstrationCube,
    FrozenCube,
    HandClient,
    HandClientError,
    HardwareKinematics,
    PoseEstimationCube,
    SafetyAbort,
    SpikeMonitor,
    TargetLimiter,
    confirm_send,
    wait_for_key,
)

ARM_DOF = 6
HAND_DOF = 20
DEFAULT_CONTROL_HZ = 60.0
DEFAULT_ARM_CONFIG = Path(
    "/home/duplo/simone/SimToolReal/deployment/simtoolreal_real/pc_ur_new.json"
)


# ---------------------------------------------------------------- contract --
def load_run_or_explain(checkpoint: Path, config: Path):
    """Load the run, turning the dimension assertion into actionable advice."""
    try:
        return load_saved_run(checkpoint, config)
    except ValueError as error:
        if "contract" not in str(error):
            raise
        raise SystemExit(
            "\n".join(
                (
                    "",
                    "Observation-contract mismatch: {}".format(error),
                    "",
                    "The checkpoint and the checked-out sim2sim package disagree "
                    "about the observation.",
                    "  sim2sim expects : {} values".format(BASE_OBSERVATION_DIM),
                    "  this checkpoint : see 'observation_dim' in its config.json",
                    "",
                    "The repository migrated the palm/cube rotations from "
                    "quaternions (108-D) to the 6-D encoding (112-D). A policy "
                    "must be deployed against the observation it was trained on; "
                    "there is no adapter between them, and running the wrong one "
                    "feeds the network a differently-encoded world.",
                    "",
                    "Either check out the sim2sim revision matching this "
                    "checkpoint (for a 108-D policy: 'git stash' the in-progress "
                    "112-D work), or deploy a checkpoint trained on the 112-D "
                    "observation.",
                    "",
                )
            )
        )


def resolve_demo_path(repo_root: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else repo_root / path


# ------------------------------------------------------------------- state --
class SimulatedSide:
    """Stand-in state for a subsystem that is not being read from hardware.

    ``demonstration`` replays the reference trajectory, including its recorded
    velocities. This is the proxy the SimToolReal hand-only controller used for
    its simulated arm, and it is the default for the same reason: it reproduces
    the observation the policy was trained against.

    ``target`` instead assumes the subsystem tracks the policy's own command.
    That has no physics behind it, so a joint would otherwise traverse the whole
    step within one control period and report an implied velocity of tens of
    rad/s -- far outside anything training ever showed the network, which then
    produces correspondingly wild actions. The implied velocity is therefore
    clamped, and the position advances only as far as that clamp allows.
    """

    def __init__(
        self,
        source: str,
        initial_q: np.ndarray,
        control_dt: float,
        max_velocity_rad_s: float,
    ) -> None:
        self.source = source
        self.q = np.asarray(initial_q, dtype=np.float64).copy()
        self.dq = np.zeros_like(self.q)
        self.control_dt = float(control_dt)
        self.max_velocity = float(max_velocity_rad_s)

    def update(
        self,
        commanded: np.ndarray,
        reference_q: np.ndarray,
        reference_dq: np.ndarray,
    ) -> None:
        if self.source == "demonstration":
            self.q = np.asarray(reference_q, dtype=np.float64).copy()
            self.dq = np.asarray(reference_dq, dtype=np.float64).copy()
            return
        requested = np.asarray(commanded, dtype=np.float64)
        velocity = np.clip(
            (requested - self.q) / self.control_dt,
            -self.max_velocity,
            self.max_velocity,
        )
        self.q = self.q + velocity * self.control_dt
        self.dq = velocity


def format_joint_list(indices, names) -> str:
    return ", ".join(str(names[int(i)]) for i in indices)


# -------------------------------------------------------------- cube check --
def check_cube_frame(args, reference) -> int:
    """Compare a live estimator pose with the demonstration pose at an index."""
    source = PoseEstimationCube(
        args.pose_address,
        board_id=args.pose_board_id,
        minimum_confidence=args.pose_min_confidence,
        pose_timeout=args.pose_timeout,
        z_offset_m=args.pose_z_offset_m,
    )
    demo = DemonstrationCube(reference)
    try:
        source.wait_for_pose(timeout=args.pose_wait_seconds)
        live_pose, _, _ = source.cube_state(args.rsi_index)
        demo_pose, _, _ = demo.cube_state(args.rsi_index)
        print()
        print("Cube frame check at reference index {}".format(args.rsi_index))
        print("  demonstration position : {}".format(np.round(demo_pose[:3], 4).tolist()))
        print("  estimator    position : {}".format(np.round(live_pose[:3], 4).tolist()))
        print("  difference            : {}".format(
            np.round(live_pose[:3] - demo_pose[:3], 4).tolist()
        ))
        print("  demonstration quat xyzw: {}".format(np.round(demo_pose[3:], 4).tolist()))
        print("  estimator    quat xyzw: {}".format(np.round(live_pose[3:], 4).tolist()))
        print()
        print(
            "Place the real cube where the demonstration has it at this index. "
            "A residual of a few millimetres is calibration error; a sign flip "
            "or a swapped axis means the estimator frame is NOT the "
            "demonstration frame, and --cube-source pose-estimation must not be "
            "used until that is resolved."
        )
    finally:
        source.close()
    return 0


# -------------------------------------------------------------------- main --
def main() -> int:
    args = parse_args()

    run = load_run_or_explain(args.checkpoint, args.config)
    env_cfg = run.env_cfg
    control_cfg = env_cfg["control"]
    demo_path = resolve_demo_path(run.repo_root, env_cfg["motion"]["file"])
    reference = JointDemonstration60Hz.load(
        demo_path, device="cpu", expected_hz=float(env_cfg["motion"]["frequency_hz"])
    )
    if not 0 <= args.rsi_index < reference.last_index:
        raise SystemExit(
            "--rsi-index must lie in [0, {}]".format(reference.last_index - 1)
        )

    if args.check_cube_frame:
        try:
            return check_cube_frame(args, reference)
        except CubeSourceError as error:
            print("\nERROR: {}".format(error))
            return 2

    import torch

    defaults = np.asarray(
        env_cfg["init_state"]["default_arm_joint_angles"]
        + env_cfg["init_state"]["default_hand_joint_angles"],
        dtype=np.float64,
    )
    arm_scale = float(control_cfg["scale_joint_target"]) * args.arm_action_scale
    hand_scale = float(control_cfg["scale_hand_joint_target"]) * args.hand_action_scale
    residual_clip = float(control_cfg["clip_joint_target"])
    control_dt = 1.0 / args.control_hz
    termination = env_cfg.get("termination", {})
    if args.max_arm_reference_error_rad is None:
        args.max_arm_reference_error_rad = float(
            termination.get("arm_position_threshold_rad", 0.35)
        )
    if args.max_hand_reference_error_rad is None:
        args.max_hand_reference_error_rad = float(
            termination.get("hand_position_threshold_rad", 1.35)
        )

    send_arm = bool(args.send_to_arm)
    send_hand = bool(args.send_to_hand)
    # Commanding a subsystem without reading it would close the loop on a
    # fiction, so hardware output implies hardware state.
    use_arm_state = bool(args.use_real_arm_state) or send_arm
    use_hand_state = bool(args.use_real_hand_state) or send_hand

    actor = AnimRLInferencePolicy(run, device=args.device)
    kinematics = HardwareKinematics(
        run,
        enable_viewer=not args.no_viewer,
        enable_reference_ghost=not args.no_viewer and not args.no_ghost,
    )

    start_sample = reference.sample(torch.tensor([args.rsi_index], dtype=torch.long))
    start_q = start_sample.q[0].numpy().astype(np.float64)

    if args.cube_source == "demonstration":
        cube = DemonstrationCube(reference)
    elif args.cube_source == "frozen":
        cube = FrozenCube(reference, args.rsi_index)
    else:
        cube = PoseEstimationCube(
            args.pose_address,
            board_id=args.pose_board_id,
            minimum_confidence=args.pose_min_confidence,
            pose_timeout=args.pose_timeout,
            z_offset_m=args.pose_z_offset_m,
        )

    arm_client = None
    hand_client = None
    exit_code = 0
    try:
        if use_arm_state:
            arm_client = ArmClient(
                args.arm_config,
                stream_hz=args.arm_stream_hz,
                state_timeout=args.state_timeout,
                connect_settle_seconds=args.arm_connect_settle_seconds,
            )
            arm_client.wait_for_state(timeout=args.state_wait_seconds)
            print(
                "UR5 state: q_deg={}".format(
                    np.rad2deg(arm_client.positions).round(2).tolist()
                )
            )
        if use_hand_state:
            hand_client = HandClient(
                bind_address=args.hand_bind_address,
                state_port=args.hand_state_port,
                command_address=args.hand_command_address,
                command_port=args.hand_command_port,
                state_timeout=args.state_timeout,
            )
            hand_client.wait_for_state(timeout=args.state_wait_seconds)
            print(
                "DG5F state: max|q|={:.3f} rad".format(
                    float(np.max(np.abs(hand_client.positions)))
                )
            )
        if isinstance(cube, PoseEstimationCube):
            cube.wait_for_pose(timeout=args.pose_wait_seconds)
            print("Cube pose stream is live on {}".format(args.pose_address))

        print_banner(args, run, reference, arm_scale, hand_scale, send_arm, send_hand)

        if send_arm or send_hand:
            outputs = []
            if send_arm:
                outputs.append("UR5e arm  -> {}".format(args.arm_config))
            if send_hand:
                outputs.append(
                    "DG5F hand -> udp://{}:{}".format(
                        args.hand_command_address, args.hand_command_port
                    )
                )
            confirm_send(outputs)

        if send_arm:
            home_arm(arm_client, start_q[:ARM_DOF], args)
        if send_hand:
            home_hand(hand_client, start_q[ARM_DOF:], args)

        exit_code = run_policy(
            args=args,
            actor=actor,
            kinematics=kinematics,
            reference=reference,
            cube=cube,
            arm_client=arm_client,
            hand_client=hand_client,
            defaults=defaults,
            start_q=start_q,
            arm_scale=arm_scale,
            hand_scale=hand_scale,
            residual_clip=residual_clip,
            control_dt=control_dt,
            send_arm=send_arm,
            send_hand=send_hand,
            use_arm_state=use_arm_state,
            use_hand_state=use_hand_state,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        exit_code = 130
    except (SafetyAbort, ArmClientError, HandClientError, CubeSourceError) as error:
        print("\nSAFETY STOP: {}".format(error))
        exit_code = 2
    finally:
        if arm_client is not None:
            if send_arm:
                print("Braking the arm (zero-velocity hold)...")
                arm_client.brake(duration_s=args.brake_seconds)
            arm_client.close()
        if hand_client is not None:
            if send_hand:
                print("Holding the hand at its measured position...")
                hand_client.hold_measured(duration_s=args.brake_seconds)
            hand_client.close()
        if isinstance(cube, PoseEstimationCube):
            cube.close()
        kinematics.close()
    return exit_code


def print_banner(args, run, reference, arm_scale, hand_scale, send_arm, send_hand):
    print()
    print("=" * 70)
    print("AnimRL real-robot deployment")
    print("  checkpoint      : {}".format(run.checkpoint_path))
    print("  observation/act : {}/{}".format(BASE_OBSERVATION_DIM, ACTION_DIM))
    print("  demonstration   : {} ({} samples)".format(
        Path(run.env_cfg["motion"]["file"]).name, reference.sample_count
    ))
    print("  start index     : {}".format(args.rsi_index))
    print("  control rate    : {:g} Hz".format(args.control_hz))
    print("  residual scale  : arm {:.4f}, hand {:.4f}".format(arm_scale, hand_scale))
    print("  step limit      : arm {:g} rad, hand {:g} rad".format(
        args.max_arm_step_rad, args.max_hand_step_rad
    ))
    print("  target smoothing: {:g}".format(args.target_smoothing))
    print("  startup policy  : {:g} demo-s reference-to-policy blend".format(
        args.startup_policy_blend_seconds
    ))
    print("  startup target  : {:g} s home-to-target ramp".format(
        args.startup_ramp_seconds
    ))
    print("  quaternion mode : {}".format(
        "event-triggered smooth transition"
        if args.smooth_quaternion_transition
        else ("continuous" if args.continuous_quaternions else "canonical")
    ))
    print("  cube source     : {}".format(args.cube_source))
    if args.commission_arm_only_ideal_context:
        print("  hybrid context  : real arm + ideal hand/cube + MuJoCo FK")
    elif args.commission_hand_only_ideal_context:
        print("  hybrid context  : real hand + ideal arm/cube + MuJoCo FK")
    else:
        print("  simulated state : {}".format(args.simulated_state_source))
    print("  arm output      : {}".format("ARMED" if send_arm else "simulated"))
    print("  hand output     : {}".format("ARMED" if send_hand else "simulated"))
    print("  action spike    : {} above {:g}".format(
        args.spike_mode, args.max_action_step
    ))
    print("  ref deviation   : {} above arm {:g} / hand {:g} rad".format(
        args.reference_error_mode,
        args.max_arm_reference_error_rad,
        args.max_hand_reference_error_rad,
    ))
    print("=" * 70)


def home_arm(arm_client, target_q, args) -> None:
    measured = arm_client.require_fresh_state()
    distance = float(np.max(np.abs(target_q - measured)))
    print()
    print("Arm homing to the start pose:")
    print("  measured q_deg: {}".format(np.rad2deg(measured).round(2).tolist()))
    print("  target   q_deg: {}".format(np.rad2deg(target_q).round(2).tolist()))
    print("  largest joint move: {:.3f} rad ({:.1f} deg)".format(
        distance, np.rad2deg(distance)
    ))
    if distance > args.max_home_distance_rad:
        raise SafetyAbort(
            "Homing move of {:.3f} rad exceeds --max-home-distance-rad {:.3f}. "
            "Jog the arm closer to the start pose by hand first.".format(
                distance, args.max_home_distance_rad
            )
        )
    # The arm state socket is polled by this process. Keep draining it while
    # the operator inspects the pose; otherwise a deliberate pause at this
    # prompt makes a healthy state stream appear stale immediately afterward.
    wait_for_key(
        "Press Space to send the homing trajectory, or q to abort: ",
        on_wait=arm_client.poll,
    )
    seconds = max(args.home_seconds, distance / max(args.home_speed_rad_s, 1e-6))
    midpoint = 0.5 * (measured + target_q)
    arm_client.send_trajectory(
        np.asarray([0.0, 0.5 * seconds, seconds]),
        np.stack([measured, midpoint, target_q]),
    )
    deadline = time.monotonic() + seconds + args.home_settle_seconds
    while time.monotonic() < deadline:
        arm_client.poll()
        time.sleep(0.02)
    measured = arm_client.require_fresh_state()
    error = float(np.max(np.abs(target_q - measured)))
    print("  homing finished, max|q-target| = {:.4f} rad".format(error))
    if error > args.home_tolerance_rad:
        raise SafetyAbort(
            "Arm did not reach the start pose (error {:.4f} rad > tolerance "
            "{:.4f} rad).".format(error, args.home_tolerance_rad)
        )
    arm_client.start_streaming()
    arm_client.set_target(measured)


def home_hand(hand_client, target_q, args) -> None:
    measured = hand_client.require_fresh_state()
    distance = float(np.max(np.abs(target_q - measured)))
    print()
    print("Hand homing to the start pose:")
    print("  largest joint move: {:.3f} rad".format(distance))
    wait_for_key("Press Space to ramp the hand to the start pose, or q to abort: ")
    steps = max(1, int(np.ceil(distance / args.hand_home_step_rad)))
    for index in range(1, steps + 1):
        blend = index / steps
        hand_client.send_target((1.0 - blend) * measured + blend * target_q)
        time.sleep(args.hand_home_step_seconds)
    for _ in range(int(args.home_settle_seconds / max(args.hand_home_step_seconds, 1e-3))):
        hand_client.send_target(target_q)
        time.sleep(args.hand_home_step_seconds)
    settle_deadline = time.monotonic() + args.hand_home_timeout_seconds
    while True:
        measured = hand_client.require_fresh_state()
        absolute_error = np.abs(target_q - measured)
        if float(np.max(absolute_error)) <= args.hand_home_tolerance_rad:
            break
        if time.monotonic() >= settle_deadline:
            break
        hand_client.send_target(target_q)
        time.sleep(args.hand_home_step_seconds)
    worst_index = int(np.argmax(absolute_error))
    error = float(absolute_error[worst_index])
    print(
        "  homing finished, max|q-target| = {:.4f} rad on {} "
        "(measured={:.4f}, target={:.4f})".format(
            error,
            HAND_JOINT_NAMES[worst_index],
            measured[worst_index],
            target_q[worst_index],
        )
    )
    if error > args.hand_home_tolerance_rad:
        raise SafetyAbort(
            "Hand did not reach the start pose within {:.1f} s: {} remains "
            "{:.4f} rad from target (limit {:.4f} rad).".format(
                args.hand_home_timeout_seconds,
                HAND_JOINT_NAMES[worst_index],
                error,
                args.hand_home_tolerance_rad,
            )
        )


def run_policy(
    *,
    args,
    actor,
    kinematics,
    reference,
    cube,
    arm_client,
    hand_client,
    defaults,
    start_q,
    arm_scale,
    hand_scale,
    residual_clip,
    control_dt,
    send_arm,
    send_hand,
    use_arm_state,
    use_hand_state,
) -> int:
    import torch

    limiter = TargetLimiter(
        kinematics.joint_lower_limits,
        kinematics.joint_upper_limits,
        max_arm_step_rad=args.max_arm_step_rad,
        max_hand_step_rad=args.max_hand_step_rad,
        smoothing=args.target_smoothing,
    )
    limiter.reset(start_q)
    if send_arm and not send_hand:
        spike_slice = slice(0, ARM_DOF)
        spike_labels = list(ARM_JOINT_NAMES)
    elif send_hand and not send_arm:
        spike_slice = slice(ARM_DOF, ARM_DOF + HAND_DOF)
        spike_labels = list(HAND_JOINT_NAMES)
    else:
        spike_slice = slice(None)
        spike_labels = list(JOINT_NAMES)
    action_spikes = SpikeMonitor(
        args.max_action_step,
        mode=args.spike_mode,
        name="action",
        labels=spike_labels,
        grace_steps=args.spike_grace_steps,
    )
    simulated_arm = SimulatedSide(
        args.simulated_state_source,
        start_q[:ARM_DOF],
        control_dt,
        args.simulated_max_velocity_rad_s,
    )
    simulated_hand = SimulatedSide(
        args.simulated_state_source,
        start_q[ARM_DOF:],
        control_dt,
        args.simulated_max_velocity_rad_s,
    )

    previous_targets = start_q.copy()
    quaternion_continuity = (
        QuaternionContinuity()
        if args.continuous_quaternions or args.smooth_quaternion_transition
        else None
    )
    motion_frequency_hz = float(reference.frequency_hz)
    reference_index = int(args.rsi_index)
    quaternion_transition_started_at = None
    steps = 0
    started_at = time.perf_counter()
    overruns = 0

    if use_arm_state:
        arm_client.poll()
    if use_hand_state:
        hand_client.poll()

    print()
    print("Running. Ctrl+C stops and brakes.")
    while reference_index < reference.last_index and kinematics.viewer_is_running():
        if args.max_steps and steps >= args.max_steps:
            break
        loop_started = time.perf_counter()

        # -- measured state ------------------------------------------------
        if use_arm_state:
            arm_q = arm_client.require_fresh_state()
            arm_dq = arm_client.velocities.copy()
        else:
            arm_q, arm_dq = simulated_arm.q.copy(), simulated_arm.dq.copy()
        if use_hand_state:
            hand_q = hand_client.require_fresh_state()
            hand_dq = hand_client.velocities.copy()
        else:
            hand_q, hand_dq = simulated_hand.q.copy(), simulated_hand.dq.copy()

        measured_q = np.concatenate((arm_q, hand_q))
        measured_dq = np.concatenate((arm_dq, hand_dq))

        # -- observation ---------------------------------------------------
        cube_pose, cube_linear, cube_angular = cube.cube_state(reference_index)
        state = kinematics.update(
            measured_q, measured_dq, cube_pose, cube_linear, cube_angular
        )
        phase = reference_index / float(reference.last_index)
        canonical_observation = build_observation(
            state,
            previous_targets,
            phase,
            kinematics.joint_lower_limits,
            kinematics.joint_upper_limits,
        )
        continuity_active = (
            quaternion_continuity is not None
            and args.continuous_quaternions
            and (
                args.continuous_quaternions_until_seconds is None
                or reference_index / motion_frequency_hz
                < args.continuous_quaternions_until_seconds
            )
        )
        observation = canonical_observation
        if continuity_active:
            observation = quaternion_continuity.apply(observation)

        # -- policy --------------------------------------------------------
        if args.smooth_quaternion_transition:
            continuous_observation = quaternion_continuity.apply(
                canonical_observation
            )
            demo_seconds = reference_index / motion_frequency_hz
            if (
                quaternion_transition_started_at is None
                and any(quaternion_continuity.last_flipped)
            ):
                quaternion_transition_started_at = demo_seconds
                print(
                    "Quaternion action transition triggered at {:.3f} s".format(
                        demo_seconds
                    )
                )
            progress = (
                0.0
                if quaternion_transition_started_at is None
                else (demo_seconds - quaternion_transition_started_at)
                / args.quaternion_transition_duration_seconds
            )
            canonical_weight = smoothstep01(progress)
            if canonical_weight <= 0.0:
                actions = actor(continuous_observation)
            elif canonical_weight >= 1.0:
                actions = actor(canonical_observation)
            else:
                actions = (
                    (1.0 - canonical_weight) * actor(continuous_observation)
                    + canonical_weight * actor(canonical_observation)
                )
        else:
            actions = actor(observation)
        if args.startup_policy_blend_seconds > 0.0:
            startup_policy_progress = (
                steps
                / (
                    args.startup_policy_blend_seconds
                    * motion_frequency_hz
                )
            )
            if startup_policy_progress < 1.0:
                current_reference = reference.sample(
                    torch.tensor([reference_index], dtype=torch.long)
                )
                # Action-space warm starting must remain in the policy's
                # training units.  Hardware action scaling is applied later
                # by actions_to_position_targets; using the reduced physical
                # scale here would inflate reference actions by 1/scale.
                native_arm_scale = arm_scale / args.arm_action_scale
                native_hand_scale = hand_scale / args.hand_action_scale
                action_scales = np.concatenate(
                    (
                        np.full(ARM_DOF, native_arm_scale, dtype=np.float64),
                        np.full(HAND_DOF, native_hand_scale, dtype=np.float64),
                    )
                )
                reference_actions = (
                    current_reference.q[0].numpy() - defaults
                ) / action_scales
                policy_weight = smoothstep01(startup_policy_progress)
                actions = (
                    (1.0 - policy_weight) * reference_actions
                    + policy_weight * actions
                )
        # A commissioning mode deliberately leaves one subsystem disconnected.
        # Its policy outputs remain useful diagnostics, but must not stop the
        # physically armed subsystem. When both are armed, monitor all 26.
        action_spikes.update(actions[spike_slice])
        raw_targets = actions_to_position_targets(
            actions, defaults, arm_scale, hand_scale, residual_clip
        )
        if args.startup_ramp_seconds > 0.0:
            startup_progress = (
                (steps + 1) * control_dt / args.startup_ramp_seconds
            )
            if startup_progress < 1.0:
                startup_weight = smoothstep01(startup_progress)
                raw_targets = (
                    (1.0 - startup_weight) * start_q
                    + startup_weight * raw_targets
                )
        applied_targets, limit_info = limiter.apply(raw_targets)

        # -- output --------------------------------------------------------
        if send_arm:
            arm_client.set_target(applied_targets[:ARM_DOF])
        if send_hand:
            hand_client.send_target(applied_targets[ARM_DOF:])

        next_sample = reference.sample(
            torch.tensor([reference_index + 1], dtype=torch.long)
        )
        reference_q = next_sample.q[0].numpy().astype(np.float64)
        reference_dq = next_sample.dq[0].numpy().astype(np.float64)
        kinematics.set_reference_ghost(reference_q)
        kinematics.sync_viewer()

        if not use_arm_state:
            simulated_arm.update(
                applied_targets[:ARM_DOF], reference_q[:ARM_DOF], reference_dq[:ARM_DOF]
            )
        if not use_hand_state:
            simulated_hand.update(
                applied_targets[ARM_DOF:], reference_q[ARM_DOF:], reference_dq[ARM_DOF:]
            )

        previous_targets = (
            raw_targets if args.previous_target_source == "raw" else applied_targets
        ).copy()
        reference_index += 1
        steps += 1

        # -- monitors ------------------------------------------------------
        # Deviation is measured against the demonstration, not against the
        # commanded target. A soft-PD finger legitimately sits far from its
        # target -- up to 1.4 rad in the reference rollout -- so target error
        # says little. Distance from the reference is what training actually
        # terminated on, so crossing it means the robot is in a state the
        # policy was never trained to continue from.
        arm_deviation = float(np.max(np.abs(measured_q[:ARM_DOF] - reference_q[:ARM_DOF])))
        hand_deviation = float(np.max(np.abs(measured_q[ARM_DOF:] - reference_q[ARM_DOF:])))
        if args.reference_error_mode != "off":
            for label, deviation, threshold in (
                ("Arm", arm_deviation, args.max_arm_reference_error_rad),
                ("Hand", hand_deviation, args.max_hand_reference_error_rad),
            ):
                if deviation <= threshold:
                    continue
                message = (
                    "{} deviates {:.3f} rad from the demonstration at step {} "
                    "(training terminated above {:.3f} rad).".format(
                        label, deviation, steps, threshold
                    )
                )
                if args.reference_error_mode == "stop":
                    raise SafetyAbort(message)
                print("WARNING: " + message)
        if use_hand_state and args.current_warning_ma > 0.0:
            hot = hand_client.over_current_joints(args.current_warning_ma)
            if hot.size:
                print(
                    "WARNING: DG5F motor current above {:g} mA on {}".format(
                        args.current_warning_ma,
                        format_joint_list(hot, HAND_JOINT_NAMES),
                    )
                )

        if args.print_every and (steps == 1 or steps % args.print_every == 0):
            arm_error, hand_error = arm_deviation, hand_deviation
            tracking_error = float(np.max(np.abs(measured_q - previous_targets)))
            note = ""
            if limit_info["step_limited_joints"].size:
                note = " step-limited:{}".format(
                    format_joint_list(limit_info["step_limited_joints"], JOINT_NAMES)
                )
            print(
                "step={:4d} ref={:4d} phase={:.3f} max|arm-ref|={:.3f} "
                "max|hand-ref|={:.3f} max_step={:.4f} track={:.3f}{}".format(
                    steps,
                    reference_index,
                    phase,
                    arm_error,
                    hand_error,
                    limit_info["max_requested_step_rad"],
                    tracking_error,
                    note,
                )
            )

        if args.debug_step:
            hand_keepalive = None
            if send_hand:
                hand_target = applied_targets[ARM_DOF:].copy()

                def hand_keepalive() -> None:
                    hand_client.require_fresh_state()
                    hand_client.send_target(hand_target)

            wait_for_key(
                "  [step {}] Space for the next step, q to stop: ".format(steps),
                on_wait=hand_keepalive,
            )
        elif not args.no_realtime:
            remaining = control_dt - (time.perf_counter() - loop_started)
            if remaining > 0.0:
                time.sleep(remaining)
            elif remaining < -0.5 * control_dt:
                overruns += 1

    elapsed = time.perf_counter() - started_at
    print()
    print(
        "Finished {} steps in {:.1f} s at reference index {}.".format(
            steps, elapsed, reference_index
        )
    )
    if action_spikes.worst_index >= 0:
        print(
            "Largest single-step action change: {:.4f} on {} (threshold {:g}, "
            "{} detection(s)).".format(
                action_spikes.worst,
                spike_labels[action_spikes.worst_index],
                action_spikes.threshold,
                action_spikes.detections,
            )
        )
    if overruns and not args.debug_step:
        print(
            "WARNING: {} control steps overran the {:g} Hz budget.".format(
                overruns, args.control_hz
            )
        )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    policy = parser.add_argument_group("policy")
    policy.add_argument("--checkpoint", type=Path, required=True)
    policy.add_argument(
        "--config", type=Path, default=None,
        help="Defaults to config.json beside the checkpoint.",
    )
    policy.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    policy.add_argument("--rsi-index", type=int, default=0)
    policy.add_argument("--max-steps", type=int, default=0)
    policy.add_argument("--control-hz", type=float, default=DEFAULT_CONTROL_HZ)

    output = parser.add_argument_group("physical output (off unless given)")
    output.add_argument("--send-to-arm", action="store_true")
    output.add_argument("--send-to-hand", action="store_true")
    output.add_argument("--use-real-arm-state", action="store_true")
    output.add_argument("--use-real-hand-state", action="store_true")
    commissioning = output.add_mutually_exclusive_group()
    commissioning.add_argument(
        "--commission-arm-only-ideal-context",
        action="store_true",
        help=(
            "Command/read only the arm; source hand q/dq and cube pose from "
            "the demonstration and rebuild kinematics in the loop."
        ),
    )
    commissioning.add_argument(
        "--commission-hand-only-ideal-context",
        action="store_true",
        help=(
            "Command/read only the hand; source arm q/dq and cube pose from "
            "the demonstration and rebuild kinematics in the loop."
        ),
    )

    scaling = parser.add_argument_group("motion scaling and limits")
    scaling.add_argument("--arm-action-scale", type=float, default=1.0)
    scaling.add_argument("--hand-action-scale", type=float, default=1.0)
    scaling.add_argument("--max-arm-step-rad", type=float, default=0.02)
    scaling.add_argument("--max-hand-step-rad", type=float, default=0.05)
    scaling.add_argument(
        "--target-smoothing", type=float, default=0.0,
        help="EMA weight on the previous target, in [0, 1). 0 disables it.",
    )
    scaling.add_argument(
        "--startup-ramp-seconds", type=float, default=1.0,
        help=(
            "Smoothly ramp targets from the verified home pose to the policy "
            "target (default: 1.0 s; 0 disables)."
        ),
    )
    scaling.add_argument(
        "--startup-policy-blend-seconds", type=float, default=1.0,
        help=(
            "Start exactly on the demonstration action and smoothly transfer "
            "to the learned action over this many seconds of 60 Hz "
            "demonstration frames (default: 1.0; 0 disables)."
        ),
    )

    monitors = parser.add_argument_group("safety monitors")
    monitors.add_argument(
        "--max-action-step", type=float, default=1.0,
        help=(
            "Largest tolerated single-step change in a raw action. The "
            "quaternion double-cover crossing produced 4.03 in the run that "
            "motivated the 6-D encoding."
        ),
    )
    monitors.add_argument(
        "--spike-mode", choices=("off", "warn", "stop"), default="stop",
    )
    monitors.add_argument(
        "--spike-grace-steps", type=int, default=2,
        help=(
            "Report but do not abort on spikes in the first N steps, where the "
            "policy's settling transient legitimately exceeds the threshold."
        ),
    )
    monitors.add_argument(
        "--max-arm-reference-error-rad", type=float, default=None,
        help="Defaults to termination.arm_position_threshold_rad from the run config.",
    )
    monitors.add_argument(
        "--max-hand-reference-error-rad", type=float, default=None,
        help="Defaults to termination.hand_position_threshold_rad from the run config.",
    )
    monitors.add_argument(
        "--reference-error-mode", choices=("off", "warn", "stop"), default="stop",
    )
    monitors.add_argument("--current-warning-ma", type=float, default=170.0)
    monitors.add_argument("--state-timeout", type=float, default=0.25)
    monitors.add_argument("--state-wait-seconds", type=float, default=10.0)

    cube_group = parser.add_argument_group("cube observation")
    cube_group.add_argument(
        "--cube-source",
        choices=("demonstration", "frozen", "pose-estimation"),
        default="demonstration",
    )
    cube_group.add_argument("--pose-address", default="tcp://127.0.0.1:5558")
    cube_group.add_argument("--pose-board-id", default="0")
    cube_group.add_argument(
        "--pose-z-offset-m",
        type=float,
        default=0.03,
        help="Add this calibration offset to the live estimator Z coordinate.",
    )
    cube_group.add_argument("--pose-min-confidence", type=float, default=0.0)
    cube_group.add_argument("--pose-timeout", type=float, default=0.5)
    cube_group.add_argument("--pose-wait-seconds", type=float, default=10.0)
    cube_group.add_argument(
        "--check-cube-frame", action="store_true",
        help="Compare the live estimator pose with the demonstration, then exit.",
    )

    hardware = parser.add_argument_group("hardware endpoints")
    hardware.add_argument("--arm-config", type=Path, default=DEFAULT_ARM_CONFIG)
    hardware.add_argument("--arm-stream-hz", type=float, default=100.0)
    hardware.add_argument(
        "--arm-connect-settle-seconds", type=float, default=1.0,
        help="Pause after binding the command socket so ZMQ PUB does not drop "
             "the first message before the controller has subscribed.",
    )
    hardware.add_argument("--hand-bind-address", default="127.0.0.1")
    hardware.add_argument("--hand-state-port", type=int, default=5563)
    hardware.add_argument("--hand-command-address", default="127.0.0.1")
    hardware.add_argument("--hand-command-port", type=int, default=5562)

    homing = parser.add_argument_group("homing")
    homing.add_argument("--home-seconds", type=float, default=5.0)
    homing.add_argument("--home-speed-rad-s", type=float, default=0.15)
    homing.add_argument("--home-settle-seconds", type=float, default=1.5)
    homing.add_argument("--home-tolerance-rad", type=float, default=0.05)
    homing.add_argument("--max-home-distance-rad", type=float, default=1.9)
    homing.add_argument("--hand-home-step-rad", type=float, default=0.03)
    homing.add_argument("--hand-home-step-seconds", type=float, default=0.02)
    homing.add_argument("--hand-home-tolerance-rad", type=float, default=0.18)
    homing.add_argument("--hand-home-timeout-seconds", type=float, default=10.0)
    homing.add_argument("--brake-seconds", type=float, default=0.5)

    behaviour = parser.add_argument_group("behaviour")
    behaviour.add_argument("--debug-step", action="store_true")
    behaviour.add_argument("--no-viewer", action="store_true")
    behaviour.add_argument("--no-ghost", action="store_true")
    behaviour.add_argument("--no-realtime", action="store_true")
    behaviour.add_argument(
        "--continuous-quaternions",
        action="store_true",
        help=(
            "For legacy 108-D policies, unwrap the palm and palm-relative "
            "cube quaternion signs against the preceding observation. This "
            "removes the artificial w=0 sign jump without changing rotation."
        ),
    )
    behaviour.add_argument(
        "--continuous-quaternions-until-seconds",
        type=float,
        default=None,
        help=(
            "Use --continuous-quaternions only before this absolute "
            "demonstration time. A branch change at the cutoff may itself "
            "be discontinuous; validate it before hardware use."
        ),
    )
    behaviour.add_argument(
        "--smooth-quaternion-transition",
        dest="smooth_quaternion_transition",
        action="store_true",
        default=True,
        help=(
            "Blend policy outputs from continuous to canonical quaternion "
            "observations after the detected sign crossing (default: enabled)."
        ),
    )
    behaviour.add_argument(
        "--no-smooth-quaternion-transition",
        dest="smooth_quaternion_transition",
        action="store_false",
        help="Disable the event-triggered quaternion policy-output transition.",
    )
    behaviour.add_argument(
        "--quaternion-transition-duration-seconds", type=float, default=1.0,
    )
    behaviour.add_argument("--print-every", type=int, default=30)
    behaviour.add_argument(
        "--simulated-state-source",
        choices=("demonstration", "target"),
        default="demonstration",
        help=(
            "State fed back for a subsystem that is not read from hardware. "
            "'demonstration' replays the reference trajectory and its recorded "
            "velocities, reproducing the training observation; 'target' assumes "
            "the subsystem tracks the policy's own command, rate-limited by "
            "--simulated-max-velocity-rad-s."
        ),
    )
    behaviour.add_argument(
        "--simulated-max-velocity-rad-s", type=float, default=3.0,
        help="Velocity clamp for --simulated-state-source target.",
    )
    behaviour.add_argument(
        "--previous-target-source",
        choices=("raw", "applied"),
        default="raw",
        help=(
            "Which target enters the observation. 'raw' reproduces sim2sim "
            "exactly; 'applied' reports what the safety limiter actually sent."
        ),
    )

    args = parser.parse_args()
    if args.commission_arm_only_ideal_context:
        if args.send_to_hand or args.use_real_hand_state:
            parser.error(
                "arm-only ideal-context mode cannot send to or read the real hand"
            )
        args.send_to_arm = True
        args.use_real_arm_state = True
        args.simulated_state_source = "demonstration"
        args.cube_source = "demonstration"
    elif args.commission_hand_only_ideal_context:
        if args.send_to_arm or args.use_real_arm_state:
            parser.error(
                "hand-only ideal-context mode cannot send to or read the real arm"
            )
        args.send_to_hand = True
        args.use_real_hand_state = True
        args.simulated_state_source = "demonstration"
        args.cube_source = "demonstration"
    if args.control_hz <= 0.0:
        parser.error("--control-hz must be positive")
    if args.max_steps < 0:
        parser.error("--max-steps cannot be negative")
    if (
        args.continuous_quaternions_until_seconds is not None
        and args.continuous_quaternions_until_seconds < 0.0
    ):
        parser.error("--continuous-quaternions-until-seconds cannot be negative")
    if (
        args.continuous_quaternions_until_seconds is not None
        and not args.continuous_quaternions
    ):
        parser.error(
            "--continuous-quaternions-until-seconds requires "
            "--continuous-quaternions"
        )
    if args.smooth_quaternion_transition and args.continuous_quaternions:
        parser.error(
            "--smooth-quaternion-transition and --continuous-quaternions "
            "are mutually exclusive"
        )
    if args.quaternion_transition_duration_seconds <= 0.0:
        parser.error("--quaternion-transition-duration-seconds must be positive")
    if not 0.0 <= args.target_smoothing < 1.0:
        parser.error("--target-smoothing must lie in [0, 1)")
    if args.startup_ramp_seconds < 0.0:
        parser.error("--startup-ramp-seconds cannot be negative")
    if args.startup_policy_blend_seconds < 0.0:
        parser.error("--startup-policy-blend-seconds cannot be negative")
    if args.hand_home_timeout_seconds <= 0.0:
        parser.error("--hand-home-timeout-seconds must be positive")
    for name in ("arm_action_scale", "hand_action_scale"):
        if getattr(args, name) <= 0.0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    return args


if __name__ == "__main__":
    raise SystemExit(main())
