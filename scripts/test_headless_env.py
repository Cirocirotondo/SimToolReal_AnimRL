#!/usr/bin/env python3
"""End-to-end headless environment milestone test (no PPO)."""

import argparse
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Import the environment only after its package path is installed. It preserves
# Isaac Gym's required import-before-torch ordering internally.
from simtoolreal_animrl.cfg import SimToolRealCfg
from simtoolreal_animrl.envs.controller import (
    ARM_JOINT_NAMES,
    ARM_PD_DAMPING,
    ARM_PD_STIFFNESS,
    HAND_PD_DAMPING,
    HAND_PD_STIFFNESS,
)
from simtoolreal_animrl.envs.motion_imitation import MotionImitationEnv


# Where the demonstrated bar is unambiguously airborne, and so where the palm
# keypoint anchor starts mattering (docs/adr/0001, CONTEXT.md). Measured on
# demo_..._stable_grasp.npz: the bar first rises 2 mm at frame 770 and 20 mm at
# 832, out of a 0.241 m total lift. 832 is the conservative end of that range
# on purpose -- the frames in between are ambiguous, and a test that straddles
# them would report a failure of the anchor when it saw only lift-off jitter.
LIFT_START_REFERENCE_INDEX = 832

# Tolerance for comparing a cuboid world pose against the reference it was
# written from. Looser than the joint-space bounds elsewhere, and it has to be:
# the cuboid's world position carries the robot base offset, so these are
# float32 values of order one metre round-tripped through the GPU root-state
# tensor. 1e-6 is below what that representation can hold and the measured
# residual sits at 1.9e-6 whatever the environment does.
CUBE_POSE_ATOL = 1e-5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Override AnimRL's production value of 4096 for a quick smoke test.",
    )
    parser.add_argument(
        "--rsi-index",
        type=int,
        default=None,
        help=(
            "Use one explicit valid RSI index for every env; by default use "
            "the configured RSI mixture."
        ),
    )
    parser.add_argument(
        "--sim-device", default="cuda:0", help="Isaac Gym simulation device."
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU PhysX and the CPU tensor pipeline.",
    )
    parser.add_argument(
        "--training-camera",
        action="store_true",
        help="Create the off-screen training camera and verify one RGB frame.",
    )
    return parser.parse_args()


def assert_correct_pd_gains(env):
    gains = env.pd_gain_summary()
    np.testing.assert_allclose(
        gains["arm_stiffness"], np.asarray(ARM_PD_STIFFNESS), rtol=0.0, atol=1e-5
    )
    np.testing.assert_allclose(
        gains["arm_damping"], np.asarray(ARM_PD_DAMPING), rtol=0.0, atol=1e-5
    )
    np.testing.assert_allclose(
        gains["hand_stiffness"], np.asarray(HAND_PD_STIFFNESS), rtol=0.0, atol=1e-5
    )
    np.testing.assert_allclose(
        gains["hand_damping"], np.asarray(HAND_PD_DAMPING), rtol=0.0, atol=1e-5
    )


def identity_transform_index(env):
    """The bank entry closest to leaving the demonstration where it was.

    The bank is sampled randomly, so it holds no exact identity; this picks the
    nearest one the same way the evaluator does, trading 10 degrees of yaw
    against 1.75 cm of translation.
    """
    import torch

    bank = env.transform_bank
    return int(
        (
            bank.translation[:, :2].square().sum(dim=1)
            + (0.1 * bank.yaw_rad).square()
        ).argmin()
    )


def assert_reset_matches_reference(env, reference_index):
    """The RSI write must land the robot exactly on the pose it was given.

    Pinned to the near-identity transform first. env.reference is the raw
    demonstration, while a reset draws from the transform bank, so comparing the
    two under a randomly sampled transform measures the transform rather than
    the state write -- and passes or fails depending on which transforms the
    bank happened to sample.
    """
    import torch

    env.reset_idx(
        env.all_env_ids,
        env.reference_index.clone(),
        torch.full_like(env.all_env_ids, identity_transform_index(env)),
    )
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    reference = env.transform_bank.sample(
        env.transform_index, env.reference_index
    )
    position_error = (env.q - reference.q).abs().max().item()
    velocity_error = (env.dq - reference.dq).abs().max().item()
    if position_error > 1e-6 or velocity_error > 1e-5:
        raise AssertionError(
            "RSI state write failed: max q error={:.3e}, max dq error={:.3e}".format(
                position_error, velocity_error
            )
        )
    if reference_index is not None:
        if not bool((env.reference_index == int(reference_index)).all()):
            raise AssertionError("The requested RSI index was not applied")
    expected_cube_root = env._cube_reference_root_states(reference)
    cube_error = (env.cube_root_state - expected_cube_root).abs().max().item()
    if cube_error > CUBE_POSE_ATOL:
        raise AssertionError(
            "RSI cube root-state write failed: max error={:.3e}".format(cube_error)
        )
    return position_error, velocity_error, cube_error


def assert_object_scene_contract(env):
    robot_shapes = env.gym.get_actor_rigid_shape_properties(
        env.envs[0], env.robot_handles[0]
    )
    cube_shapes = env.gym.get_actor_rigid_shape_properties(
        env.envs[0], env.cube_handles[0]
    )
    table_shapes = env.gym.get_actor_rigid_shape_properties(
        env.envs[0], env.table_handles[0]
    )
    table_filter_bit = int(env.robot_table_collision_filter_bit)
    arm_cube_filter_bit = int(env.arm_cube_collision_filter_bit)
    if table_filter_bit <= 0 or arm_cube_filter_bit <= 0:
        raise AssertionError("Robot/table collision filter bit is invalid")
    if table_filter_bit == arm_cube_filter_bit:
        raise AssertionError("Table and arm/cube filters share one bit")
    if not all(int(shape.filter) & table_filter_bit for shape in robot_shapes):
        raise AssertionError("Not every robot shape filters the table")
    if not all(int(shape.filter) & table_filter_bit for shape in table_shapes):
        raise AssertionError("Table does not filter robot collisions")
    if not all(int(shape.filter) & arm_cube_filter_bit for shape in cube_shapes):
        raise AssertionError("Cube does not carry the arm collision-filter bit")

    asset_body_names = tuple(
        env.gym.get_asset_rigid_body_names(env.robot_asset)
    )
    asset_shape_ranges = env.gym.get_asset_rigid_body_shape_indices(
        env.robot_asset
    )
    asset_shapes = env.gym.get_asset_rigid_shape_properties(env.robot_asset)
    for body_name, shape_range in zip(asset_body_names, asset_shape_ranges):
        shapes = asset_shapes[
            shape_range.start:shape_range.start + shape_range.count
        ]
        if body_name in env.arm_collision_body_names:
            if not all(int(shape.filter) & arm_cube_filter_bit for shape in shapes):
                raise AssertionError(
                    "Arm body {!r} still collides with the cube".format(body_name)
                )
        elif body_name in env.hand_collision_body_names:
            if any(int(shape.filter) & arm_cube_filter_bit for shape in shapes):
                raise AssertionError(
                    "Hand body {!r} is filtered from the cube".format(body_name)
                )
        else:
            raise AssertionError(
                "Collision body {!r} belongs to neither group".format(body_name)
            )
    if any(
        int(table.filter) & int(cube.filter)
        for table in table_shapes
        for cube in cube_shapes
    ):
        raise AssertionError("Table-cube collisions are filtered")

    cube_body = env.gym.get_actor_rigid_body_properties(
        env.envs[0], env.cube_handles[0]
    )[0]
    expected_inertia = np.asarray(env.cfg.object.inertia_kg_m2, dtype=np.float64)
    actual_inertia = np.asarray(
        [cube_body.inertia.x.x, cube_body.inertia.y.y, cube_body.inertia.z.z],
        dtype=np.float64,
    )
    if not np.isclose(cube_body.mass, env.cfg.object.mass_kg, atol=1e-7):
        raise AssertionError("Cube mass does not match configuration")
    np.testing.assert_allclose(actual_inertia, expected_inertia, rtol=0.0, atol=1e-8)
    if not all(
        np.isclose(shape.friction, env.cfg.object.friction, atol=1e-7)
        and np.isclose(shape.restitution, env.cfg.object.restitution, atol=1e-7)
        for shape in cube_shapes
    ):
        raise AssertionError("Cube material does not match configuration")


def assert_vectorized_object_rsi_contract(env):
    """Check distinct per-env RSI states and one indexed partial reset.

    Every reset here pins the near-identity transform. Left to sample its own,
    each reset would also move the bar, and the comparison would measure the
    transform rather than the RSI write it is meant to check.
    """
    import torch

    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    identity = identity_transform_index(env)
    transforms = torch.full_like(env_ids, identity)
    env.reset_idx(env_ids, None, transforms)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    sampled_expected = env._cube_reference_root_states(
        env.transform_bank.sample(env.transform_index, env.reference_index)
    )
    if not bool(
        torch.allclose(
            env.cube_root_state, sampled_expected, rtol=0.0, atol=CUBE_POSE_ATOL
        )
    ):
        raise AssertionError("Configured RSI did not reset cubes from sampled phases")
    if bool((env.reference_index > env.rsi_max_start_index).any()):
        raise AssertionError("Automatic RSI sampled a forbidden post-grasp frame")

    spread = torch.linspace(
        0,
        env.reference.last_index - 1,
        steps=env.num_envs,
        device=env.device,
    ).round().long()
    env.reset_idx(env_ids, spread, transforms)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    expected = env._cube_reference_root_states(
        env.transform_bank.sample(env.transform_index, spread)
    )
    if not bool(
        torch.allclose(env.cube_root_state, expected, rtol=0.0, atol=CUBE_POSE_ATOL)
    ):
        raise AssertionError("Vectorized RSI did not apply each cube reference state")

    if env.num_envs > 1:
        before = env.cube_root_state.clone()
        partial_env_ids = env_ids[:1]
        partial_reference = torch.tensor(
            [min(850, env.reference.last_index - 1)],
            dtype=torch.long,
            device=env.device,
        )
        env.reset_idx(partial_env_ids, partial_reference, transforms[:1])
        env.gym.refresh_actor_root_state_tensor(env.sim)
        expected_partial = env._cube_reference_root_states(
            env.transform_bank.sample(
                env.transform_index[:1], partial_reference
            )
        )
        if not bool(
            torch.allclose(
                env.cube_root_state[:1],
                expected_partial,
                rtol=0.0,
                atol=CUBE_POSE_ATOL,
            )
        ):
            raise AssertionError("Partial RSI did not reset the selected cube")
        if not bool(
            torch.equal(env.cube_root_state[1:], before[1:])
        ):
            raise AssertionError("Partial RSI changed a non-selected cube")


def assert_continuous_placement_contract(env):
    """A caller-chosen placement must move the cuboid exactly, not snap it.

    The interactive evaluator (scripts/evaluate_viser.py) places the cuboid at a
    continuous transform while the arm reference comes from the nearest bank
    entry -- the same approximation training makes, where the bank is only the
    nearest-neighbour source for arm IK. If the placement were silently snapped
    to the bank instead, every measurement taken through that GUI would describe
    a pose other than the one on screen.

    The offset below is deliberately small compared with the bank's spacing so
    the same entry stays nearest, which is what lets the two resets be compared
    exactly: same reference, same yaw, translations differing by the offset.
    """
    import torch

    from simtoolreal_animrl.envs.transform_bank import nearest_transform_indices

    env_ids = env.all_env_ids
    count = env_ids.numel()
    frame_zero = torch.zeros(count, dtype=torch.long, device=env.device)
    index = identity_transform_index(env)
    indices = torch.full_like(env_ids, index)
    bank_translation = env.transform_bank.translation[indices]
    bank_yaw = env.transform_bank.yaw_rad[indices]

    env.reset_idx(env_ids, frame_zero, indices)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    snapped_position = env.cube_position.clone()

    offset = torch.tensor([0.004, -0.003, 0.0], device=env.device)
    requested = bank_translation + offset
    env.reset_idx(
        env_ids,
        frame_zero,
        None,
        episode_translation=requested,
        episode_yaw_rad=bank_yaw,
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)

    if not bool(torch.allclose(env.episode_translation, requested, atol=1e-6)):
        raise AssertionError(
            "A requested continuous placement was not stored verbatim"
        )
    if not bool(torch.allclose(env.episode_yaw_rad, bank_yaw, atol=1e-6)):
        raise AssertionError("A requested yaw was not stored verbatim")
    expected_index = nearest_transform_indices(
        requested,
        bank_yaw,
        env.transform_bank.translation,
        env.transform_bank.yaw_rad,
        float(env.cfg.object_randomization.nearest_yaw_lever_arm_m),
    )
    if not bool(torch.equal(env.transform_index, expected_index)):
        raise AssertionError(
            "A continuous placement did not resolve to the nearest bank entry"
        )
    if not bool(torch.equal(env.transform_index, indices)):
        raise AssertionError(
            "The test offset is too large: a different bank entry won, so the "
            "two resets no longer share a reference and cannot be compared"
        )
    measured_shift = env.cube_position - snapped_position
    deviation = (measured_shift - offset.expand_as(measured_shift)).abs()
    worst = int(deviation.max(dim=1).values.argmax())
    # Both operands are metre-scale float32 world positions, so differencing
    # them to recover a 4 mm offset cancels most of the mantissa. Report the
    # worst environment rather than env 0, which can agree while another does
    # not -- a snap to the bank would miss by millimetres, not by this.
    if float(deviation.max()) > CUBE_POSE_ATOL:
        raise AssertionError(
            "The cuboid was snapped to the bank instead of placed at the "
            "requested transform: worst env {} shifted {}, expected {} "
            "(max deviation {:.3e})".format(
                worst,
                measured_shift[worst].tolist(),
                offset.tolist(),
                float(deviation.max()),
            )
        )

    try:
        env.reset_idx(
            env_ids,
            frame_zero,
            indices,
            episode_translation=requested,
            episode_yaw_rad=bank_yaw,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Passing both transform_indices and a continuous placement must "
            "raise: one of the two would have been silently ignored"
        )


def assert_observation_contract(env):
    import torch

    # Recompute first. reset_idx writes state but does not refill the
    # observation buffer, so whatever a caller reset last would otherwise be
    # compared against a buffer built for an earlier pose -- which measures
    # staleness rather than the block layout this checks.
    env.compute_observations()
    obs = env.get_observations()
    # Read the width from the configuration rather than pinning a literal: the
    # observation has already grown from 79 to 108 to 112, and a hardcoded
    # number here fails on the widening rather than on a real defect.
    if obs.shape != (env.num_envs, env.num_obs):
        raise AssertionError(
            "Expected {}D observations, got {}".format(env.num_obs, obs.shape)
        )

    expected_phase = (
        env.reference_index.float() / float(env.reference.last_index)
    ).unsqueeze(1)
    task_space = env._task_space_observation_components()
    expected = torch.cat(
        (
            env.normalize_positions(env.q),
            env.previous_targets,
            env.dq,
            expected_phase,
            *task_space,
        ),
        dim=1,
    )
    if not bool(torch.allclose(obs, expected, rtol=0.0, atol=1e-6)):
        raise AssertionError("The observation blocks are inconsistent")

    if not bool(((obs[:, :6] >= -1.0) & (obs[:, :6] <= 1.0)).all()):
        raise AssertionError("Normalized arm positions escaped [-1, 1]")

    # palm position, palm rotation as the continuous 6D representation,
    # five palm-relative fingertips, cube rotation as 6D, cube centre.
    block_widths = (3, 6, 15, 6, 3)
    if tuple(component.shape[1] for component in task_space) != block_widths:
        raise AssertionError("Task-space observation block widths are incorrect")
    fingertip_positions_palm = task_space[2].reshape(env.num_envs, 5, 3)
    cube_center_palm = task_space[4].unsqueeze(1)
    observed_tip_cube_distances = torch.linalg.vector_norm(
        fingertip_positions_palm - cube_center_palm, dim=2
    )
    expected_tip_cube_distances = torch.linalg.vector_norm(
        env._fingertip_positions_world() - env.cube_position.unsqueeze(1),
        dim=2,
    )
    if not bool(
        torch.allclose(
            observed_tip_cube_distances,
            expected_tip_cube_distances,
            rtol=0.0,
            atol=1e-6,
        )
    ):
        raise AssertionError(
            "Fingertip observations are not palm-relative positions"
        )
    # Both rotations travel as the first two columns of their rotation matrix,
    # so the invariant is orthonormality of that pair -- not the unit norm and
    # canonical sign a quaternion carried before the representation changed.
    for name, rotation_6d in (
        ("palm", task_space[1]),
        ("cube relative to palm", task_space[3]),
    ):
        first, second = rotation_6d[:, 0:3], rotation_6d[:, 3:6]
        ones = torch.ones(env.num_envs, device=env.device)
        for column, values in (("first", first), ("second", second)):
            if not bool(
                torch.allclose(
                    torch.linalg.vector_norm(values, dim=1),
                    ones,
                    rtol=0.0,
                    atol=1e-5,
                )
            ):
                raise AssertionError(
                    "{} rotation 6D {} column is not a unit vector".format(
                        name, column
                    )
                )
        if not bool(
            torch.allclose(
                (first * second).sum(dim=1),
                torch.zeros_like(ones),
                rtol=0.0,
                atol=1e-5,
            )
        ):
            raise AssertionError(
                "{} rotation 6D columns are not orthogonal".format(name)
            )

    if not bool(torch.isfinite(obs).all()):
        raise AssertionError("The observation contains NaN or infinity")


def assert_reward_contract(env):
    import torch

    metrics = env._compute_reward_and_errors()
    if not bool(
        torch.allclose(
            metrics["object_com_height_m"],
            env.cube_position[:, 2],
            rtol=0.0,
            atol=0.0,
        )
    ):
        raise AssertionError("Cube COM height is not the cube root world z")
    expected_lift = (
        env.cube_position[:, 2] - env.episode_initial_object_com_height_m
    )
    if not bool(
        torch.allclose(
            metrics["object_com_lift_m"], expected_lift, rtol=0.0, atol=0.0
        )
    ):
        raise AssertionError("Cube COM lift is not relative to the RSI reset height")
    if metrics["q_error"].shape != (env.num_envs, 6):
        raise AssertionError("Position reward is not restricted to the arm")
    if metrics["dq_error"].shape != (env.num_envs, 6):
        raise AssertionError("Velocity reward is not restricted to the arm")
    if metrics["hand_q_error"].shape != (env.num_envs, 20):
        raise AssertionError("Hand diagnostics have an unexpected shape")
    for name in (
        "palm_tilt_reward",
        "palm_tilt_error_rad",
        "ee_action_rate_reward",
        "arm_joint_rate_reward",
        "ik_residual_reward",
        "ik_residual_norm",
        "arm_joint_delta_clipped",
    ):
        if metrics[name].shape != (env.num_envs,):
            raise AssertionError(
                "{} has an unexpected shape".format(name)
            )
    if metrics["object_position_error_m"].shape != (env.num_envs,):
        raise AssertionError("Object position error has an unexpected shape")
    if metrics["object_orientation_error_rad"].shape != (env.num_envs,):
        raise AssertionError("Object orientation error has an unexpected shape")
    if metrics["fingertip_object_distance_reward"].shape != (env.num_envs,):
        raise AssertionError("Fingertip-distance reward has an unexpected shape")
    proximity_active = env.reference_index >= env.rsi_pregrasp_start_index
    if not bool(
        (
            (metrics["fingertip_object_distance_reward"] >= 0.0)
            & (metrics["fingertip_object_distance_reward"] <= 1.0)
            & (metrics["fingertip_object_distance_m"] >= 0.0)
        ).all()
    ):
        raise AssertionError("Fingertip-distance diagnostics are outside their bounds")
    if bool(
        (metrics["fingertip_object_distance_reward"][~proximity_active] != 0.0).any()
    ):
        raise AssertionError("Fingertip-distance reward activated before pre-grasp")
    if bool(proximity_active.any()) and bool(
        (metrics["fingertip_object_distance_reward"][proximity_active] <= 0.0).any()
    ):
        off = proximity_active & (
            metrics["fingertip_object_distance_reward"] <= 0.0
        )
        distances = metrics["fingertip_object_distance_m"][off]
        raise AssertionError(
            "Fingertip-distance reward stayed off during pre-grasp in {} of {} "
            "environments: distances {:.4f}..{:.4f} m against std {:.3f} m, "
            "reference indices {}..{}".format(
                int(off.sum()),
                int(proximity_active.sum()),
                float(distances.min()),
                float(distances.max()),
                float(env.cfg.rewards.fingertip_object_distance_std_m),
                int(env.reference_index[off].min()),
                int(env.reference_index[off].max()),
            )
        )
    if metrics["fingertip_contact_reward"].shape != (env.num_envs,):
        raise AssertionError("Fingertip-contact reward has an unexpected shape")
    expected_max_contacts = float(len(env.contact_fingertip_names))
    if not bool(
        (
            (metrics["fingertip_contact_reward"] >= 0.0)
            & (metrics["fingertip_contact_reward"] <= expected_max_contacts)
            & (metrics["fingertip_contact_fraction"] >= 0.0)
            & (metrics["fingertip_contact_fraction"] <= 1.0)
            & (metrics["mean_fingertip_contact_force_n"] >= 0.0)
        ).all()
    ):
        raise AssertionError("Fingertip-contact diagnostics are outside their bounds")
    if not bool(
        (
            (metrics["object_position_reward"] >= 0.0)
            & (metrics["object_position_reward"] <= 1.0)
            & (metrics["object_orientation_reward"] >= 0.0)
            & (metrics["object_orientation_reward"] <= 1.0)
        ).all()
    ):
        raise AssertionError("Object Gaussian rewards escaped [0, 1]")
    r = env.cfg.rewards
    expected_reward = (
        float(r.palm_tilt_weight) * metrics["palm_tilt_reward"]
        + float(r.ee_action_rate_weight) * metrics["ee_action_rate_reward"]
        + float(r.arm_joint_rate_weight) * metrics["arm_joint_rate_reward"]
        + float(r.ik_residual_weight) * metrics["ik_residual_reward"]
        + float(r.position_hand_weight) * metrics["hand_position_reward"]
        + float(r.velocity_hand_weight) * metrics["hand_velocity_reward"]
        + float(r.hand_action_rate_weight) * metrics["hand_action_rate_reward"]
        + float(r.object_position_weight) * metrics["object_position_reward"]
        + float(r.object_orientation_weight)
        * metrics["object_orientation_reward"]
        + float(r.fingertip_object_distance_weight)
        * metrics["fingertip_object_distance_reward"]
        # The two heaviest terms in the sum, and they were missing here: 1.28
        # of weight that the reconstruction never checked, so any change to
        # either of them passed this assertion untouched.
        + float(r.palm_keypoint_weight) * metrics["palm_keypoint_reward"]
        + float(r.fingertip_keypoint_weight)
        * metrics["fingertip_keypoint_reward"]
        # contact_shaping_weight, not contact_reward_per_finger: the two agree
        # only while contact.reward_enabled is True, and the env sums the
        # former.
        + float(env.contact_shaping_weight)
        * metrics["fingertip_contact_reward"]
    )
    if not bool(torch.allclose(env.rew_buf, expected_reward, rtol=0.0, atol=1e-6)):
        residual = (env.rew_buf - expected_reward)
        raise AssertionError(
            "Reward does not match the configured weights: residual "
            "{:+.6f}..{:+.6f} against a reward of {:.4f}..{:.4f}".format(
                float(residual.min()),
                float(residual.max()),
                float(env.rew_buf.min()),
                float(env.rew_buf.max()),
            )
        )


def assert_palm_jacobian_matches_urdf(env):
    """The Isaac Gym palm Jacobian against pytorch_kinematics on the URDF.

    This one assertion covers every way the task-space controller can be wired
    to the wrong numbers: the Jacobian's DOF columns are in asset order and
    demo_to_asset is a real permutation, the link axis is offset by one for a
    fixed-base actor, and the palm is a point transfer off wrist_3_link rather
    than a body Isaac Gym reports. Any of those being wrong still produces a
    smoothly moving arm, just not one that moves where the reward is measured.

    Checked at several reference indices because a transfer error scales with the
    wrist's angular rate and would hide at a single pose.
    """
    import torch

    kinematics = env._palm_kinematics
    worst = 0.0
    hold = torch.zeros(
        (env.num_envs, env.num_actions), dtype=torch.float32, device=env.device
    )
    for reference_index in (0, 200, 400, 600, 740, 830):
        env.reset(reference_index=reference_index)
        # Isaac Gym does not propagate a DOF-state write into the rigid-body or
        # Jacobian tensors until physics runs, so compare only after a step --
        # otherwise this reads a Jacobian for the previous configuration and
        # both sides describe different poses.
        env.step(hold)
        expected = kinematics.jacobian(env.arm_q.double().cpu())
        actual = env._palm_jacobian_arm().double().cpu()
        if actual.shape != expected.shape:
            raise AssertionError(
                "Palm Jacobian has shape {}, expected {}".format(
                    tuple(actual.shape), tuple(expected.shape)
                )
            )
        worst = max(worst, float((actual - expected).abs().max()))
    # float32 Isaac Gym against float64 pytorch_kinematics.
    if worst > 2e-4:
        raise AssertionError(
            "Isaac Gym palm Jacobian differs from the URDF by {:.3e}; suspect "
            "the asset-order DOF columns, the link index, or the point "
            "transfer".format(worst)
        )


def assert_zero_twist_holds_the_arm(env):
    """A zero action must mean "hold", and must cost nothing in residual."""
    import torch

    env.reset(reference_index=400)
    before = env.previous_arm_targets.clone()
    actions = torch.zeros(
        (env.num_envs, env.num_actions), dtype=torch.float32, device=env.device
    )
    for _ in range(50):
        env.step(actions)
    drift = float((env.previous_arm_targets - before).abs().max())
    if drift > 1e-5:
        raise AssertionError(
            "A zero twist moved the arm target by {:.3e} rad".format(drift)
        )
    residual = float(env.ik_residual_norm.max())
    if residual > 1e-5:
        raise AssertionError(
            "A zero twist left an IK residual of {:.3e}".format(residual)
        )


def assert_infeasible_commands_are_reported(env):
    """The feasibility signal has to actually reach the diagnostics.

    An impossible twist must show up as a saturated clamp and a large residual.
    If these stayed silent the whole design would be unobservable: the policy
    would be free to ask for the impossible and nothing would say so.
    """
    import torch

    env.reset(reference_index=400)
    actions = torch.zeros(
        (env.num_envs, env.num_actions), dtype=torch.float32, device=env.device
    )

    # An over-range request must be bounded in magnitude and left pointing the
    # way it was asked to point. Per-component clipping would satisfy the first
    # and quietly break the second.
    actions[:, 0] = 10.0
    actions[:, 1] = 20.0
    env.step(actions)
    translation = env.requested_twist[:, :3]
    limit = env.arm_translation_speed * env.dt
    if float(translation.norm(dim=1).max()) > limit * (1.0 + 1e-6):
        raise AssertionError("The commanded translation exceeded the speed limit")
    # (10, 20, 0) normalised is (0.4472, 0.8944, 0); clipping each component
    # would have returned (0.7071, 0.7071, 0) instead.
    direction = translation / translation.norm(dim=1, keepdim=True)
    expected = torch.tensor(
        [10.0, 20.0, 0.0], device=env.device, dtype=direction.dtype
    )
    expected = expected / expected.norm()
    if not bool(torch.allclose(direction, expected.expand_as(direction), atol=1e-5)):
        raise AssertionError(
            "Saturating the twist turned it: got {}, expected {}".format(
                direction[0].tolist(), expected.tolist()
            )
        )

    # In the regime the policy actually operates in -- tracking the reference --
    # the solver must deliver essentially all of what it is asked for. Measured
    # at 0.5% lost and the clamp never binding; a full-scale command is a
    # different matter and saturates about 17% of the time, which is the clamp
    # doing its job as a joint-speed limit rather than a defect.
    env.reset(reference_index=400)
    reference_actions, _ = env.next_reference_action()
    env.step(reference_actions)
    requested_norm = env.requested_twist.norm(dim=1).clamp_min(1e-12)
    residual_fraction = float((env.ik_residual_norm / requested_norm).max())
    if residual_fraction > 0.05:
        raise AssertionError(
            "Tracking the reference lost {:.1%} of the commanded twist; "
            "suspect ik_damping".format(residual_fraction)
        )
    if bool((env.arm_joint_delta_clipped > 0.5).any()):
        raise AssertionError(
            "The per-joint IK clamp bound while merely tracking the reference"
        )

    # Now force the clamp to bind, and check the report reaches the outside.
    # Shrinking the clamp is the cheapest way to make any command infeasible;
    # the mechanism it exercises -- applied delta read back, pushed through the
    # Jacobian, differenced against the request -- is the same one a joint limit
    # or a singularity triggers in training.
    original = env.ik_max_joint_delta
    env.ik_max_joint_delta = 1e-5
    try:
        env.step(actions)
        if not bool((env.arm_joint_delta_clipped > 0.5).all()):
            raise AssertionError(
                "A clamped IK step did not set the saturation flag"
            )
        if float(env.ik_residual_norm.min()) <= 1e-4:
            raise AssertionError(
                "A clamped IK step reported no unresolved residual"
            )
        achieved = float(env.achieved_twist.abs().max())
        requested = float(env.requested_twist.abs().max())
        if achieved >= requested:
            raise AssertionError(
                "A clamped step claims to have delivered the full twist"
            )
    finally:
        env.ik_max_joint_delta = original


def assert_the_ik_runs_once_per_step(env):
    """The arm mapping is a stateful integrator, so a second call double-counts.

    This is the bug the scale_actions -> command_targets rename exists to
    prevent, so it is worth pinning rather than trusting.
    """
    import torch

    calls = {"count": 0}
    original = env._operational_space_arm_targets

    def counting(arm_actions):
        calls["count"] += 1
        return original(arm_actions)

    env._operational_space_arm_targets = counting
    try:
        actions = torch.zeros(
            (env.num_envs, env.num_actions),
            dtype=torch.float32,
            device=env.device,
        )
        env.step(actions)
    finally:
        env._operational_space_arm_targets = original
    if calls["count"] != 1:
        raise AssertionError(
            "The IK ran {} times in one step; it integrates onto the previous "
            "target, so it must run exactly once".format(calls["count"])
        )


def assert_demonstration_action_delta_reward(env):
    import torch

    original_actions = env.actions.clone()
    original_previous_actions = env.previous_actions.clone()
    reference = env.reference.sample(env.reference_index)
    # Hand only: the arm's regularizer is pure command smoothness and is not
    # compared against the demonstration, so there is no arm counterpart here.
    expected_delta = (
        reference.dq[:, 6:] * env.dt / env.hand_action_scale
    )
    if not bool(
        torch.allclose(
            env.demonstration_hand_action_delta(reference.dq),
            expected_delta,
            rtol=0.0,
            atol=1e-8,
        )
    ):
        raise AssertionError("Demonstration velocity was mapped incorrectly")

    env.actions.copy_(env.previous_actions)
    env.actions[:, 6:] += expected_delta
    matched = env._compute_reward_and_errors()
    if not bool(
        torch.allclose(
            matched["hand_action_rate_mse"],
            torch.zeros_like(matched["hand_action_rate_mse"]),
            rtol=0.0,
            # a_{t-1} can be O(10) in residual space; adding then subtracting
            # the small demonstrated delta loses a few float32 ulps.
            atol=1e-6,
        )
    ):
        raise AssertionError(
            "The demonstrated action delta does not maximize regularization"
        )

    env.actions.copy_(original_actions)
    env.previous_actions.copy_(original_previous_actions)
    env._compute_reward_and_errors()


def assert_ppo_step_contract(env, obs, critic_obs, rewards, dones, extras):
    import torch

    if obs.shape != (env.num_envs, env.num_obs):
        raise AssertionError("PPO observation shape mismatch")
    # Asymmetric actor-critic: the critic may see the fingertip forces, the
    # actor never does. Assert the shape contract rather than the absence,
    # which only held while contact.critic_observes_fingertip_forces was off.
    if not (env.critic_force_observation_dim or env.critic_parameter_dim):
        if critic_obs is not None:
            raise AssertionError(
                "A symmetric environment must not expose privileged observations"
            )
    else:
        if critic_obs is None:
            raise AssertionError(
                "The critic observation was configured but never produced"
            )
        if critic_obs.shape != (env.num_envs, env.num_privileged_obs):
            raise AssertionError("PPO critic observation shape mismatch")
        if not bool(torch.equal(critic_obs[:, : env.num_obs], obs)):
            raise AssertionError(
                "The critic observation must extend the actor's, not replace it"
            )
    if rewards.shape != (env.num_envs,) or rewards.dtype != torch.float32:
        raise AssertionError("PPO rewards must be float32 with shape (num_envs,)")
    if dones.shape != (env.num_envs,) or dones.dtype != torch.bool:
        raise AssertionError("PPO dones must be bool with shape (num_envs,)")

    required = (
        "time_outs",
        "horizon_time_outs",
        "reference_end",
        "early_termination",
    )
    for name in required:
        values = extras.get(name)
        if values is None or values.shape != (env.num_envs,):
            raise AssertionError("Missing or malformed PPO info {!r}".format(name))
        if values.dtype != torch.bool or values.device != env.device:
            raise AssertionError("PPO info {!r} has wrong dtype/device".format(name))

    if not bool(
        torch.equal(
            extras["time_outs"],
            extras["horizon_time_outs"] | extras["reference_end"],
        )
    ):
        raise AssertionError("time_outs is not horizon OR reference_end")
    if bool((extras["time_outs"] & extras["early_termination"]).any()):
        raise AssertionError("A termination was both timeout and task failure")

    if bool(dones.any()):
        episode = extras.get("episode")
        if episode is None:
            raise AssertionError("Completed episodes did not produce episode statistics")
        required_episode_keys = (
            "return",
            "length",
            "mean_reward",
            "mean_palm_tilt_reward",
            "mean_ee_action_rate_reward",
            "mean_arm_joint_rate_reward",
            "mean_ik_residual_reward",
            "mean_rms_position_error",
            "mean_rms_velocity_error",
            "mean_rms_ee_action_rate",
            "mean_rms_arm_joint_rate",
            "mean_arm_joint_delta_clipped",
            "early_termination_fraction",
            "horizon_fraction",
            "reference_end_fraction",
            "completed_episodes",
            "mean_peak_object_com_height_m",
            "max_peak_object_com_height_m",
            "mean_peak_object_com_lift_m",
            "max_peak_object_com_lift_m",
        )
        for name in required_episode_keys:
            value = episode.get(name)
            if value is None or value.ndim != 0 or not bool(torch.isfinite(value)):
                raise AssertionError(
                    "Missing, non-scalar, or non-finite episode statistic {!r}".format(
                        name
                    )
                )
        if int(episode["completed_episodes"]) != int(dones.sum()):
            raise AssertionError("Episode completion count does not match dones")
    elif "episode" in extras:
        raise AssertionError("Episode statistics were emitted without a completed episode")


def run_ideal_episode(env, initial_indices):
    import torch

    peak_position_error = 0.0
    peak_velocity_error = 0.0
    minimum_reward = float("inf")
    mean_rewards = []
    reference_end_count = 0
    horizon_count = 0

    # One initial episode per environment. With Cartwheel-style RSI, episodes
    # near the end of the motion are intentionally shorter than the horizon.
    expected_steps = torch.minimum(
        torch.full_like(initial_indices, env.max_episode_length),
        env.reference.last_index - initial_indices,
    )
    pending = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    observed_steps = torch.zeros_like(initial_indices)
    peak_arm_action = 0.0
    peak_arm_action_where = None
    peak_palm_keypoint_error = 0.0
    peak_lifting_palm_keypoint_error = 0.0

    for step in range(1, env.max_episode_length + 1):
        actions, complete_target = env.next_reference_action()
        # The hand's residual mapping is still exactly invertible. The arm's is
        # not, by construction -- an integrator through a damped, clamped,
        # limit-saturated IK has no pointwise inverse -- so the arm is held to a
        # task-space standard below instead: it must track the reference palm
        # without terminating, using well under the full action range.
        reconstructed_hand = env.scale_hand_actions(actions[:, 6:])
        if not bool(
            torch.allclose(
                reconstructed_hand, complete_target[:, 6:], rtol=0.0, atol=1e-6
            )
        ):
            raise AssertionError("The hand residual mapping is not invertible")
        # Only environments still inside their first episode. Once one ends it
        # is reset onto a fresh RSI index, and the correction its next action
        # asks for describes that reset, not the demonstration this bound is
        # about -- measured as a peak of 0.814 against 0.39 for the tracked
        # population.
        tracked = actions[pending, :6].abs()
        if tracked.numel():
            step_peak = float(tracked.max())
            if step_peak > peak_arm_action:
                peak_arm_action = step_peak
                rows = pending.nonzero(as_tuple=False).flatten()
                worst = int(rows[int(tracked.max(dim=1).values.argmax())])
                peak_arm_action_where = (
                    step,
                    int(env.reference_index[worst]),
                    int(initial_indices[worst]),
                    int(actions[worst, :6].abs().argmax()),
                )

        obs, critic_obs, rewards, dones, extras = env.step(actions)
        assert_ppo_step_contract(
            env, obs, critic_obs, rewards, dones, extras
        )
        if not bool(torch_isfinite(rewards)):
            raise AssertionError("Reward contains NaN or infinity")
        r = env.cfg.rewards
        reward_upper_bound = (
            float(r.palm_tilt_weight)
            + float(r.ee_action_rate_weight)
            + float(r.arm_joint_rate_weight)
            + float(r.ik_residual_weight)
            + float(r.position_hand_weight)
            + float(r.velocity_hand_weight)
            + float(r.hand_action_rate_weight)
            + float(r.object_position_weight)
            + float(r.object_orientation_weight)
            + float(r.fingertip_object_distance_weight)
            + float(r.palm_keypoint_weight)
            + float(r.fingertip_keypoint_weight)
            + float(env.contact_shaping_weight)
            * float(len(env.contact_fingertip_names))
        )
        if bool((rewards < -1e-7).any()) or bool(
            (rewards > reward_upper_bound + 1e-6).any()
        ):
            raise AssertionError(
                "Gaussian weighted reward is outside [0, {}]".format(
                    reward_upper_bound
                )
            )

        peak_position_error = max(
            peak_position_error, float(extras["max_abs_position_error"].max())
        )
        peak_velocity_error = max(
            peak_velocity_error, float(extras["rms_velocity_error"].max())
        )
        minimum_reward = min(minimum_reward, float(rewards.min()))
        mean_rewards.append(float(rewards.mean()))

        # The palm keypoints are anchored on the *reference* bar
        # (docs/adr/0001), so this error is now an absolute deviation from the
        # demonstrated palm trajectory rather than relative grasp geometry.
        # Under ideal playback it must stay near zero through the lift: if the
        # anchor or its symmetry handling were wrong, the two would disagree
        # exactly where the reference bar leaves the table, and nowhere else.
        palm_error = extras["palm_keypoint_error_m"]
        peak_palm_keypoint_error = max(
            peak_palm_keypoint_error, float(palm_error.max())
        )
        lifting = extras["reference_index"] >= LIFT_START_REFERENCE_INDEX
        if bool(lifting.any()):
            peak_lifting_palm_keypoint_error = max(
                peak_lifting_palm_keypoint_error,
                float(palm_error[lifting].max()),
            )

        still_active = pending & ~dones
        if bool(still_active.any()):
            expected_hand_target = complete_target[
                still_active, len(ARM_JOINT_NAMES):
            ]
            actual_hand_target = env.previous_hand_targets[still_active]
            # The hand is policy-driven now; under the ideal reference action
            # its target must still reconstruct the demonstration exactly.
            if not bool(
                torch.allclose(
                    actual_hand_target,
                    expected_hand_target,
                    rtol=0.0,
                    atol=1e-6,
                )
            ):
                raise AssertionError(
                    "Hand position targets do not reconstruct the demonstration"
                )

        early_count = int((extras["early_termination"] & pending).sum())
        if early_count:
            failed = (
                extras["early_termination"] & pending
            ).nonzero(as_tuple=False).flatten()
            details = []
            for env_id in failed[:5].tolist():
                joint_index = int(extras["worst_joint_index"][env_id])
                details.append(
                    "env={} start={} ref={} joint={} arm_err={:.4f} hand_err={:.4f}".format(
                        env_id,
                        int(initial_indices[env_id]),
                        int(extras["reference_index"][env_id]),
                        ARM_JOINT_NAMES[joint_index],
                        float(extras["max_abs_arm_position_error"][env_id]),
                        float(extras["max_abs_hand_position_error"][env_id]),
                    )
                )
            raise AssertionError(
                "{} environments terminated early at ideal-reference step {}: {}".format(
                    early_count, step, "; ".join(details)
                )
            )

        expected_now = pending & (expected_steps == step)
        observed_now = pending & dones
        if not bool(torch.equal(expected_now, observed_now)):
            raise AssertionError(
                "RSI episode termination mismatch at step {}: expected {}, got {}".format(
                    step, int(expected_now.sum()), int(observed_now.sum())
                )
            )
        if bool(observed_now.any()) and not bool(
            extras["time_outs"][observed_now].all()
        ):
            raise AssertionError(
                "Reference-end/horizon termination was not classified as a timeout"
            )
        expected_reference_end = observed_now & (
            initial_indices + step >= env.reference.last_index
        )
        expected_horizon = observed_now & (step >= env.max_episode_length)
        if not bool(
            torch.equal(
                extras["reference_end"] & observed_now,
                expected_reference_end,
            )
        ):
            raise AssertionError("Reference-end classification mismatch")
        if not bool(
            torch.equal(
                extras["horizon_time_outs"] & observed_now,
                expected_horizon,
            )
        ):
            raise AssertionError("Horizon classification mismatch")
        reference_end_count += int(expected_reference_end.sum())
        horizon_count += int(expected_horizon.sum())
        observed_steps[observed_now] = step
        pending &= ~observed_now
        if not bool(pending.any()):
            break

    if bool(pending.any()) or not bool(torch.equal(observed_steps, expected_steps)):
        raise AssertionError("Not every initial RSI episode ended at the expected step")

    print(
        "    ideal playback: peak |arm action| {:.3f} at (step, ref, rsi "
        "start, channel) {}, peak palm keypoint error {:.4f} m "
        "(lift only {:.4f} m)".format(
            peak_arm_action,
            peak_arm_action_where,
            peak_palm_keypoint_error,
            peak_lifting_palm_keypoint_error,
        )
    )

    # The empirical check on control.arm_translation_speed_m_per_s. Tracking the
    # demonstration measured 0.373 at the configured scales, so the headroom
    # above that is what the policy has left for correcting a perturbation. A
    # materially larger number means the Jacobian, the frame or a speed scale is
    # wrong; the fix is to raise the speed scale, never to relax this bound.
    if peak_arm_action > 0.5:
        raise AssertionError(
            "Reference playback needs |arm action| up to {:.3f}, leaving no "
            "headroom under the +/-1 clip".format(peak_arm_action)
        )

    # Test (a) of docs/adr/0001: playing the demonstration back open-loop must
    # leave the reference-anchored palm term satisfied, the lift included. The
    # bound is the termination threshold, which the run above already enforces
    # implicitly; asserting it here separates "the anchor is wired correctly"
    # from "nothing terminated", which can fail for unrelated reasons.
    if peak_lifting_palm_keypoint_error > float(
        env.cfg.termination.palm_keypoint_threshold_m
    ):
        raise AssertionError(
            "Reference playback leaves {:.4f} m of palm keypoint error during "
            "the lift; the reference anchor or its symmetry handling is "
            "wrong".format(peak_lifting_palm_keypoint_error)
        )

    return {
        "peak_arm_action": peak_arm_action,
        "peak_palm_keypoint_error": peak_palm_keypoint_error,
        "peak_lifting_palm_keypoint_error": peak_lifting_palm_keypoint_error,
        "peak_position_error": peak_position_error,
        "peak_velocity_rms_error": peak_velocity_error,
        "minimum_reward": minimum_reward,
        "mean_reward": float(np.mean(mean_rewards)),
        "minimum_episode_steps": int(observed_steps.min()),
        "maximum_episode_steps": int(observed_steps.max()),
        "reference_end_count": reference_end_count,
        "horizon_count": horizon_count,
    }


def torch_isfinite(values):
    # Local import keeps the script's top-level import order unambiguous for
    # Isaac Gym.
    import torch

    return torch.isfinite(values).all()


def assert_early_termination_logic(env):
    import torch

    env.episode_length_buf.zero_()
    env.reference_index.zero_()
    env.arm_violation_steps.zero_()
    env.hand_violation_steps.zero_()
    env.object_violation_steps.zero_()
    # The arm's termination is task space now: one RMS palm keypoint error per
    # environment against palm_keypoint_threshold_m, not a per-joint angle
    # against a rad limit. Under the reference anchor (docs/adr/0001) that
    # error is absolute deviation from the demonstrated palm trajectory, which
    # is what makes a policy that never lifts terminate rather than collect
    # reward forever.
    arm_error = torch.zeros(
        env.num_envs, dtype=torch.float32, device=env.device
    )
    hand_error = torch.zeros(
        (env.num_envs, env.hand_q.shape[1]), dtype=torch.float32, device=env.device
    )
    object_error = torch.zeros(
        env.num_envs, dtype=torch.float32, device=env.device
    )
    arm_error.fill_(float(env.cfg.termination.palm_keypoint_threshold_m) + 0.1)

    grace = int(env.cfg.termination.grace_steps)
    for count in range(1, grace + 1):
        done, early, timeout = env._compute_termination(
            arm_error, hand_error, object_error
        )
        if bool(timeout.any()):
            raise AssertionError("Synthetic early-termination test unexpectedly timed out")
        expected = count >= grace
        if bool(early.all()) != expected or bool(done.all()) != expected:
            raise AssertionError(
                "Early termination grace mismatch at violation step {}".format(count)
            )
    if (
        not bool(env.arm_violation.all())
        or bool(env.hand_violation.any())
        or bool(env.object_violation.any())
    ):
        raise AssertionError("Arm-only violation was not attributed to the arm")

    # The hand alone must be able to end an episode.
    env.arm_violation_steps.zero_()
    env.hand_violation_steps.zero_()
    env.object_violation_steps.zero_()
    arm_error.zero_()
    hand_error[:, 0] = float(env.cfg.termination.hand_position_threshold_rad) + 0.1
    for count in range(1, grace + 1):
        done, early, timeout = env._compute_termination(
            arm_error, hand_error, object_error
        )
        expected = count >= grace
        if bool(early.all()) != expected or bool(done.all()) != expected:
            raise AssertionError(
                "Hand early termination grace mismatch at step {}".format(count)
            )
    if (
        not bool(env.hand_violation.all())
        or bool(env.arm_violation.any())
        or bool(env.object_violation.any())
    ):
        raise AssertionError("Hand-only violation was not attributed to the hand")

    # The optional cube condition must work when enabled, even though this
    # no-object-reward experiment keeps it disabled by default.
    original_object_enabled = env.cfg.termination.object_position_enabled
    env.cfg.termination.object_position_enabled = True
    env.arm_violation_steps.zero_()
    env.hand_violation_steps.zero_()
    env.object_violation_steps.zero_()
    hand_error.zero_()
    object_error.fill_(
        float(env.cfg.termination.object_position_threshold_m) + 0.01
    )
    for count in range(1, grace + 1):
        done, early, timeout = env._compute_termination(
            arm_error, hand_error, object_error
        )
        expected = count >= grace
        if bool(early.all()) != expected or bool(done.all()) != expected:
            raise AssertionError(
                "Object early termination grace mismatch at step {}".format(count)
            )
    if (
        not bool(env.object_violation.all())
        or bool(env.arm_violation.any())
        or bool(env.hand_violation.any())
    ):
        raise AssertionError("Object-only violation was not attributed to the object")

    # The object condition can be disabled without switching off the arm and
    # hand early-termination machinery.
    env.cfg.termination.object_position_enabled = False
    env.arm_violation_steps.zero_()
    env.hand_violation_steps.zero_()
    env.object_violation_steps.zero_()
    for _ in range(2 * grace):
        done, early, timeout = env._compute_termination(
            arm_error, hand_error, object_error
        )
        if bool(done.any()) or bool(early.any()) or bool(timeout.any()):
            raise AssertionError(
                "Disabled object early termination still ended an episode"
            )
    if bool(env.object_violation.any()) or bool(env.object_violation_steps.any()):
        raise AssertionError("Disabled object termination retained a violation")
    env.cfg.termination.object_position_enabled = original_object_enabled

    # The three counters are independent by design: a source that comes back
    # inside its threshold clears its own count, so alternating violations
    # never accumulate to the grace limit.
    env.arm_violation_steps.zero_()
    env.hand_violation_steps.zero_()
    env.object_violation_steps.zero_()
    arm_over = float(env.cfg.termination.palm_keypoint_threshold_m) + 0.1
    hand_over = float(env.cfg.termination.hand_position_threshold_rad) + 0.1
    object_over = float(env.cfg.termination.object_position_threshold_m) + 0.01
    for step in range(6 * grace):
        arm_error.zero_()
        hand_error.zero_()
        object_error.zero_()
        if step % 3 == 0:
            arm_error.fill_(arm_over)
        elif step % 3 == 1:
            hand_error[:, 0] = hand_over
        else:
            object_error.fill_(object_over)
        done, early, _ = env._compute_termination(
            arm_error, hand_error, object_error
        )
        if bool(early.any()) or bool(done.any()):
            raise AssertionError(
                "Alternating arm/hand/object violations terminated at step {}; the "
                "grace counters are not independent".format(step)
            )


def assert_early_termination_step_contract(env):
    """Force a task failure and verify that PPO must not bootstrap it."""
    original_arm_threshold = env.cfg.termination.palm_keypoint_threshold_m
    original_hand_threshold = env.cfg.termination.hand_position_threshold_rad
    grace = int(env.cfg.termination.grace_steps)
    env.reset(reference_index=0)

    try:
        # A negative diagnostic threshold makes every finite tracking error a
        # violation without perturbing the simulator state itself.
        env.cfg.termination.palm_keypoint_threshold_m = -1.0
        env.cfg.termination.hand_position_threshold_rad = -1.0
        for step in range(1, grace + 1):
            actions, _ = env.next_reference_action()
            obs, critic_obs, rewards, dones, extras = env.step(actions)
            assert_ppo_step_contract(
                env, obs, critic_obs, rewards, dones, extras
            )
            if step < grace and bool(dones.any()):
                raise AssertionError("Early termination ignored its grace period")

        if not bool(dones.all()):
            raise AssertionError("Forced task failures did not terminate")
        if bool(extras["time_outs"].any()):
            raise AssertionError("Task failures were incorrectly marked for bootstrap")
        if not bool(extras["early_termination"].all()):
            raise AssertionError("Task failures lack the early-termination flag")
        episode = extras["episode"]
        if abs(float(episode["length"]) - grace) > 1e-6:
            raise AssertionError("Early-termination episode length is wrong")
        if abs(float(episode["early_termination_fraction"]) - 1.0) > 1e-6:
            raise AssertionError("Early-termination episode statistics are wrong")
    finally:
        env.cfg.termination.palm_keypoint_threshold_m = original_arm_threshold
        env.cfg.termination.hand_position_threshold_rad = original_hand_threshold
        env.reset(reference_index=0)


def main():
    args = parse_args()
    cfg = SimToolRealCfg()
    sim_device = args.sim_device
    if args.cpu:
        sim_device = "cpu"
        cfg.sim.use_gpu_pipeline = False
        cfg.sim.physx.use_gpu = False
    cfg.viewer.training_camera_enabled = bool(args.training_camera)

    env = MotionImitationEnv(
        cfg,
        sim_device=sim_device,
        headless=True,
        num_envs_override=args.num_envs,
    )
    try:
        if args.rsi_index is not None:
            env.reset(reference_index=args.rsi_index)
        if args.training_camera:
            frame = env.capture_training_camera_frame()
            expected_shape = (
                int(cfg.viewer.training_camera_height),
                int(cfg.viewer.training_camera_width),
                3,
            )
            if frame.shape != expected_shape or frame.dtype != np.uint8:
                raise AssertionError(
                    "Training camera returned {} {}, expected {} uint8".format(
                        frame.shape, frame.dtype, expected_shape
                    )
                )

        initial_indices = env.reference_index.clone()
        reset_q_error, reset_dq_error, reset_cube_error = (
            assert_reset_matches_reference(env, args.rsi_index)
        )
        assert_object_scene_contract(env)
        assert_vectorized_object_rsi_contract(env)
        assert_continuous_placement_contract(env)
        env.reset_idx(env.all_env_ids, initial_indices)
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        # A DOF-state write does not move the links. Isaac Gym recomputes body
        # transforms during simulate(), and refresh_* re-reads its buffer
        # rather than recomputing it, so without a step every keypoint below is
        # read from the asset's default pose -- the palm a stale 1.2 m from the
        # bar, and both keypoint rewards reporting over a metre of error on a
        # state the simulator had never been in.
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        env.gym.refresh_jacobian_tensors(env.sim)
        assert_observation_contract(env)
        assert_reward_contract(env)
        assert_demonstration_action_delta_reward(env)
        assert_correct_pd_gains(env)
        # The task-space controller. These reset the env themselves, so they run
        # before the horizon walk below re-establishes the initial indices.
        assert_palm_jacobian_matches_urdf(env)
        assert_zero_twist_holds_the_arm(env)
        assert_infeasible_commands_are_reported(env)
        assert_the_ik_runs_once_per_step(env)
        env.reset_idx(env.all_env_ids, initial_indices)
        # Step before reading, for the same reason as the reset above: the
        # controller tests just above leave the arm somewhere else entirely,
        # and the playback's first palm Jacobian is transferred through the
        # wrist quaternion in the rigid-body tensor.
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        env.gym.refresh_jacobian_tensors(env.sim)
        # Exact joint targets do not guarantee that the dynamic cube remains
        # grasped, so exercise the complete horizon without the object-distance
        # reset here. Its threshold, grace period and opt-out behavior are
        # verified independently by assert_early_termination_logic().
        object_termination_enabled = env.cfg.termination.object_position_enabled
        env.cfg.termination.object_position_enabled = False
        try:
            metrics = run_ideal_episode(env, initial_indices)
        finally:
            env.cfg.termination.object_position_enabled = (
                object_termination_enabled
            )
        assert_observation_contract(env)
        assert_early_termination_logic(env)
        assert_early_termination_step_contract(env)

        print("HEADLESS ENVIRONMENT TEST PASSED")
        print("  environments              : {}".format(env.num_envs))
        print("  episode length            : {}".format(env.max_episode_length))
        print("  demonstration samples     : {}".format(env.reference.sample_count))
        print("  measured frequency [Hz]   : {:.6f}".format(env.reference.frequency_hz))
        print(
            "  initial RSI range         : [{}, {}]".format(
                int(initial_indices.min()), int(initial_indices.max())
            )
        )
        print("  reset max q error [rad]   : {:.3e}".format(reset_q_error))
        print("  reset max dq error [rad/s]: {:.3e}".format(reset_dq_error))
        print("  reset max cube-state error : {:.3e}".format(reset_cube_error))
        print(
            "  peak tracking error [rad] : {:.6f}".format(
                metrics["peak_position_error"]
            )
        )
        print(
            "  peak velocity RMS [rad/s] : {:.6f}".format(
                metrics["peak_velocity_rms_error"]
            )
        )
        print("  minimum reward            : {:.6f}".format(metrics["minimum_reward"]))
        print("  mean reward               : {:.6f}".format(metrics["mean_reward"]))
        print(
            "  initial episode steps     : [{}, {}]".format(
                metrics["minimum_episode_steps"],
                metrics["maximum_episode_steps"],
            )
        )
        print(
            "  initial termination types : reference_end={}, horizon={}".format(
                metrics["reference_end_count"], metrics["horizon_count"]
            )
        )
        print("  PD gains                  : verified")
        print("  cube/table physics        : verified")
        print("  collision filtering       : verified")
        print("  vectorized object RSI      : verified")
        print("  observation contract      : verified ({}D)".format(env.num_obs))
        print("  policy-driven hand        : verified")
        print("  no-object reward contract : verified")
        print("  demo action-delta reward  : verified")
        print("  PPO step/info contract     : verified")
        print("  horizon/reference timeout : verified")
        print("  early termination/no boot.: verified")
    finally:
        env.close()


if __name__ == "__main__":
    main()
