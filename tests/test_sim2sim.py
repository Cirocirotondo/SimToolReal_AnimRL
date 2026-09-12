import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from simtoolreal_animrl import ROOT_DIR
from simtoolreal_animrl.sim2sim.constants import (
    ACTION_DIM,
    BASE_OBSERVATION_DIM,
    PALM_ORIENTATION_IN_WRIST_XYZW,
    PALM_POSITION_IN_WRIST,
)
from simtoolreal_animrl.sim2sim.observation import (
    actions_to_position_targets,
    build_observation,
    normalize_canonical_quaternion,
    quat_conjugate_xyzw,
    quat_rotate_xyzw,
    quaternion_to_rotation_6d,
)
from simtoolreal_animrl.sim2sim.policy import (
    AnimRLInferencePolicy,
    load_saved_run,
)
from simtoolreal_animrl.sim2sim.plotting import save_rollout_plots


RUN_DIR = (
    ROOT_DIR
    / "logs/simtoolreal/2026-09-07_003258_pg830_blind512_n256"
)
CHECKPOINT = RUN_DIR / "best_model.pt"
CONFIG = RUN_DIR / "config.json"
EVALUATION = RUN_DIR / "eval_best_model.json"


class Sim2SimMathTest(unittest.TestCase):
    def test_quaternion_rotation_and_canonical_sign(self):
        half_sqrt = np.sqrt(0.5)
        quarter_turn_z = np.asarray((0.0, 0.0, half_sqrt, half_sqrt))
        rotated = quat_rotate_xyzw(quarter_turn_z, np.asarray((1.0, 0.0, 0.0)))
        np.testing.assert_allclose(rotated, (0.0, 1.0, 0.0), atol=1.0e-12)
        np.testing.assert_allclose(
            normalize_canonical_quaternion(-quarter_turn_z),
            quarter_turn_z,
            atol=1.0e-12,
        )

    def test_residual_action_mapping_uses_separate_arm_and_hand_scales(self):
        actions = np.ones(ACTION_DIM)
        defaults = np.linspace(-0.5, 0.5, ACTION_DIM)
        targets = actions_to_position_targets(
            actions,
            defaults,
            arm_scale=0.25,
            hand_scale=0.15,
            residual_clip=100.0,
        )
        np.testing.assert_allclose(targets[:6], defaults[:6] + 0.25)
        np.testing.assert_allclose(targets[6:], defaults[6:] + 0.15)

    def test_observation_has_the_saved_112_dimension_layout(self):
        lower = -np.ones(ACTION_DIM)
        upper = np.ones(ACTION_DIM)
        previous_targets = np.linspace(-0.4, 0.4, ACTION_DIM)
        velocities = np.linspace(-1.0, 1.0, ACTION_DIM)
        state = {
            "joint_positions": np.zeros(ACTION_DIM),
            "joint_velocities": velocities,
            "robot_position_world": np.zeros(3),
            "robot_orientation_world_xyzw": np.asarray((0.0, 0.0, 0.0, 1.0)),
            "wrist_position_world": np.zeros(3),
            "wrist_orientation_world_xyzw": np.asarray((0.0, 0.0, 0.0, 1.0)),
            "fingertip_body_positions_world": np.zeros((5, 3)),
            "fingertip_body_orientations_world_xyzw": np.tile(
                np.asarray((0.0, 0.0, 0.0, 1.0)), (5, 1)
            ),
            "cube_position_world": np.asarray((0.1, 0.2, 0.3)),
            "cube_orientation_world_xyzw": np.asarray((0.0, 0.0, 0.0, 1.0)),
        }
        observation = build_observation(
            state, previous_targets, 0.25, lower, upper
        )
        self.assertEqual(observation.shape, (BASE_OBSERVATION_DIM,))
        np.testing.assert_allclose(observation[0:26], 0.0)
        np.testing.assert_allclose(observation[26:52], previous_targets)
        np.testing.assert_allclose(observation[52:78], velocities)
        self.assertAlmostEqual(float(observation[78]), 0.25)
        np.testing.assert_allclose(observation[79:82], PALM_POSITION_IN_WRIST)
        # Both rotations are the continuous 6D encoding, not the quaternion:
        # 3 + 6 palm, 15 fingertips, 6 + 3 cube.
        np.testing.assert_allclose(
            observation[82:88],
            quaternion_to_rotation_6d(PALM_ORIENTATION_IN_WRIST_XYZW),
            rtol=1e-6,
        )
        self.assertEqual(observation[88:103].size, 15)
        np.testing.assert_allclose(
            observation[103:109],
            quaternion_to_rotation_6d(
                quat_conjugate_xyzw(PALM_ORIENTATION_IN_WRIST_XYZW)
            ),
            rtol=1e-6,
        )
        self.assertTrue(np.isfinite(observation).all())

    def test_rollout_plotter_writes_four_figures_and_raw_data(self):
        frame_count = 4
        values = np.arange(frame_count * ACTION_DIM, dtype=np.float64).reshape(
            frame_count, ACTION_DIM
        )
        trace = {
            "reference_indices": np.arange(10, 10 + frame_count),
            "policy_actions": values * 0.01,
            "reference_actions": values * 0.02,
            "action_deltas": values * 0.001,
            "actual_joint_positions": values * 0.03,
            "applied_position_targets": values * 0.04,
            "raw_position_targets": values * 0.05,
            "reference_joint_positions": values * 0.06,
        }
        with TemporaryDirectory() as directory:
            paths = save_rollout_plots(Path(directory), trace, show=False)
            self.assertEqual(
                set(paths),
                {
                    "data",
                    "arm_actions",
                    "hand_actions",
                    "arm_joint_tracking",
                    "hand_joint_tracking",
                },
            )
            for path in paths.values():
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)
            with np.load(paths["data"]) as saved:
                np.testing.assert_array_equal(
                    saved["reference_indices"], trace["reference_indices"]
                )
                np.testing.assert_allclose(
                    saved["action_deltas"], trace["action_deltas"]
                )


@unittest.skipUnless(
    CHECKPOINT.is_file() and CONFIG.is_file() and EVALUATION.is_file(),
    "the local blind checkpoint and evaluation artifacts are unavailable",
)
class SavedBlindCheckpointTest(unittest.TestCase):
    def test_saved_contract_is_blind_108_by_26(self):
        run = load_saved_run(CHECKPOINT, CONFIG)
        self.assertEqual(run.env_cfg["env"]["num_observations"], 112)
        self.assertEqual(run.env_cfg["env"]["num_actions"], 26)
        self.assertFalse(run.env_cfg["contact"]["observe_fingertip_forces"])
        self.assertFalse(run.env_cfg["object_assist"]["enabled"])

    def test_actor_reproduces_logged_deterministic_actions(self):
        run = load_saved_run(CHECKPOINT, CONFIG)
        actor = AnimRLInferencePolicy(run, device="cpu")
        trajectory = json.loads(EVALUATION.read_text(encoding="utf-8"))[
            "trajectory_env_0"
        ]
        observations = np.asarray(trajectory["observations"], dtype=np.float32)
        actions = np.asarray(trajectory["actions"], dtype=np.float32)
        # The evaluator records the post-step observation alongside action_t,
        # hence that observation is the input that produces action_(t+1).
        for index in (0, 200, 799, 900, 1105):
            actual = actor(observations[index])
            np.testing.assert_allclose(
                actual, actions[index + 1], rtol=0.0, atol=2.0e-6
            )


try:
    import mujoco  # noqa: F401

    from simtoolreal_animrl.envs.demonstration import JointDemonstration60Hz
    from simtoolreal_animrl.sim2sim.mujoco_sim import (
        AnimRLMujocoSim,
        MujocoSceneConfig,
    )

    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False


@unittest.skipUnless(
    MUJOCO_AVAILABLE and CHECKPOINT.is_file() and CONFIG.is_file(),
    "MuJoCo or the local blind run is unavailable",
)
class MujocoBackendTest(unittest.TestCase):
    @staticmethod
    def _pair_enabled(model, first, second):
        return bool(
            (
                int(model.geom_contype[first])
                & int(model.geom_conaffinity[second])
            )
            or (
                int(model.geom_contype[second])
                & int(model.geom_conaffinity[first])
            )
        )

    def test_only_hand_cube_and_cube_table_collision_pairs_are_enabled(self):
        run = load_saved_run(CHECKPOINT, CONFIG)
        config = MujocoSceneConfig.from_saved_config(
            ROOT_DIR, run.env_cfg, enable_viewer=False
        )
        with AnimRLMujocoSim(config) as simulation:
            cube = simulation.model.geom("cube_geom").id
            table = simulation.model.geom("table_geom").id
            floor = simulation.model.geom("floor").id
            enabled_pairs = set()
            for first in range(simulation.model.ngeom):
                for second in range(first + 1, simulation.model.ngeom):
                    if not self._pair_enabled(simulation.model, first, second):
                        continue
                    if cube in (first, second):
                        other = second if first == cube else first
                        if other == table:
                            enabled_pairs.add("cube-table")
                        else:
                            body = int(simulation.model.geom_bodyid[other])
                            self.assertTrue(
                                simulation._body_is_descendant_of(
                                    body, simulation._wrist_body_id
                                )
                            )
                            enabled_pairs.add("hand-cube")
                    else:
                        self.fail(
                            "Unexpected enabled pair: {} / {}".format(
                                simulation.model.geom(first).name,
                                simulation.model.geom(second).name,
                            )
                        )
            self.assertEqual(enabled_pairs, {"cube-table", "hand-cube"})
            self.assertEqual(int(simulation.model.geom_contype[floor]), 0)
            self.assertEqual(int(simulation.model.geom_conaffinity[floor]), 0)

    def test_per_joint_pd_gains_are_installed_in_mujoco_actuators(self):
        run = load_saved_run(CHECKPOINT, CONFIG)
        joint_kp = tuple(float(index + 1) for index in range(ACTION_DIM))
        joint_kv = tuple(float(index + 2) for index in range(ACTION_DIM))
        config = MujocoSceneConfig.from_saved_config(
            ROOT_DIR,
            run.env_cfg,
            enable_viewer=False,
            joint_kp=joint_kp,
            joint_kv=joint_kv,
        )
        np.testing.assert_allclose(config.joint_kp, joint_kp)
        np.testing.assert_allclose(config.joint_kv, joint_kv)
        with AnimRLMujocoSim(config) as simulation:
            np.testing.assert_allclose(
                simulation.model.actuator_gainprm[
                    simulation._actuator_ids, 0
                ],
                config.joint_kp,
            )
            np.testing.assert_allclose(
                simulation.model.actuator_biasprm[
                    simulation._actuator_ids, 2
                ],
                -np.asarray(config.joint_kv),
            )

    def test_reference_ghost_is_green_offset_and_kinematic(self):
        run = load_saved_run(CHECKPOINT, CONFIG)
        reference = JointDemonstration60Hz.load(
            ROOT_DIR / run.env_cfg["motion"]["file"],
            device="cpu",
            expected_hz=60.0,
        )
        config = MujocoSceneConfig.from_saved_config(
            ROOT_DIR,
            run.env_cfg,
            enable_viewer=False,
            enable_reference_ghost=True,
        )
        initial = reference.sample(np_to_long_tensor(0))
        later = reference.sample(np_to_long_tensor(100))
        with AnimRLMujocoSim(config) as simulation:
            simulation.reset(
                initial.q[0].numpy(),
                initial.dq[0].numpy(),
                initial.cube_pose[0].numpy(),
                initial.cube_linear_velocity[0].numpy(),
                initial.cube_angular_velocity[0].numpy(),
            )
            physical_before = simulation.get_state()["joint_positions"]
            simulation.set_reference_ghost(later.q[0].numpy())
            np.testing.assert_allclose(
                simulation.data.qpos[simulation._ghost_joint_qpos_adrs],
                later.q[0].numpy(),
                atol=1.0e-7,
            )
            np.testing.assert_allclose(
                simulation.get_state()["joint_positions"], physical_before
            )
            np.testing.assert_allclose(
                simulation.data.xpos[simulation._ghost_base_body_id],
                config.robot_position_world + config.reference_ghost_offset_world,
                atol=1.0e-10,
            )
            ghost_geom_ids = [
                geom_id
                for geom_id in range(simulation.model.ngeom)
                if simulation._body_is_descendant_of(
                    int(simulation.model.geom_bodyid[geom_id]),
                    simulation._ghost_base_body_id,
                )
            ]
            self.assertTrue(ghost_geom_ids)
            np.testing.assert_allclose(
                simulation.model.geom_rgba[ghost_geom_ids, :3],
                np.tile(config.reference_ghost_color, (len(ghost_geom_ids), 1)),
                atol=1.0e-7,
            )
            self.assertEqual(simulation.model.nu, ACTION_DIM)

    def test_reset_gravity_and_one_control_step(self):
        run = load_saved_run(CHECKPOINT, CONFIG)
        reference = JointDemonstration60Hz.load(
            ROOT_DIR / run.env_cfg["motion"]["file"],
            device="cpu",
            expected_hz=60.0,
        )
        config = MujocoSceneConfig.from_saved_config(
            ROOT_DIR, run.env_cfg, enable_viewer=False
        )
        sample = reference.sample(np_to_long_tensor(0))
        with AnimRLMujocoSim(config) as simulation:
            simulation.reset(
                sample.q[0].numpy(),
                sample.dq[0].numpy(),
                sample.cube_pose[0].numpy(),
                sample.cube_linear_velocity[0].numpy(),
                sample.cube_angular_velocity[0].numpy(),
            )
            state = simulation.get_state()
            expected_cube, _, _, _ = simulation.reference_cube_state_to_world(
                sample.cube_pose[0].numpy(),
                sample.cube_linear_velocity[0].numpy(),
                sample.cube_angular_velocity[0].numpy(),
            )
            np.testing.assert_allclose(
                state["joint_positions"], sample.q[0].numpy(), atol=1.0e-7
            )
            np.testing.assert_allclose(
                state["cube_position_world"], expected_cube, atol=1.0e-9
            )
            np.testing.assert_allclose(
                simulation.model.body_gravcomp[simulation._robot_body_ids()], 1.0
            )
            self.assertEqual(
                float(simulation.model.body_gravcomp[simulation._cube_body_id]),
                0.0,
            )
            observation = build_observation(
                state,
                sample.q[0].numpy(),
                0.0,
                simulation.joint_lower_limits,
                simulation.joint_upper_limits,
            )
            self.assertEqual(observation.shape, (112,))
            simulation.set_position_targets(sample.q[0].numpy())
            simulation.step_for(1.0 / 60.0)
            stepped = simulation.get_state()
            self.assertTrue(np.isfinite(stepped["joint_positions"]).all())
            self.assertTrue(np.isfinite(stepped["cube_position_world"]).all())

    def test_rsi_contact_settling_preserves_cube_and_relaxes_penetration(self):
        run = load_saved_run(CHECKPOINT, CONFIG)
        reference = JointDemonstration60Hz.load(
            ROOT_DIR / run.env_cfg["motion"]["file"],
            device="cpu",
            expected_hz=60.0,
        )
        config = MujocoSceneConfig.from_saved_config(
            ROOT_DIR, run.env_cfg, enable_viewer=False
        )
        sample = reference.sample(np_to_long_tensor(800))
        with AnimRLMujocoSim(config) as simulation:
            simulation.reset(
                sample.q[0].numpy(),
                sample.dq[0].numpy(),
                sample.cube_pose[0].numpy(),
                sample.cube_linear_velocity[0].numpy(),
                sample.cube_angular_velocity[0].numpy(),
            )
            cube_before = simulation.get_state()["cube_position_world"]
            arm_before = simulation.get_state()["joint_positions"][:6]
            result = simulation.settle_robot_cube_contacts(0.1)
            state = simulation.get_state()
            self.assertGreaterEqual(result["contacts_before"], 1)
            self.assertGreaterEqual(result["contacts_after"], 1)
            self.assertLess(result["minimum_distance_before_m"], -1.0e-4)
            self.assertGreater(
                result["minimum_distance_after_m"],
                result["minimum_distance_before_m"],
            )
            self.assertGreater(result["max_joint_displacement_rad"], 0.0)
            self.assertLess(result["max_joint_displacement_rad"], 0.05)
            np.testing.assert_allclose(
                state["cube_position_world"], cube_before, atol=1.0e-10
            )
            np.testing.assert_allclose(
                state["joint_positions"][:6], arm_before, atol=1.0e-10
            )
            np.testing.assert_allclose(
                state["cube_linear_velocity_world"], 0.0, atol=1.0e-12
            )


def np_to_long_tensor(index):
    import torch

    return torch.tensor([index], dtype=torch.long)


if __name__ == "__main__":
    unittest.main()
