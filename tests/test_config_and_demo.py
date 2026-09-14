import unittest

import torch

from simtoolreal_animrl import ROOT_DIR
from simtoolreal_animrl.cfg import SimToolRealCfg, SimToolRealTrainCfg
from simtoolreal_animrl.envs.demonstration import JointDemonstration60Hz


class ConfigAndDemoTest(unittest.TestCase):
    def test_animrl_configuration_values(self):
        env_cfg = SimToolRealCfg()
        train_cfg = SimToolRealTrainCfg()
        self.assertEqual(env_cfg.env.num_envs, 256)
        self.assertEqual(env_cfg.env.episode_length, 360)
        self.assertEqual(env_cfg.env.num_actions, 26)
        self.assertEqual(
            env_cfg.env.reference_init_distribution, "pregrasp_mixture"
        )
        self.assertEqual(env_cfg.env.rsi_early_probability, 0.20)
        self.assertEqual(env_cfg.env.rsi_pregrasp_start_index, 740)
        self.assertEqual(env_cfg.env.rsi_max_start_index, 830)
        self.assertEqual(
            env_cfg.control.action_parameterization, "operational_space_arm"
        )
        self.assertEqual(env_cfg.control.arm_translation_speed_m_per_s, 0.40)
        self.assertEqual(env_cfg.control.arm_rotation_speed_rad_per_s, 1.0)
        # Removed on purpose: a binding per-component clip made the policy's
        # range past the rail unreachable. Magnitude saturation replaced it.
        self.assertFalse(hasattr(env_cfg.control, "arm_action_clip"))
        self.assertEqual(env_cfg.control.ik_damping, 0.02)
        self.assertEqual(env_cfg.control.ik_max_joint_delta_rad, 0.05)
        # Deleted with the arm's joint-space path. Deployment and sim2sim still
        # read it and are meant to fail loudly, so its absence is the contract.
        self.assertFalse(hasattr(env_cfg.control, "scale_joint_target"))
        self.assertEqual(env_cfg.control.scale_hand_joint_target, 0.15)
        self.assertEqual(env_cfg.control.clip_joint_target, 100.0)
        # The fingers are allowed to pass through each other, which is what
        # makes the step roughly twice as fast; see asset.self_collision.
        self.assertFalse(env_cfg.asset.self_collision)
        self.assertEqual(env_cfg.object.size_m, [0.15, 0.05, 0.05])
        # 112, not the 108 every run before the object-centric reference used:
        # both rotations became the continuous 6D representation.
        self.assertEqual(env_cfg.env.num_observations, 112)
        # The reachable yaw envelope is asymmetric, so this is a (low, high)
        # pair rather than a +/- scalar. Measured, not chosen.
        self.assertEqual(env_cfg.object_randomization.translation_x_min_m, -0.09)
        self.assertEqual(env_cfg.object_randomization.translation_x_max_m, 0.09)
        self.assertEqual(env_cfg.object_randomization.translation_y_min_m, 0.0)
        self.assertEqual(env_cfg.object_randomization.translation_y_max_m, 0.15)
        self.assertEqual(env_cfg.object_randomization.yaw_min_deg, -22.5)
        self.assertEqual(env_cfg.object_randomization.yaw_max_deg, 45.0)
        self.assertEqual(env_cfg.object.mass_kg, 0.2)
        self.assertEqual(env_cfg.object.friction, 0.5)
        self.assertEqual(env_cfg.object.restitution, 0.0)
        # The object-centric reward. Palm and fingertip keypoints in the
        # cuboid's frame carry the tracking; the joint-space terms survive at a
        # small weight purely to pick one solution out of the arm's null space
        # and the five spare finger degrees of freedom.
        self.assertEqual(env_cfg.rewards.palm_keypoint_weight, 0.80)
        self.assertEqual(env_cfg.rewards.fingertip_keypoint_weight, 0.48)
        self.assertEqual(env_cfg.rewards.palm_keypoint_std_m, 0.05)
        self.assertEqual(env_cfg.rewards.fingertip_keypoint_std_m, 0.025)
        # Must match the value the transform bank was built with, or the
        # reference keypoints describe a differently shaped hand.
        self.assertEqual(env_cfg.rewards.palm_lever_arm_m, 0.1)
        expected_robot_rewards = {
            # Pitch and roll of the palm; yaw excluded because the bar's yaw is
            # randomised and the hand has to follow it.
            "palm_tilt_weight": 0.25,
            "palm_tilt_std_rad": 0.35,
            "ee_action_rate_weight": 0.2,
            "ee_action_rate_std": 0.03,
            "arm_joint_rate_weight": 0.05,
            "arm_joint_rate_std_rad": 0.02,
            # Ships off: the feasibility pressure is meant to arrive through the
            # keypoint tracking reward, not through an explicit penalty.
            "ik_residual_weight": 0.0,
            "ik_residual_std": 0.01,
            "position_hand_weight": 0.05,
            "velocity_hand_weight": 0.12,
            "hand_action_rate_weight": 0.12,
            "position_hand_std_rad": 0.223607,
            "velocity_hand_std_rad_per_s": 1.0,
            "hand_action_rate_std": 1.0,
        }
        for name, expected in expected_robot_rewards.items():
            self.assertEqual(getattr(env_cfg.rewards, name), expected)
        # The arm's joint-space tracking terms were removed outright, not
        # zero-weighted; their absence is what keeps them from creeping back.
        for name in (
            "position_arm_weight",
            "velocity_arm_weight",
            "position_arm_std_rad",
            "velocity_arm_std_rad_per_s",
        ):
            self.assertFalse(hasattr(env_cfg.rewards, name))
        self.assertEqual(env_cfg.rewards.object_position_weight, 0.8)
        self.assertEqual(env_cfg.rewards.object_orientation_weight, 0.4)
        # Off: the object-frame fingertip keypoints subsume what this shaped.
        self.assertEqual(
            env_cfg.rewards.fingertip_object_distance_weight, 0.0
        )
        self.assertEqual(
            env_cfg.rewards.fingertip_object_distance_std_m, 0.04
        )
        self.assertEqual(
            env_cfg.rewards.fingertip_object_distance_names,
            ["thumb", "index", "middle"],
        )
        self.assertTrue(env_cfg.contact.enabled)
        # The actor must stay blind to contact; only the critic may see it.
        self.assertFalse(env_cfg.contact.observe_fingertip_forces)
        self.assertTrue(env_cfg.contact.critic_observes_fingertip_forces)
        self.assertEqual(env_cfg.rewards.object_position_std_m, 0.12)
        self.assertEqual(env_cfg.rewards.object_orientation_std_rad, 0.30)
        self.assertTrue(env_cfg.termination.object_position_enabled)
        self.assertEqual(env_cfg.termination.object_position_threshold_m, 0.07)
        self.assertTrue(env_cfg.termination.enabled)
        # Task space, not joint space: the reward deliberately lets the arm
        # leave the retargeted joint angles.
        self.assertEqual(env_cfg.termination.palm_keypoint_threshold_m, 0.20)
        self.assertEqual(env_cfg.termination.hand_position_threshold_rad, 1.35)
        self.assertEqual(env_cfg.termination.grace_steps, 5)
        self.assertEqual(env_cfg.table.surface_below_robot_base_m, 0.035)
        self.assertEqual(train_cfg.algorithm.num_learning_epochs, 5)
        self.assertEqual(train_cfg.algorithm.num_mini_batches, 4)
        self.assertEqual(train_cfg.algorithm.learning_rate, 0.5e-4)
        self.assertEqual(train_cfg.algorithm.entropy_coef, 0.001)
        self.assertEqual(train_cfg.policy.max_action_std, 3.0)
        self.assertEqual(train_cfg.algorithm.schedule, "fixed")

    def test_processed_demonstration_contract(self):
        cfg = SimToolRealCfg()
        demo = JointDemonstration60Hz.load(
            ROOT_DIR / cfg.motion.file, device=torch.device("cpu")
        )
        self.assertEqual(demo.q.shape, (1108, 26))
        self.assertEqual(demo.dq.shape, (1108, 26))
        self.assertEqual(demo.cube_pose.shape, (1108, 7))
        self.assertEqual(demo.cube_linear_velocity.shape, (1108, 3))
        self.assertEqual(demo.cube_angular_velocity.shape, (1108, 3))
        self.assertTrue(torch.isfinite(demo.cube_pose).all())
        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(demo.cube_pose[:, 3:7], dim=1),
                torch.ones(1108),
                rtol=0.0,
                atol=1e-6,
            )
        )
        self.assertLess(abs(demo.frequency_hz - 60.0), 0.05)


if __name__ == "__main__":
    unittest.main()
