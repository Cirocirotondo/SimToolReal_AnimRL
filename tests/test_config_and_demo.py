import unittest

import torch

from simtoolreal_animrl import ROOT_DIR
from simtoolreal_animrl.cfg import SimToolRealCfg, SimToolRealTrainCfg
from simtoolreal_animrl.envs.demonstration import JointDemonstration60Hz


class ConfigAndDemoTest(unittest.TestCase):
    def test_animrl_configuration_values(self):
        env_cfg = SimToolRealCfg()
        train_cfg = SimToolRealTrainCfg()
        self.assertEqual(env_cfg.env.num_envs, 4096)
        self.assertEqual(env_cfg.env.episode_length, 100)
        self.assertEqual(env_cfg.env.num_observations, 114)
        self.assertEqual(env_cfg.env.num_actions, 26)
        self.assertEqual(
            env_cfg.control.action_parameterization, "animrl_residual"
        )
        self.assertEqual(env_cfg.control.scale_joint_target, 0.25)
        self.assertEqual(env_cfg.control.scale_hand_joint_target, 0.15)
        self.assertEqual(env_cfg.control.clip_joint_target, 100.0)
        # The fingers are allowed to pass through each other, which is what
        # makes the step roughly twice as fast; see asset.self_collision.
        self.assertFalse(env_cfg.asset.self_collision)
        self.assertEqual(env_cfg.object.size_m, [0.15, 0.05, 0.05])
        self.assertEqual(env_cfg.object.mass_kg, 0.2)
        self.assertEqual(env_cfg.object.friction, 0.5)
        self.assertEqual(env_cfg.object.restitution, 0.0)
        self.assertEqual(env_cfg.rewards.object_position_weight, 0.8)
        self.assertEqual(env_cfg.rewards.object_orientation_weight, 0.2)
        self.assertEqual(env_cfg.rewards.object_position_std_m, 0.05)
        self.assertEqual(env_cfg.rewards.object_orientation_std_rad, 0.5)
        self.assertEqual(env_cfg.table.surface_below_robot_base_m, 0.03)
        self.assertEqual(train_cfg.algorithm.num_learning_epochs, 5)
        self.assertEqual(train_cfg.algorithm.num_mini_batches, 4)
        self.assertEqual(train_cfg.algorithm.learning_rate, 1.0e-4)
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
