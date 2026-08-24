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
        self.assertEqual(env_cfg.env.episode_length, 500)
        self.assertEqual(env_cfg.env.num_observations, 19)
        self.assertEqual(env_cfg.env.num_actions, 6)
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
        self.assertLess(abs(demo.frequency_hz - 60.0), 0.05)


if __name__ == "__main__":
    unittest.main()
