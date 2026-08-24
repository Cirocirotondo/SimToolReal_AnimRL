import unittest

import torch

from simtoolreal_animrl.cfg import (
    SimToolRealCfg,
    SimToolRealTrainCfg,
    config_to_dict,
    update_config_from_dict,
)
from simtoolreal_animrl.runners.modules import (
    EmpiricalNormalization,
    Policy,
    Value,
)
from simtoolreal_animrl.runners.storage import RolloutStorage


class RunnerModulesTest(unittest.TestCase):
    def setUp(self):
        self.env_cfg = SimToolRealCfg()
        self.train_cfg = SimToolRealTrainCfg()

    def test_animrl_network_state_dict_contract(self):
        policy = Policy(
            self.env_cfg.env.num_observations,
            self.env_cfg.env.num_actions,
            self.train_cfg.policy.actor_hidden_dims,
            self.train_cfg.policy.activation,
            self.train_cfg.policy.log_std_init,
        )
        value = Value(
            self.env_cfg.env.num_observations,
            self.train_cfg.policy.critic_hidden_dims,
            self.train_cfg.policy.activation,
        )
        self.assertEqual(
            set(policy.state_dict()),
            {
                "log_std",
                "policy_latent_net.0.weight",
                "policy_latent_net.0.bias",
                "policy_latent_net.2.weight",
                "policy_latent_net.2.bias",
                "action_mean_net.weight",
                "action_mean_net.bias",
            },
        )
        self.assertEqual(
            set(value.state_dict()),
            {
                "value.0.weight",
                "value.0.bias",
                "value.2.weight",
                "value.2.bias",
                "value.4.weight",
                "value.4.bias",
            },
        )

        observations = torch.zeros(8, self.env_cfg.env.num_observations)
        actions, log_prob = policy.act_and_log_prob(observations)
        self.assertEqual(actions.shape, (8, self.env_cfg.env.num_actions))
        self.assertEqual(log_prob.shape, (8,))
        self.assertEqual(value(observations).shape, (8, 1))

    def test_normalizer_and_rollout_shapes(self):
        normalizer = EmpiricalNormalization(19)
        normalized = normalizer(torch.randn(16, 19))
        self.assertEqual(normalized.shape, (16, 19))
        self.assertEqual(
            set(normalizer.state_dict()), {"_mean", "_var", "_std"}
        )

        storage = RolloutStorage(8, 24, 19, 19, 6)
        self.assertEqual(storage.observations.shape, (24, 8, 19))
        self.assertEqual(storage.actions.shape, (24, 8, 6))
        self.assertEqual(storage.values.shape, (24, 8, 1))

    def test_training_config_matches_animrl_cartwheel(self):
        train = self.train_cfg
        self.assertEqual(train.runner.num_steps_per_env, 24)
        self.assertEqual(train.algorithm.num_learning_epochs, 5)
        self.assertEqual(train.algorithm.num_mini_batches, 4)
        self.assertEqual(train.algorithm.learning_rate, 1.0e-4)
        self.assertEqual(train.algorithm.schedule, "fixed")
        self.assertEqual(train.policy.actor_hidden_dims, [512, 256])
        self.assertEqual(train.policy.critic_hidden_dims, [512, 256])

        snapshot = config_to_dict(train)
        self.assertEqual(snapshot["algorithm"]["learning_rate"], 1.0e-4)
        self.assertEqual(snapshot["runner"]["num_steps_per_env"], 24)

        restored = SimToolRealTrainCfg()
        restored.algorithm.learning_rate = 9.9
        update_config_from_dict(restored, snapshot)
        self.assertEqual(restored.algorithm.learning_rate, 1.0e-4)
        with self.assertRaises(KeyError):
            update_config_from_dict(restored, {"unknown": 1})


if __name__ == "__main__":
    unittest.main()
