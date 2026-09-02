import importlib.util
import unittest
from argparse import Namespace
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_pregrasp_proximity.py"
)
SPEC = importlib.util.spec_from_file_location("pregrasp_proximity", SCRIPT_PATH)
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


class PregraspProximityLauncherTest(unittest.TestCase):
    def _args(self):
        return Namespace(
            iterations=12000,
            seed=43,
            sim_device="cuda:0",
            num_envs=None,
            save_interval=None,
            eval_interval=None,
            eval_num_envs=None,
            python=Path("/test/venv/bin/python"),
            object_position_weight=0.8,
            object_orientation_weight=0.2,
            proximity_weight=0.2,
            no_periodic_eval=False,
            no_final_eval=False,
            record_video=None,
            overrides=[],
            dry_run=True,
        )

    def test_command_is_scratch_and_uses_only_pregrasp_rsi(self):
        command = LAUNCHER.build_command(self._args(), Path("/tmp/fresh-run"))
        self.assertNotIn("--resume", command)
        self.assertNotIn("--initialize-from", command)
        self.assertEqual(command[command.index("--seed") + 1], "43")
        self.assertIn("env.episode_length=200", command)
        self.assertIn("env.reference_init_distribution=pregrasp_mixture", command)
        self.assertIn("env.rsi_early_probability=0", command)
        self.assertIn("env.rsi_pregrasp_start_index=740", command)
        self.assertIn("env.rsi_max_start_index=830", command)

    def test_command_enables_parent_object_rewards_but_not_contact(self):
        command = LAUNCHER.build_command(self._args(), Path("/tmp/fresh-run"))
        self.assertIn("rewards.object_position_weight=0.8", command)
        self.assertIn("rewards.object_orientation_weight=0.2", command)
        self.assertIn("rewards.fingertip_object_distance_weight=0.2", command)
        self.assertIn("contact.enabled=false", command)
        self.assertIn("termination.object_position_enabled=true", command)


if __name__ == "__main__":
    unittest.main()
