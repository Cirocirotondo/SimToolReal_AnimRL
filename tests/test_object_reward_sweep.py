import importlib.util
import unittest
from argparse import Namespace
from decimal import Decimal
from pathlib import Path
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory
from unittest.mock import patch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_object_reward_sweep.py"
)
SPEC = importlib.util.spec_from_file_location("object_reward_sweep", SCRIPT_PATH)
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)


class ObjectRewardSweepTest(unittest.TestCase):
    def _args(self, checkpoint):
        return Namespace(
            checkpoint=checkpoint,
            warm_start_mode="full",
            python=Path("/test/venv/bin/python"),
            iterations_per_run=6000,
            sim_device="cuda:0",
            num_envs=None,
            seed=None,
            save_interval=None,
            no_periodic_eval=False,
            record_video=None,
            dry_run=True,
        )

    def test_warm_runs_all_use_the_same_source_checkpoint(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "model_7500.pt"
            checkpoint.touch()
            manifest = {"experiments": []}
            scales = (Decimal("0.1"), Decimal("0.2"), Decimal("0.3"))
            weights = {
                "position": Decimal("0.8"),
                "orientation": Decimal("0.2"),
                "proximity": Decimal("0.2"),
            }

            succeeded = SWEEP.run_initialization(
                self._args(checkpoint),
                root / "output",
                "warm",
                scales,
                weights,
                manifest,
            )

            self.assertTrue(succeeded)
            self.assertEqual(len(manifest["experiments"]), len(scales))
            expected_checkpoint = str(checkpoint.resolve())
            for experiment in manifest["experiments"]:
                self.assertEqual(
                    experiment["input_checkpoint"], expected_checkpoint
                )
                self.assertIn("--resume", experiment["command"])
                self.assertNotIn("--initialize-from", experiment["command"])

    def test_scratch_runs_never_receive_a_checkpoint(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = {"experiments": []}
            args = self._args(root / "unused.pt")

            succeeded = SWEEP.run_initialization(
                args,
                root / "output",
                "scratch",
                (Decimal("0.1"), Decimal("1.0")),
                {
                    "position": Decimal("0.8"),
                    "orientation": Decimal("0.2"),
                    "proximity": Decimal("0.2"),
                },
                manifest,
            )

            self.assertTrue(succeeded)
            for experiment in manifest["experiments"]:
                self.assertIsNone(experiment["input_checkpoint"])
                self.assertEqual(experiment["checkpoint_loading"], "random")
                self.assertNotIn("--resume", experiment["command"])
                self.assertNotIn("--initialize-from", experiment["command"])

    def test_diverged_run_does_not_stop_remaining_experiments(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_root = root / "output"
            output_root.mkdir()
            checkpoint = root / "model_7500.pt"
            checkpoint.touch()
            manifest = {"experiments": []}
            call_count = 0

            def fake_training(command):
                nonlocal call_count
                run_dir = Path(command[command.index("--log-dir") + 1])
                run_dir.mkdir(parents=True)
                call_count += 1
                if call_count == 1:
                    (run_dir / "diverged_model.pt").touch()
                else:
                    (run_dir / "model_13500.pt").touch()
                return CompletedProcess(command, 0)

            args = self._args(checkpoint)
            args.dry_run = False
            with patch.object(SWEEP.subprocess, "run", side_effect=fake_training):
                succeeded = SWEEP.run_initialization(
                    args,
                    output_root,
                    "warm",
                    (Decimal("0.1"), Decimal("0.2")),
                    {
                        "position": Decimal("0.8"),
                        "orientation": Decimal("0.2"),
                        "proximity": Decimal("0.2"),
                    },
                    manifest,
                )

            self.assertFalse(succeeded)
            self.assertEqual(call_count, 2)
            self.assertEqual(
                [item["status"] for item in manifest["experiments"]],
                ["diverged", "completed"],
            )
            self.assertEqual(
                manifest["experiments"][0]["input_checkpoint"],
                manifest["experiments"][1]["input_checkpoint"],
            )


if __name__ == "__main__":
    unittest.main()
