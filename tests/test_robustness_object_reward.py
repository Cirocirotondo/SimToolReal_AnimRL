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
    / "run_robustness_object_reward.py"
)
SPEC = importlib.util.spec_from_file_location(
    "robustness_object_reward", SCRIPT_PATH
)
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)


class RobustnessObjectRewardTest(unittest.TestCase):
    def _args(self, dry_run=True):
        return Namespace(
            robustness_seeds=(42, 43, 44),
            object_scales=(
                Decimal("1.0"),
                Decimal("0.8"),
                Decimal("0.6"),
                Decimal("0.4"),
            ),
            object_seed=42,
            iterations=12000,
            python=Path("/test/venv/bin/python"),
            sim_device="cuda:0",
            num_envs=None,
            save_interval=None,
            eval_interval=None,
            eval_num_envs=None,
            no_periodic_eval=False,
            no_final_eval=False,
            record_video=None,
            overrides=[],
            dry_run=dry_run,
        )

    def test_default_plan_has_three_baselines_and_four_object_scales(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = self._args()
            plans = SWEEP.planned_runs(args, root)
            manifest = {"runs": []}
            succeeded = SWEEP.run_experiments(args, root, manifest)

        self.assertTrue(succeeded)
        self.assertEqual(len(plans), 7)
        self.assertEqual(
            [plan["seed"] for plan in plans[:3]], [42, 43, 44]
        )
        self.assertTrue(
            all(plan["object_scale"] is None for plan in plans[:3])
        )
        self.assertEqual(
            [plan["object_scale"] for plan in plans[3:]],
            [Decimal("1.0"), Decimal("0.8"), Decimal("0.6"), Decimal("0.4")],
        )
        self.assertTrue(all(plan["seed"] == 42 for plan in plans[3:]))

        for replica in manifest["runs"]:
            command = replica["command"]
            self.assertEqual(command[command.index("--iterations") + 1], "12000")
            self.assertNotIn("--resume", command)
            self.assertNotIn("--initialize-from", command)

        expected_weights = [
            ("0", "0"),
            ("0", "0"),
            ("0", "0"),
            ("0.8", "0.2"),
            ("0.64", "0.16"),
            ("0.48", "0.12"),
            ("0.32", "0.08"),
        ]
        self.assertEqual(
            [
                (
                    run["object_position_weight"],
                    run["object_orientation_weight"],
                )
                for run in manifest["runs"]
            ],
            expected_weights,
        )

    def test_failed_run_does_not_stop_remaining_experiments(self):
        with TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "output"
            output_root.mkdir()
            manifest = {"runs": []}
            call_count = 0

            def fake_training(command):
                nonlocal call_count
                run_dir = Path(command[command.index("--log-dir") + 1])
                run_dir.mkdir(parents=True)
                call_count += 1
                if call_count == 1:
                    return CompletedProcess(command, 1)
                (run_dir / "model_12000.pt").touch()
                return CompletedProcess(command, 0)

            args = self._args(dry_run=False)
            with patch.object(SWEEP.subprocess, "run", side_effect=fake_training):
                succeeded = SWEEP.run_experiments(
                    args, output_root, manifest
                )

        self.assertFalse(succeeded)
        self.assertEqual(call_count, 7)
        self.assertEqual(manifest["runs"][0]["status"], "failed")
        self.assertTrue(
            all(run["status"] == "completed" for run in manifest["runs"][1:])
        )


if __name__ == "__main__":
    unittest.main()
