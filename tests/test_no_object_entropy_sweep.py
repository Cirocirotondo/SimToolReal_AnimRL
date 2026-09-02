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
    / "run_no_object_entropy_sweep.py"
)
SPEC = importlib.util.spec_from_file_location("no_object_entropy_sweep", SCRIPT_PATH)
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)


class NoObjectEntropySweepTest(unittest.TestCase):
    def _args(self, dry_run=True):
        return Namespace(
            entropy_coefficients=(
                Decimal("0.005"),
                Decimal("0.002"),
                Decimal("0.001"),
                Decimal("0.0005"),
                Decimal("0"),
            ),
            seed=42,
            iterations=12000,
            output_root=None,
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

    def test_plan_is_five_independent_seed_42_scratch_runs(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = self._args()
            plans = SWEEP.planned_runs(args, root)
            manifest = {"runs": []}
            succeeded = SWEEP.run_sweep(args, root, manifest)

        self.assertTrue(succeeded)
        self.assertEqual(len(plans), 5)
        self.assertEqual(
            [plan["entropy_coefficient"] for plan in plans],
            list(args.entropy_coefficients),
        )
        for run, coefficient in zip(manifest["runs"], args.entropy_coefficients):
            command = run["command"]
            self.assertEqual(command[command.index("--seed") + 1], "42")
            self.assertEqual(
                command[-1],
                "train.algorithm.entropy_coef={}".format(
                    SWEEP.decimal_text(coefficient)
                ),
            )
            self.assertNotIn("--resume", command)
            self.assertNotIn("--initialize-from", command)
            self.assertEqual(run["initialization"], "random")

    def test_every_run_disables_all_object_rewards(self):
        command = SWEEP.build_command(
            self._args(), Path("/tmp/fresh-run"), Decimal("0.002")
        )
        self.assertIn("contact.enabled=false", command)
        self.assertIn("rewards.fingertip_object_distance_weight=0", command)
        self.assertIn("rewards.object_position_weight=0", command)
        self.assertIn("rewards.object_orientation_weight=0", command)

    def test_failed_run_does_not_stop_the_remaining_runs(self):
        with TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "output"
            output_root.mkdir()
            args = self._args(dry_run=False)
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

            with patch.object(SWEEP.subprocess, "run", side_effect=fake_training):
                succeeded = SWEEP.run_sweep(args, output_root, manifest)

        self.assertFalse(succeeded)
        self.assertEqual(call_count, 5)
        self.assertEqual(manifest["runs"][0]["status"], "failed")
        self.assertTrue(
            all(run["status"] == "completed" for run in manifest["runs"][1:])
        )

    def test_values_must_be_strictly_descending_and_below_reference(self):
        args = self._args()
        args.entropy_coefficients = (Decimal("0.01"), Decimal("0.005"))
        with self.assertRaises(ValueError):
            SWEEP.validate_args(args)
        args.entropy_coefficients = (Decimal("0.001"), Decimal("0.002"))
        with self.assertRaises(ValueError):
            SWEEP.validate_args(args)


if __name__ == "__main__":
    unittest.main()
