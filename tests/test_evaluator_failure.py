"""A failed evaluation must cost its score, not the training run."""

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from simtoolreal_animrl.runners.evaluation import SubprocessDeterministicEvaluator


class _FakeRunner:
    """Stands in for PPO: the evaluator only ever asks it to save."""

    def __init__(self):
        self.saved = []

    def save(self, path, infos=None):
        self.saved.append(Path(path))
        Path(path).write_bytes(b"checkpoint")


class EvaluatorFailureTest(unittest.TestCase):
    def _evaluator(self, workspace):
        config = Path(workspace) / "config.json"
        config.write_text("{}", encoding="utf-8")
        return SubprocessDeterministicEvaluator(
            interval=500,
            num_envs=64,
            seed=123,
            fixed_phases=[0.0],
            sim_device="cuda:0",
            config_path=config,
            run_dir=workspace,
        )

    def test_a_crashed_evaluator_returns_none_instead_of_raising(self):
        """PPO treats None as 'no evaluation this iteration' and keeps going."""
        with TemporaryDirectory() as workspace:
            evaluator = self._evaluator(workspace)
            failure = subprocess.CalledProcessError(1, ["periodic_evaluate.py"])
            with mock.patch("subprocess.run", side_effect=failure):
                result = evaluator(500, _FakeRunner())
        self.assertIsNone(result)
        self.assertEqual(evaluator.consecutive_failures, 1)

    def test_a_truncated_metrics_file_is_survived(self):
        """An evaluator killed mid-write leaves unreadable JSON behind."""
        with TemporaryDirectory() as workspace:
            evaluator = self._evaluator(workspace)

            def write_garbage(command, check=False):
                output = Path(workspace) / ".periodic_evaluation_metrics.json"
                output.write_text("{not json", encoding="utf-8")
                return mock.Mock(returncode=0)

            with mock.patch("subprocess.run", side_effect=write_garbage):
                result = evaluator(500, _FakeRunner())
        self.assertIsNone(result)
        self.assertEqual(evaluator.consecutive_failures, 1)

    def test_a_missing_metrics_file_is_survived(self):
        with TemporaryDirectory() as workspace:
            evaluator = self._evaluator(workspace)
            with mock.patch("subprocess.run", return_value=mock.Mock(returncode=0)):
                result = evaluator(500, _FakeRunner())
        self.assertIsNone(result)
        self.assertEqual(evaluator.consecutive_failures, 1)

    def test_failures_are_counted_consecutively_and_reset_on_success(self):
        with TemporaryDirectory() as workspace:
            evaluator = self._evaluator(workspace)
            failure = subprocess.CalledProcessError(1, ["periodic_evaluate.py"])
            with mock.patch("subprocess.run", side_effect=failure):
                evaluator(500, _FakeRunner())
                evaluator(1000, _FakeRunner())
            self.assertEqual(evaluator.consecutive_failures, 2)

            def write_metrics(command, check=False):
                output = Path(workspace) / ".periodic_evaluation_metrics.json"
                output.write_text(json.dumps({"evaluation_score": 0.5}), "utf-8")
                return mock.Mock(returncode=0)

            with mock.patch("subprocess.run", side_effect=write_metrics):
                result = evaluator(1500, _FakeRunner())
        self.assertEqual(result, {"evaluation_score": 0.5})
        self.assertEqual(evaluator.consecutive_failures, 0)

    def test_success_saves_a_full_phase_zero_plot_set_by_iteration(self):
        with TemporaryDirectory() as workspace:
            evaluator = self._evaluator(workspace)
            commands = []

            def complete(command, check=False):
                del check
                commands.append(command)
                if Path(command[1]).name == "periodic_evaluate.py":
                    output = Path(workspace) / ".periodic_evaluation_metrics.json"
                    output.write_text(
                        json.dumps({"evaluation_score": 0.5}), encoding="utf-8"
                    )
                return mock.Mock(returncode=0)

            with mock.patch("subprocess.run", side_effect=complete):
                result = evaluator(1500, _FakeRunner())

        self.assertEqual(result, {"evaluation_score": 0.5})
        self.assertEqual(len(commands), 2)
        plot_command = commands[1]
        self.assertEqual(Path(plot_command[1]).name, "evaluate.py")

        def argument(name):
            return plot_command[plot_command.index(name) + 1]

        self.assertEqual(argument("--rsi-index"), "0")
        self.assertEqual(argument("--num-envs"), "1")
        self.assertEqual(
            Path(argument("--plot-dir")),
            Path(workspace) / "eval_plots" / "iteration_001500",
        )
        self.assertIn("--no-show-plots", plot_command)

    def test_a_plot_failure_keeps_valid_evaluation_metrics(self):
        with TemporaryDirectory() as workspace:
            evaluator = self._evaluator(workspace)

            def score_then_fail(command, check=False):
                del check
                if Path(command[1]).name == "periodic_evaluate.py":
                    output = Path(workspace) / ".periodic_evaluation_metrics.json"
                    output.write_text(
                        json.dumps({"evaluation_score": 0.5}), encoding="utf-8"
                    )
                    return mock.Mock(returncode=0)
                raise subprocess.CalledProcessError(1, command)

            with mock.patch("subprocess.run", side_effect=score_then_fail):
                result = evaluator(500, _FakeRunner())

        self.assertEqual(result, {"evaluation_score": 0.5})
        self.assertEqual(evaluator.consecutive_failures, 0)

    def test_the_temporary_checkpoint_is_removed_even_on_failure(self):
        """The finally clause still has to clean up after a crash."""
        with TemporaryDirectory() as workspace:
            evaluator = self._evaluator(workspace)
            failure = subprocess.CalledProcessError(1, ["periodic_evaluate.py"])
            with mock.patch("subprocess.run", side_effect=failure):
                evaluator(500, _FakeRunner())
            leftovers = list(Path(workspace).glob(".periodic_evaluation*"))
        self.assertEqual(leftovers, [])

    def test_off_interval_iterations_still_skip_without_running_anything(self):
        with TemporaryDirectory() as workspace:
            evaluator = self._evaluator(workspace)
            with mock.patch("subprocess.run") as run:
                result = evaluator(499, _FakeRunner())
        self.assertIsNone(result)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
