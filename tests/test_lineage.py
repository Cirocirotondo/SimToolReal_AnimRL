import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from simtoolreal_animrl.runners.lineage import (
    SIDECAR_NAME,
    ancestry,
    read_sidecar,
    resolve_lineage,
)


class ResolveLineageTest(unittest.TestCase):
    def test_a_scratch_run_records_no_lineage(self):
        self.assertIsNone(resolve_lineage(None))

    def test_a_direct_resume_names_the_checkpoint_s_own_run(self):
        with TemporaryDirectory() as workspace:
            parent = Path(workspace) / "2026-09-06_114700_parent_run"
            parent.mkdir()
            checkpoint = parent / "best_model.pt"
            checkpoint.write_bytes(b"x")
            lineage = resolve_lineage(checkpoint)
        self.assertEqual(lineage["parent_run"], "2026-09-06_114700_parent_run")
        self.assertEqual(lineage["checkpoint_name"], "best_model.pt")
        self.assertFalse(lineage["copied"])

    def test_a_copied_checkpoint_reports_the_true_parent_not_its_own_directory(self):
        """The failure this exists to prevent: a copy hides where it came from."""
        with TemporaryDirectory() as workspace:
            child = Path(workspace) / "2026-09-07_102937_smooth_warm"
            child.mkdir()
            checkpoint = child / "best_model.pt"
            checkpoint.write_bytes(b"x")
            (child / SIDECAR_NAME).write_text(
                json.dumps(
                    {
                        "parent_run": "2026-09-06_114700_pg830_obs_n512",
                        "parent_checkpoint": "/logs/pg830_obs_n512/best_model.pt",
                        "parent_iteration": 11500,
                    }
                ),
                encoding="utf-8",
            )
            lineage = resolve_lineage(checkpoint)
        self.assertEqual(lineage["parent_run"], "2026-09-06_114700_pg830_obs_n512")
        self.assertTrue(lineage["copied"])
        self.assertEqual(lineage["parent_iteration"], 11500)

    def test_checkpoint_metadata_is_folded_in_when_available(self):
        with TemporaryDirectory() as workspace:
            parent = Path(workspace) / "run"
            parent.mkdir()
            checkpoint = parent / "best_model.pt"
            checkpoint.write_bytes(b"x")
            lineage = resolve_lineage(
                checkpoint,
                {"iteration": 11500, "best_evaluation_score": 0.9657},
            )
        self.assertEqual(lineage["checkpoint_iteration"], 11500)
        self.assertAlmostEqual(lineage["checkpoint_best_evaluation_score"], 0.9657)

    def test_an_unreadable_sidecar_is_ignored_rather_than_fatal(self):
        with TemporaryDirectory() as workspace:
            child = Path(workspace) / "child"
            child.mkdir()
            checkpoint = child / "best_model.pt"
            checkpoint.write_bytes(b"x")
            (child / SIDECAR_NAME).write_text("{not json", encoding="utf-8")
            self.assertIsNone(read_sidecar(checkpoint))
            self.assertEqual(resolve_lineage(checkpoint)["parent_run"], "child")


class AncestryTest(unittest.TestCase):
    def _run(self, root, name, parent=None):
        directory = root / name
        directory.mkdir()
        config = {"lineage": {"parent_run": parent} if parent else None}
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return directory

    def test_the_chain_is_walked_back_to_the_scratch_run(self):
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            self._run(root, "a_scratch")
            self._run(root, "b_warm", parent="a_scratch")
            third = self._run(root, "c_warm2", parent="b_warm")
            self.assertEqual(ancestry(third), ["c_warm2", "b_warm", "a_scratch"])

    def test_a_missing_ancestor_ends_the_chain_without_raising(self):
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            child = self._run(root, "child", parent="vanished_run")
            self.assertEqual(ancestry(child), ["child", "vanished_run"])

    def test_a_cycle_cannot_loop_forever(self):
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            self._run(root, "one", parent="two")
            self._run(root, "two", parent="one")
            self.assertEqual(ancestry(root / "one"), ["one", "two"])


if __name__ == "__main__":
    unittest.main()
