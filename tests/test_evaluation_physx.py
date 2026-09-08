import unittest

from simtoolreal_animrl.cfg import SimToolRealCfg
from simtoolreal_animrl.runners.evaluation import (
    EVALUATION_MAX_GPU_CONTACT_PAIRS,
    clamp_evaluation_physx,
)


class ClampEvaluationPhysxTest(unittest.TestCase):
    def test_a_training_budget_is_cut_to_the_evaluation_limit(self):
        """The evaluator runs 64 environments beside a live training context."""
        physx = SimToolRealCfg().sim.physx
        physx.max_gpu_contact_pairs = 16 * 1024 * 1024
        settled = clamp_evaluation_physx(physx)
        self.assertEqual(settled, EVALUATION_MAX_GPU_CONTACT_PAIRS)
        self.assertEqual(physx.max_gpu_contact_pairs, EVALUATION_MAX_GPU_CONTACT_PAIRS)

    def test_the_default_budget_is_also_cut(self):
        physx = SimToolRealCfg().sim.physx
        self.assertGreater(
            physx.max_gpu_contact_pairs, EVALUATION_MAX_GPU_CONTACT_PAIRS
        )
        self.assertEqual(
            clamp_evaluation_physx(physx), EVALUATION_MAX_GPU_CONTACT_PAIRS
        )

    def test_a_smaller_budget_is_never_raised(self):
        """Clamping, not assigning: a run that asked for less keeps it."""
        physx = SimToolRealCfg().sim.physx
        physx.max_gpu_contact_pairs = 4096
        self.assertEqual(clamp_evaluation_physx(physx), 4096)

    def test_the_limit_can_be_overridden(self):
        physx = SimToolRealCfg().sim.physx
        physx.max_gpu_contact_pairs = 8 * 1024 * 1024
        self.assertEqual(clamp_evaluation_physx(physx, 2048), 2048)


if __name__ == "__main__":
    unittest.main()
