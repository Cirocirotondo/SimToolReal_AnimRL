"""The training-time disturbance draws, without a simulator."""

import math
import unittest

import torch

from simtoolreal_animrl.cfg.simtoolreal_config import SimToolRealCfg
from simtoolreal_animrl.envs.domain_randomization import (
    log_uniform_scales,
    random_vectors_in_ball,
    resolve_settings,
    velocity_noise_for_position_noise,
)


class DomainRandomizationTest(unittest.TestCase):
    def test_it_is_enabled_by_default_to_match_grasp_asym_scratch(self):
        self.assertIsNotNone(resolve_settings(SimToolRealCfg().domain_randomization))

    def test_the_defaults_are_the_probe_realistic_column(self):
        cfg = SimToolRealCfg().domain_randomization
        cfg.enabled = True
        settings = resolve_settings(cfg)
        self.assertAlmostEqual(settings["obs_q_noise_rad"], 0.005)
        self.assertAlmostEqual(settings["obs_q_bias_rad"], 0.005)
        self.assertAlmostEqual(settings["init_q_offset_rad"], 0.030)
        self.assertAlmostEqual(settings["init_dq_offset_rad_s"], 0.5)
        self.assertEqual(settings["action_delay_max_steps"], 1)
        self.assertEqual(settings["object_mass_scale_range"], (0.8, 1.2))
        self.assertEqual(settings["object_inertia_scale_range"], (0.8, 1.2))
        self.assertEqual(settings["object_friction_range"], (0.35, 0.65))
        self.assertEqual(settings["object_restitution_range"], (0.0, 0.10))
        self.assertEqual(settings["robot_friction_scale_range"], (0.8, 1.2))
        self.assertEqual(settings["table_friction_range"], (0.4, 0.6))
        self.assertEqual(settings["gravity_z_scale_range"], (0.97, 1.03))
        self.assertAlmostEqual(settings["gravity_xy_max_m_s2"], 0.15)
        self.assertAlmostEqual(
            settings["external_wrench_probability_per_step"], 0.002
        )
        self.assertEqual(settings["external_wrench_duration_steps"], 6)
        self.assertAlmostEqual(settings["external_force_max_n"], 1.0)
        self.assertAlmostEqual(settings["external_torque_max_nm"], 0.02)
        # Well under the 0.35 rad arm termination threshold, or the condition
        # degenerates into self-inflicted failure.
        self.assertLess(settings["init_q_offset_rad"], 0.35 / 4.0)

    def test_velocity_noise_follows_from_the_encoder_and_the_rate(self):
        """sigma_dq is the cost of differentiating q once per control step."""
        self.assertAlmostEqual(
            velocity_noise_for_position_noise(0.005, 60.0), 0.42426, places=5
        )
        cfg = SimToolRealCfg().domain_randomization
        cfg.enabled = True
        cfg.obs_q_noise_rad = 0.020
        settings = resolve_settings(cfg)
        self.assertAlmostEqual(
            settings["obs_dq_noise_rad_s"],
            math.sqrt(2.0) * 60.0 * 0.020,
            places=6,
        )

    def test_the_coupling_can_be_turned_off(self):
        cfg = SimToolRealCfg().domain_randomization
        cfg.enabled = True
        cfg.couple_velocity_noise_to_position_noise = False
        cfg.obs_dq_noise_rad_s = 1.5
        self.assertAlmostEqual(resolve_settings(cfg)["obs_dq_noise_rad_s"], 1.5)

    def test_a_negative_magnitude_is_rejected_rather_than_silently_trained(self):
        cfg = SimToolRealCfg().domain_randomization
        cfg.enabled = True
        cfg.obs_q_noise_rad = -0.001
        with self.assertRaises(ValueError):
            resolve_settings(cfg)
        cfg.obs_q_noise_rad = 0.005
        cfg.action_delay_max_steps = -1
        with self.assertRaises(ValueError):
            resolve_settings(cfg)

    def test_zero_halfwidth_is_exactly_one_so_the_null_gate_holds(self):
        scales = log_uniform_scales((64,), 0.0, torch.device("cpu"))
        self.assertTrue(bool((scales == 1.0).all()))

    def test_external_wrenches_are_isotropic_and_bounded(self):
        generator = torch.Generator().manual_seed(12)
        vectors = random_vectors_in_ball(
            10000, 1.5, torch.device("cpu"), generator=generator
        )
        norms = torch.linalg.vector_norm(vectors, dim=1)
        self.assertLessEqual(float(norms.max()), 1.5 + 1e-6)
        self.assertLess(float(vectors.mean(dim=0).abs().max()), 0.03)

    def test_invalid_physics_ranges_are_rejected(self):
        cfg = SimToolRealCfg().domain_randomization
        cfg.object_mass_scale_range = [1.2, 0.8]
        with self.assertRaises(ValueError):
            resolve_settings(cfg)
        cfg.object_mass_scale_range = [0.8, 1.2]
        cfg.object_restitution_range = [0.0, 1.1]
        with self.assertRaises(ValueError):
            resolve_settings(cfg)
        cfg.object_restitution_range = [0.0, 0.1]
        cfg.external_wrench_probability_per_step = 1.1
        with self.assertRaises(ValueError):
            resolve_settings(cfg)

    def test_gains_land_inside_the_configured_band_symmetrically_in_log(self):
        generator = torch.Generator().manual_seed(0)
        scales = log_uniform_scales(
            (200000,), 0.3219, torch.device("cpu"), generator=generator
        )
        self.assertGreaterEqual(float(scales.min()), 2.0 ** -0.3219 - 1e-6)
        self.assertLessEqual(float(scales.max()), 2.0 ** 0.3219 + 1e-6)
        # Symmetric in log space, which is the point of drawing there: halving
        # and doubling a gain have to be equally likely.
        self.assertAlmostEqual(
            float(torch.log2(scales).mean()), 0.0, places=2
        )


if __name__ == "__main__":
    unittest.main()
