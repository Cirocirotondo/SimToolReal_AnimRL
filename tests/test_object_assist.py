import math
import unittest

import torch

from simtoolreal_animrl.cfg import SimToolRealCfg
from simtoolreal_animrl.envs.object_assist import (
    assist_scale_at,
    object_assist_wrench,
    orientation_error_rotation_vector,
    resolve_object_assist_settings,
)


LAST_INDEX = 1107


def _settings(**overrides):
    cfg = SimToolRealCfg()
    cfg.object_assist.enabled = True
    for name, value in overrides.items():
        if not hasattr(cfg.object_assist, name):
            raise AttributeError(name)
        setattr(cfg.object_assist, name, value)
    return resolve_object_assist_settings(cfg.object_assist, LAST_INDEX)


class ObjectAssistConfigTest(unittest.TestCase):
    def test_defaults_are_off_so_existing_experiments_are_unchanged(self):
        cfg = SimToolRealCfg()
        self.assertFalse(cfg.object_assist.enabled)
        self.assertEqual(cfg.object_assist.schedule, "linear")
        self.assertEqual(cfg.object_assist.start_iteration, 0)
        self.assertEqual(cfg.object_assist.end_iteration, 6000)
        self.assertEqual(cfg.object_assist.initial_scale, 1.0)
        self.assertEqual(cfg.object_assist.final_scale, 0.0)
        self.assertTrue(cfg.object_assist.gravity_compensation)
        self.assertTrue(cfg.object_assist.torque_enabled)
        settings = resolve_object_assist_settings(
            cfg.object_assist, LAST_INDEX
        )
        self.assertFalse(settings.enabled)

    def test_invalid_configurations_are_rejected(self):
        with self.assertRaises(ValueError):
            _settings(schedule="cosine")
        with self.assertRaises(ValueError):
            _settings(start_iteration=-1)
        with self.assertRaises(ValueError):
            _settings(start_iteration=100, end_iteration=50)
        # A linear schedule with an empty window has no defined slope.
        with self.assertRaises(ValueError):
            _settings(start_iteration=100, end_iteration=100)
        with self.assertRaises(ValueError):
            _settings(position_stiffness_n_per_m=-1.0)
        with self.assertRaises(ValueError):
            _settings(max_force_n=0.0)
        with self.assertRaises(ValueError):
            _settings(active_from_reference_index=LAST_INDEX + 1)

    def test_constant_schedule_needs_no_window(self):
        settings = _settings(
            schedule="constant", start_iteration=10, end_iteration=10
        )
        self.assertEqual(assist_scale_at(settings, 0), 1.0)
        self.assertEqual(assist_scale_at(settings, 10_000), 1.0)


class ObjectAssistScheduleTest(unittest.TestCase):
    def test_linear_decay_over_the_window(self):
        settings = _settings(start_iteration=1000, end_iteration=3000)
        self.assertEqual(assist_scale_at(settings, 0), 1.0)
        self.assertEqual(assist_scale_at(settings, 1000), 1.0)
        self.assertAlmostEqual(assist_scale_at(settings, 2000), 0.5)
        self.assertAlmostEqual(assist_scale_at(settings, 2500), 0.25)
        self.assertEqual(assist_scale_at(settings, 3000), 0.0)
        # Past the window the run is exactly the unassisted problem.
        self.assertEqual(assist_scale_at(settings, 9000), 0.0)

    def test_a_disabled_assist_is_always_zero(self):
        cfg = SimToolRealCfg()
        settings = resolve_object_assist_settings(
            cfg.object_assist, LAST_INDEX
        )
        self.assertEqual(assist_scale_at(settings, 0), 0.0)
        self.assertEqual(assist_scale_at(settings, 12_000), 0.0)

    def test_final_scale_can_be_a_non_zero_floor(self):
        settings = _settings(
            start_iteration=0, end_iteration=100, final_scale=0.2
        )
        self.assertAlmostEqual(assist_scale_at(settings, 50), 0.6)
        self.assertAlmostEqual(assist_scale_at(settings, 500), 0.2)


class OrientationErrorTest(unittest.TestCase):
    def test_quarter_turn_about_z(self):
        identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        half = math.pi / 4.0
        target = torch.tensor([[0.0, 0.0, math.sin(half), math.cos(half)]])
        rotation_vector = orientation_error_rotation_vector(identity, target)
        self.assertAlmostEqual(float(rotation_vector[0, 2]), math.pi / 2.0, places=5)
        self.assertAlmostEqual(float(rotation_vector[0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(rotation_vector[0, 1]), 0.0, places=6)
        # The opposite error points the other way with the same magnitude.
        reversed_vector = orientation_error_rotation_vector(target, identity)
        self.assertAlmostEqual(
            float(reversed_vector[0, 2]), -math.pi / 2.0, places=5
        )

    def test_identical_orientations_need_no_torque(self):
        orientation = torch.nn.functional.normalize(
            torch.tensor([[0.3, -0.2, 0.5, 0.78]]), dim=1
        )
        rotation_vector = orientation_error_rotation_vector(
            orientation, orientation.clone()
        )
        self.assertLess(float(rotation_vector.abs().max()), 1.0e-6)

    def test_the_shortest_path_is_taken_for_a_sign_flipped_quaternion(self):
        orientation = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        half = math.pi * 0.45
        target = torch.tensor([[0.0, 0.0, math.sin(half), math.cos(half)]])
        angle = float(
            torch.linalg.vector_norm(
                orientation_error_rotation_vector(orientation, target), dim=1
            )
        )
        flipped_angle = float(
            torch.linalg.vector_norm(
                orientation_error_rotation_vector(orientation, -target), dim=1
            )
        )
        self.assertAlmostEqual(angle, flipped_angle, places=5)
        self.assertLessEqual(flipped_angle, math.pi + 1.0e-5)


class ObjectAssistWrenchTest(unittest.TestCase):
    def setUp(self):
        self.mass_kg = 0.2
        self.gravity = torch.tensor([0.0, 0.0, -9.81])
        self.position = torch.zeros(2, 3)
        self.orientation = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2)
        self.linear_velocity = torch.zeros(2, 3)
        self.angular_velocity = torch.zeros(2, 3)
        self.reference = torch.zeros(2, 13)
        self.reference[:, 6] = 1.0
        self.active = torch.ones(2, dtype=torch.bool)

    def _wrench(self, settings, scale=1.0):
        return object_assist_wrench(
            self.position,
            self.orientation,
            self.linear_velocity,
            self.angular_velocity,
            self.reference,
            settings,
            scale,
            self.mass_kg,
            self.gravity,
            self.active,
        )

    def test_a_cube_on_target_only_gets_gravity_compensation(self):
        settings = _settings()
        force, torque = self._wrench(settings)
        self.assertAlmostEqual(float(force[0, 2]), self.mass_kg * 9.81, places=5)
        self.assertAlmostEqual(float(force[0, 0]), 0.0, places=6)
        self.assertLess(float(torque.abs().max()), 1.0e-6)

    def test_position_and_velocity_errors_enter_with_the_configured_gains(self):
        settings = _settings(
            gravity_compensation=False,
            position_stiffness_n_per_m=100.0,
            position_damping_ns_per_m=10.0,
        )
        self.reference[:, 0] = 0.05
        self.reference[:, 7] = 0.2
        self.linear_velocity[:, 0] = 0.1
        force, _ = self._wrench(settings)
        # 100 * 0.05 + 10 * (0.2 - 0.1)
        self.assertAlmostEqual(float(force[0, 0]), 6.0, places=5)

    def test_the_scale_multiplies_the_whole_wrench(self):
        settings = _settings()
        self.reference[:, 0] = 0.02
        full_force, full_torque = self._wrench(settings, scale=1.0)
        half_force, half_torque = self._wrench(settings, scale=0.5)
        self.assertTrue(torch.allclose(half_force, 0.5 * full_force))
        self.assertTrue(torch.allclose(half_torque, 0.5 * full_torque))
        zero_force, zero_torque = self._wrench(settings, scale=0.0)
        self.assertEqual(float(zero_force.abs().max()), 0.0)
        self.assertEqual(float(zero_torque.abs().max()), 0.0)

    def test_saturation_preserves_the_direction(self):
        settings = _settings(
            gravity_compensation=False,
            position_stiffness_n_per_m=1000.0,
            max_force_n=5.0,
        )
        self.reference[:, 0] = 0.3
        self.reference[:, 1] = 0.4
        force, _ = self._wrench(settings)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(force[0])), 5.0, places=5)
        self.assertAlmostEqual(
            float(force[0, 1] / force[0, 0]), 0.4 / 0.3, places=5
        )

    def test_inactive_environments_receive_nothing(self):
        settings = _settings()
        self.reference[:, 0] = 0.1
        self.active[1] = False
        force, torque = self._wrench(settings)
        self.assertGreater(float(force[0].abs().max()), 0.0)
        self.assertEqual(float(force[1].abs().max()), 0.0)
        self.assertEqual(float(torque[1].abs().max()), 0.0)

    def test_torque_can_be_disabled(self):
        settings = _settings(torque_enabled=False)
        half = math.pi / 4.0
        self.reference[:, 3:7] = torch.tensor(
            [0.0, 0.0, math.sin(half), math.cos(half)]
        )
        _, torque = self._wrench(settings)
        self.assertEqual(float(torque.abs().max()), 0.0)

    def test_torque_opposes_the_orientation_error(self):
        settings = _settings(
            orientation_stiffness_nm_per_rad=0.1,
            orientation_damping_nms_per_rad=0.0,
            max_torque_nm=10.0,
        )
        half = math.pi / 4.0
        self.reference[:, 3:7] = torch.tensor(
            [0.0, 0.0, math.sin(half), math.cos(half)]
        )
        _, torque = self._wrench(settings)
        self.assertAlmostEqual(
            float(torque[0, 2]), 0.1 * math.pi / 2.0, places=5
        )

    def test_a_malformed_reference_state_is_rejected(self):
        settings = _settings()
        with self.assertRaises(ValueError):
            object_assist_wrench(
                self.position,
                self.orientation,
                self.linear_velocity,
                self.angular_velocity,
                self.reference[:, :7],
                settings,
                1.0,
                self.mass_kg,
                self.gravity,
                self.active,
            )


if __name__ == "__main__":
    unittest.main()
