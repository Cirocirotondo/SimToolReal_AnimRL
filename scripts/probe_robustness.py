#!/usr/bin/env python3
"""Measure how fragile a trained policy is to disturbances, without retraining.

The training environment has no disturbances of any kind: gravity off, fixed PD
gains, exact observations, instantaneous actions, episodes starting exactly on
the reference. Before randomizing any of that, this script measures which
disturbances the current policy already survives and which break it, so the
randomization phase can spend its budget where it matters.

Every perturbation is applied from here by wrapping the constructed environment
or by flipping a config field before construction. No library file is touched.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


def _configure_isaac_gym_graphics_environment():
    """Prevent ROS/Gazebo libraries from shadowing NVIDIA graphics libs."""
    if os.environ.get("ISAACGYM_PRESERVE_ROS_GRAPHICS_PATH") != "1":
        library_path = os.environ.get("LD_LIBRARY_PATH", "")
        kept_paths = [
            entry
            for entry in library_path.split(":")
            if "/opt/ros/" not in entry and "/gazebo" not in entry.lower()
        ]
        if kept_paths:
            os.environ["LD_LIBRARY_PATH"] = ":".join(kept_paths)
        else:
            os.environ.pop("LD_LIBRARY_PATH", None)
    nvidia_icd = "/usr/share/vulkan/icd.d/nvidia_icd.json"
    if os.path.isfile(nvidia_icd):
        os.environ.setdefault("VK_ICD_FILENAMES", nvidia_icd)
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")


_configure_isaac_gym_graphics_environment()

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Preserve Isaac Gym's required import-before-torch ordering.
from simtoolreal_animrl.cfg import (  # noqa: E402
    SimToolRealCfg,
    SimToolRealTrainCfg,
    update_config_from_dict,
)
from simtoolreal_animrl.envs.controller import ARM_JOINT_NAMES, JOINT_NAMES  # noqa: E402
from simtoolreal_animrl.envs.motion_imitation import MotionImitationEnv  # noqa: E402
from simtoolreal_animrl.runners import DeterministicEvaluator, PPO  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


NUM_JOINTS = len(JOINT_NAMES)
NUM_ARM_JOINTS = len(ARM_JOINT_NAMES)
MAX_DELAY_STEPS = 16


# --------------------------------------------------------------------------- #
# Conditions
# --------------------------------------------------------------------------- #


@dataclass
class PerturbSpec:
    """One disturbance setting. All-default is the undisturbed environment."""

    pd_stiffness_scale: float = 1.0
    pd_damping_scale: float = 1.0
    pd_blocks: tuple = ("arm", "hand")
    pd_per_env_log2_halfwidth: float = 0.0
    action_delay_steps: int = 0
    obs_q_noise_rad: float = 0.0
    obs_q_bias_rad: float = 0.0
    obs_dq_noise_rad_s: float = 0.0
    obs_target_noise_rad: float = 0.0
    init_q_offset_rad: float = 0.0
    init_dq_offset_rad_s: float = 0.0

    def is_nominal(self):
        return self == PerturbSpec()

    def touches_pd(self):
        return (
            self.pd_stiffness_scale != 1.0
            or self.pd_damping_scale != 1.0
            or self.pd_per_env_log2_halfwidth != 0.0
        )


@dataclass
class Condition:
    name: str
    family: str
    magnitude: str
    spec: PerturbSpec = field(default_factory=PerturbSpec)
    # "null": must reproduce the baseline exactly. "efficacy": must break the
    # policy, proving the mechanism reaches the simulator at all.
    gate: str = ""


def _pd(name, magnitude, gate="", **kwargs):
    return Condition(name, "pd_gain", magnitude, PerturbSpec(**kwargs), gate)


def runtime_conditions():
    """Conditions that need no rebuild, ordered gates-first then mild-to-severe."""
    return [
        # Null gates: the mechanism runs but adds nothing, so the numbers must
        # match the baseline bit for bit. A failure here means a shim leaks.
        Condition("gate_null_pd", "gate", "x1.00", PerturbSpec(), "null"),
        Condition(
            "gate_null_latency", "gate", "0 steps",
            PerturbSpec(action_delay_steps=0), "null",
        ),
        Condition(
            "gate_null_obs", "gate", "sigma 0",
            PerturbSpec(obs_q_noise_rad=0.0, obs_dq_noise_rad_s=0.0), "null",
        ),
        Condition(
            "gate_null_init", "gate", "sigma 0",
            PerturbSpec(init_q_offset_rad=0.0), "null",
        ),
        # Efficacy gates: deliberately destructive, so a mechanism that silently
        # does nothing is caught instead of being reported as robustness.
        # The PD gate scales stiffness alone. Scaling K and D together leaves
        # D/K untouched, and with gravity off and no contact the steady-state
        # tracking error of a position drive is D*v/K -- so a common factor,
        # however extreme, is very nearly invisible and would make a useless
        # gate.
        _pd("gate_break_pd", "K x0.02", "efficacy", pd_stiffness_scale=0.02),
        Condition(
            "gate_break_latency", "gate", "12 steps",
            PerturbSpec(action_delay_steps=12), "efficacy",
        ),
        Condition(
            "gate_break_obs_dq", "gate", "20 rad/s",
            PerturbSpec(obs_dq_noise_rad_s=20.0), "efficacy",
        ),
        Condition(
            "gate_break_init", "gate", "0.30 rad",
            PerturbSpec(init_q_offset_rad=0.30), "efficacy",
        ),
        # PD gains. Stiffness and damping move the natural frequency and the
        # damping ratio differently, so they are separate knobs; arm and hand
        # gains are independently uncertain against hardware.
        _pd("pd_both_0.70", "K,D x0.70", pd_stiffness_scale=0.70, pd_damping_scale=0.70),
        _pd("pd_both_0.85", "K,D x0.85", pd_stiffness_scale=0.85, pd_damping_scale=0.85),
        _pd("pd_both_1.30", "K,D x1.30", pd_stiffness_scale=1.30, pd_damping_scale=1.30),
        _pd("pd_stiff_0.70", "K x0.70", pd_stiffness_scale=0.70),
        _pd("pd_stiff_1.30", "K x1.30", pd_stiffness_scale=1.30),
        _pd("pd_damp_0.50", "D x0.50", pd_damping_scale=0.50),
        _pd("pd_damp_2.00", "D x2.00", pd_damping_scale=2.00),
        _pd("pd_arm_0.70", "arm x0.70", pd_stiffness_scale=0.70,
            pd_damping_scale=0.70, pd_blocks=("arm",)),
        _pd("pd_hand_0.70", "hand x0.70", pd_stiffness_scale=0.70,
            pd_damping_scale=0.70, pd_blocks=("hand",)),
        # Drawn independently for K and D, because a common per-env factor
        # leaves D/K fixed and is therefore nearly a no-op here.
        _pd("pd_perenv_25", "per-env +-25%", pd_per_env_log2_halfwidth=0.3219),
        _pd("pd_ratio_0.70", "D/K x0.70", pd_stiffness_scale=1.0 / 0.70 ** 0.5,
            pd_damping_scale=0.70 ** 0.5),
        _pd("pd_ratio_1.40", "D/K x1.40", pd_stiffness_scale=1.0 / 1.40 ** 0.5,
            pd_damping_scale=1.40 ** 0.5),
        # Command latency, swept wide because the real stack is uncharacterized.
        Condition("latency_1", "latency", "16.7 ms", PerturbSpec(action_delay_steps=1)),
        Condition("latency_2", "latency", "33.3 ms", PerturbSpec(action_delay_steps=2)),
        Condition("latency_3", "latency", "50.0 ms", PerturbSpec(action_delay_steps=3)),
        # Observation noise. The dq ladder is anchored to the q ladder by
        # sigma_dq = sqrt(2)*60*sigma_q, the cost of differentiating an encoder
        # at 60 Hz, rather than being guessed independently.
        Condition("obs_q_1mrad", "obs_noise", "sigma_q 1 mrad",
                  PerturbSpec(obs_q_noise_rad=0.001)),
        Condition("obs_q_5mrad", "obs_noise", "sigma_q 5 mrad",
                  PerturbSpec(obs_q_noise_rad=0.005)),
        Condition("obs_q_20mrad", "obs_noise", "sigma_q 20 mrad",
                  PerturbSpec(obs_q_noise_rad=0.020)),
        Condition("obs_qbias_5mrad", "obs_noise", "bias 5 mrad",
                  PerturbSpec(obs_q_bias_rad=0.005)),
        Condition("obs_dq_0.085", "obs_noise", "sigma_dq 0.085",
                  PerturbSpec(obs_dq_noise_rad_s=0.085)),
        Condition("obs_dq_0.42", "obs_noise", "sigma_dq 0.42",
                  PerturbSpec(obs_dq_noise_rad_s=0.42)),
        Condition("obs_dq_1.5", "obs_noise", "sigma_dq 1.5",
                  PerturbSpec(obs_dq_noise_rad_s=1.5)),
        Condition("obs_realistic", "obs_noise", "q 5m + bias + dq 0.42",
                  PerturbSpec(obs_q_noise_rad=0.005, obs_q_bias_rad=0.005,
                              obs_dq_noise_rad_s=0.42)),
        # Initial state. Kept well under the 0.35 rad arm termination threshold
        # so the condition does not degenerate into self-inflicted failure.
        Condition("init_q_10mrad", "init_state", "sigma_q 10 mrad",
                  PerturbSpec(init_q_offset_rad=0.010)),
        Condition("init_q_30mrad", "init_state", "sigma_q 30 mrad",
                  PerturbSpec(init_q_offset_rad=0.030)),
        Condition("init_q_80mrad", "init_state", "sigma_q 80 mrad",
                  PerturbSpec(init_q_offset_rad=0.080)),
        Condition("init_dq_0.5", "init_state", "sigma_dq 0.5",
                  PerturbSpec(init_dq_offset_rad_s=0.5)),
        Condition("init_dq_2.0", "init_state", "sigma_dq 2.0",
                  PerturbSpec(init_dq_offset_rad_s=2.0)),
        Condition("init_both", "init_state", "30 mrad + 0.5",
                  PerturbSpec(init_q_offset_rad=0.030, init_dq_offset_rad_s=0.5)),
        # Everything plausible at once. If this exceeds the sum of its parts,
        # the randomization phase has to randomize jointly, not one axis at a
        # time.
        Condition("combo_realistic", "combo", "pd+lat+obs+init",
                  PerturbSpec(pd_stiffness_scale=0.85, pd_damping_scale=0.85,
                              action_delay_steps=1, obs_q_noise_rad=0.005,
                              obs_q_bias_rad=0.005, obs_dq_noise_rad_s=0.42,
                              init_q_offset_rad=0.030, init_dq_offset_rad_s=0.5)),
    ]


# Variants differ in how the environment is BUILT, so each needs its own
# process: disable_gravity is consumed by _load_robot_asset and self_collision
# by _create_envs, both before prepare_sim.
VARIANTS = {
    "nominal": {},
    # An identical rebuild in a separate process. Its deviation from `nominal`
    # is the resolution of the whole experiment, and the only yardstick for the
    # two variants below, which are also compared across a process boundary.
    "nominal_control": {},
    "gravity": {"disable_gravity": False},
    "self_collision": {"self_collision": True},
}


# --------------------------------------------------------------------------- #
# Perturbation harness
# --------------------------------------------------------------------------- #


class PerturbationHarness:
    """Applies disturbances to a live environment by shimming its methods.

    The shims are installed once and stay installed; under the nominal spec they
    are strict pass-throughs. That is what makes the null gates meaningful: they
    exercise the same code path as a real perturbation and must still reproduce
    the undisturbed numbers exactly.
    """

    def __init__(self, env, seed):
        self.env = env
        self.seed = int(seed)
        self.spec = PerturbSpec()
        self._cursor = 0

        # Allocated here, outside any inference_mode block, because tensors
        # created inside one cannot be mutated later.
        self._delay_buffer = torch.zeros(
            (MAX_DELAY_STEPS, env.num_envs, env.num_actions),
            dtype=torch.float32,
            device=env.device,
        )
        self._q_bias = torch.zeros(
            (env.num_envs, NUM_JOINTS), dtype=torch.float32, device=env.device
        )
        self._generator = torch.Generator(device=env.device)
        # Never mutate env.pd_properties: _create_envs handed that same array to
        # every actor and pd_gain_summary() still reads it.
        self._nominal_properties = np.array(env.pd_properties, copy=True)
        self._scratch_properties = np.array(env.pd_properties, copy=True)

        self._orig_step = env.step
        self._orig_reset_idx = env.reset_idx
        self._orig_get_observations = env.get_observations
        # Bound onto the instance, not wrapped around it: step() calls
        # self.reset_idx() internally for the auto-reset, and a wrapper object
        # would never see that call.
        env.step = self._step
        env.reset_idx = self._reset_idx
        env.get_observations = self._get_observations

    # -- lifecycle ---------------------------------------------------------- #

    def activate(self, spec):
        self.spec = spec
        self._cursor = 0
        self._delay_buffer.zero_()
        self._q_bias.zero_()
        self._generator.manual_seed(self.seed)
        # Applied unconditionally, so restoring the pristine gains is structural
        # rather than conditional on the previous condition.
        self._apply_pd_properties(spec)

    def deactivate(self):
        self.activate(PerturbSpec())

    def uninstall(self):
        self.env.step = self._orig_step
        self.env.reset_idx = self._orig_reset_idx
        self.env.get_observations = self._orig_get_observations

    # -- shims -------------------------------------------------------------- #

    def _step(self, actions):
        delay = int(self.spec.action_delay_steps)
        if delay > 0:
            slot = self._cursor % delay
            delayed = self._delay_buffer[slot].clone()
            self._delay_buffer[slot].copy_(actions)
            self._cursor += 1
            actions = delayed
        obs, critic_obs, rewards, dones, infos = self._orig_step(actions)
        return self._perturb_observations(obs), critic_obs, rewards, dones, infos

    def _get_observations(self):
        # Required as well as _step: the evaluator takes the first policy input
        # of every suite from _reset_to_indices -> env.get_observations(), which
        # never passes through step().
        return self._perturb_observations(self._orig_get_observations())

    def _reset_idx(self, env_ids, reference_indices=None):
        self._orig_reset_idx(env_ids, reference_indices)
        if env_ids.numel() == 0:
            return
        spec = self.spec
        if spec.obs_q_bias_rad:
            self._q_bias[env_ids] = (
                self._randn((env_ids.numel(), NUM_JOINTS)) * spec.obs_q_bias_rad
            )
        if spec.init_q_offset_rad or spec.init_dq_offset_rad_s:
            self._offset_initial_state(env_ids, spec)
        if spec.action_delay_steps > 0:
            # reset_idx has just set actions to positions_to_actions(sample.q),
            # so a delayed episode re-commands its own start pose instead of
            # replaying a zero action for its first few steps.
            self._delay_buffer[:, env_ids] = self.env.actions[env_ids]

    # -- mechanisms --------------------------------------------------------- #

    def _randn(self, shape):
        return torch.empty(
            shape, dtype=torch.float32, device=self.env.device
        ).normal_(0.0, 1.0, generator=self._generator)

    def _apply_pd_properties(self, spec):
        env = self.env
        properties = self._scratch_properties
        blocks = []
        if "arm" in spec.pd_blocks:
            blocks.append(env.demo_to_asset[:NUM_ARM_JOINTS])
        if "hand" in spec.pd_blocks:
            blocks.append(env.demo_to_asset[NUM_ARM_JOINTS:])
        indices = (
            np.concatenate(blocks).astype(np.int64)
            if blocks
            else np.empty(0, dtype=np.int64)
        )
        rng = np.random.RandomState(self.seed)
        halfwidth = float(spec.pd_per_env_log2_halfwidth)
        for env_index in range(env.num_envs):
            # Restores driveMode and the joint limits along with the gains.
            properties[:] = self._nominal_properties
            if halfwidth:
                # Independent draws: a shared factor would preserve D/K and so
                # leave tracking untouched in this gravity-free setup.
                stiffness_spread = 2.0 ** rng.uniform(-halfwidth, halfwidth)
                damping_spread = 2.0 ** rng.uniform(-halfwidth, halfwidth)
            else:
                stiffness_spread = damping_spread = 1.0
            properties["stiffness"][indices] *= spec.pd_stiffness_scale * stiffness_spread
            properties["damping"][indices] *= spec.pd_damping_scale * damping_spread
            env.gym.set_actor_dof_properties(
                env.envs[env_index], env.robot_handles[env_index], properties
            )

    def _perturb_observations(self, observations):
        spec = self.spec
        perturbs_q = bool(spec.obs_q_noise_rad or spec.obs_q_bias_rad)
        if not (
            perturbs_q or spec.obs_dq_noise_rad_s or spec.obs_target_noise_rad
        ):
            return observations
        env = self.env
        noisy = observations.clone()
        if perturbs_q:
            # Re-derived through normalize_positions rather than scaled in
            # normalized space: the rad-to-normalized factor 2/(upper-lower)
            # varies about twelvefold across these joints, and this also
            # reproduces the clamp the environment applies.
            measured = env.q + self._q_bias
            if spec.obs_q_noise_rad:
                measured = measured + (
                    self._randn((env.num_envs, NUM_JOINTS)) * spec.obs_q_noise_rad
                )
            noisy[:, :NUM_JOINTS] = env.normalize_positions(measured)
        if spec.obs_target_noise_rad:
            noisy[:, NUM_JOINTS : 2 * NUM_JOINTS] += (
                self._randn((env.num_envs, NUM_JOINTS)) * spec.obs_target_noise_rad
            )
        if spec.obs_dq_noise_rad_s:
            noisy[:, 2 * NUM_JOINTS : 3 * NUM_JOINTS] += (
                self._randn((env.num_envs, NUM_JOINTS)) * spec.obs_dq_noise_rad_s
            )
        return noisy

    def _offset_initial_state(self, env_ids, spec):
        env = self.env
        order = env.demo_to_asset_tensor
        count = env_ids.numel()
        # Read-modify-write of the full buffer, then one indexed upload: Isaac
        # Gym keeps only the last indexed DOF-state write of a frame, so writing
        # a partial buffer here would discard the reset seeding that
        # _orig_reset_idx just uploaded.
        subset = env.dof_state[env_ids]
        if spec.init_q_offset_rad:
            positions = subset[:, order, 0] + (
                self._randn((count, NUM_JOINTS)) * spec.init_q_offset_rad
            )
            subset[:, order, 0] = torch.clamp(
                positions, env.joint_lower_limits, env.joint_upper_limits
            )
        if spec.init_dq_offset_rad_s:
            subset[:, order, 1] += (
                self._randn((count, NUM_JOINTS)) * spec.init_dq_offset_rad_s
            )
        env.dof_state[env_ids] = subset
        env._upload_dof_state(env.actor_indices[env_ids])


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #


def load_configuration(config_path, variant, args):
    with open(config_path, "r", encoding="utf-8") as handle:
        saved = json.load(handle)
    if "env_cfg" not in saved or "train_cfg" not in saved:
        raise ValueError("Saved configuration needs env_cfg and train_cfg")
    env_cfg = SimToolRealCfg()
    train_cfg = SimToolRealTrainCfg()
    update_config_from_dict(env_cfg, saved["env_cfg"])
    update_config_from_dict(train_cfg, saved["train_cfg"])
    for field_name, value in VARIANTS[variant].items():
        setattr(env_cfg.asset, field_name, value)
    env_cfg.seed = int(args.seed)
    env_cfg.env.num_envs = int(args.num_envs)
    env_cfg.env.episode_length = int(args.episode_length)
    env_cfg.env.play = True
    env_cfg.viewer.enable_viewer = False
    # A ghost would make actors_per_env 2 and break the indexed DOF upload the
    # initial-state offset relies on.
    env_cfg.viewer.reference_ghost = False
    return env_cfg, train_cfg


def observation_diagnostics(env, runner):
    """Per-joint amplification from radians of q noise to network input units."""
    normalizer = runner.actor_obs_normalizer
    std = getattr(normalizer, "_std", None)
    if std is None:
        return {}
    std = std.detach().flatten().float().cpu().numpy()
    ranges = (env.joint_upper_limits - env.joint_lower_limits).cpu().numpy()
    q_gain = (2.0 / ranges) / np.maximum(std[:NUM_JOINTS], 1e-9)
    return {
        "obs_q_rad_to_network_units": [float(v) for v in q_gain],
        "obs_dq_normalizer_std": [
            float(v) for v in std[2 * NUM_JOINTS : 3 * NUM_JOINTS]
        ],
    }


def run_condition(harness, evaluator, runner, condition, seed):
    # The evaluator does not seed, and env.step's auto-reset draws its RSI
    # indices from the global generator, so every condition must start from the
    # same global RNG state or a zero-magnitude condition would not reproduce
    # the baseline.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    harness.activate(condition.spec)
    pd_readback = None
    if condition.spec.touches_pd() and "arm" in condition.spec.pd_blocks:
        pd_readback = assert_pd_applied(harness, harness.env, condition.spec)
    start = time.perf_counter()
    metrics = evaluator(0, runner)
    elapsed = time.perf_counter() - start
    harness.deactivate()
    non_finite = [k for k, v in metrics.items() if not math.isfinite(float(v))]
    return {
        "name": condition.name,
        "family": condition.family,
        "magnitude": condition.magnitude,
        "gate": condition.gate,
        "spec": asdict(condition.spec),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "wall_time_s": elapsed,
        "diverged": bool(non_finite),
        "pd_readback_ok": pd_readback,
    }


def assert_pd_applied(harness, env, spec, sample_count=4):
    """Read the gains back out of the simulator while the spec is active.

    This is the one mechanism whose effect is not structurally guaranteed: a
    per-actor property write after prepare_sim under the GPU pipeline could be
    silently ignored, and that would look exactly like a robust policy.
    """
    if spec.pd_per_env_log2_halfwidth:
        return None  # Per-env draws have no single expected value.
    indices = env.demo_to_asset[:NUM_ARM_JOINTS]
    expected = harness._nominal_properties["stiffness"][indices] * spec.pd_stiffness_scale
    for env_index in range(min(sample_count, env.num_envs)):
        current = env.gym.get_actor_dof_properties(
            env.envs[env_index], env.robot_handles[env_index]
        )["stiffness"][indices]
        if not np.allclose(current, expected, rtol=1e-4):
            raise AssertionError(
                "PD write did not reach the simulator in env {}: {} vs {}".format(
                    env_index, current, expected
                )
            )
    return True


def run_worker(args):
    variant = args.variant
    env_cfg, train_cfg = load_configuration(args.config, variant, args)
    env = MotionImitationEnv(
        env_cfg,
        sim_device=args.sim_device,
        headless=True,
        num_envs_override=None,
    )
    runner = None
    try:
        expected_obs = 108
        if env.num_obs != expected_obs:
            raise ValueError(
                "Observation layout changed: {} != {}. The noise slices in this "
                "probe would perturb the wrong block.".format(
                    env.num_obs, expected_obs
                )
            )
        runner = PPO(env, train_cfg, log_dir=None, device=env.device)
        runner.load(args.checkpoint, load_optimizer=False, load_normalizers=True)
        evaluator = DeterministicEvaluator(
            env,
            interval=1,
            seed=args.seed,
            fixed_phases=args.fixed_phases,
        )

        baseline_condition = Condition("baseline", "baseline", "-")
        results = []

        # Reference numbers taken before the shims exist at all, so "the shim is
        # a no-op" is a real claim and not a self-fulfilling one.
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        reference = evaluator(0, runner)
        reference = {k: float(v) for k, v in reference.items()}

        harness = PerturbationHarness(env, args.seed)
        conditions = [baseline_condition]
        if variant == "nominal" and not args.baseline_only:
            conditions += runtime_conditions()
        conditions.append(Condition("baseline_final", "baseline", "-"))

        state_checks = {}
        for condition in conditions:
            record = run_condition(harness, evaluator, runner, condition, args.seed)
            results.append(record)
            print(
                "  [{}] {:<22s} arm_rms={:.6f} hand_rms={:.6f} early={:.3f}".format(
                    variant,
                    condition.name,
                    _suite_mean(record["metrics"], "mean_rms_position_error"),
                    _suite_mean(record["metrics"], "mean_rms_hand_position_error"),
                    _suite_mean(record["metrics"], "early_termination_fraction"),
                ),
                flush=True,
            )

        state_checks["shim_is_noop_delta"] = _metrics_delta(
            reference, results[0]["metrics"]
        )
        state_checks["sandwich_delta"] = _metrics_delta(
            results[0]["metrics"], results[-1]["metrics"]
        )
        state_checks["pd_reaches_simulator"] = any(
            r.get("pd_readback_ok") for r in results
        ) or all(r.get("pd_readback_ok") is None for r in results)

        payload = {
            "variant": variant,
            "reference_metrics": reference,
            "conditions": results,
            "state_checks": state_checks,
            "diagnostics": observation_diagnostics(env, runner),
            "mean_step_time_s": _mean_step_time(results, args.episode_length),
        }
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
    finally:
        if runner is not None:
            runner.close()
        env.close()


def _mean_step_time(results, episode_length):
    baselines = [r for r in results if r["family"] == "baseline"]
    if not baselines:
        return None
    # Two suites per evaluator call, each at most episode_length steps.
    return baselines[0]["wall_time_s"] / float(2 * episode_length)


def _metrics_equal(left, right):
    return _metrics_delta(left, right) == 0.0


def _metrics_delta(left, right):
    """Largest relative disagreement between two metric dicts.

    Exact zero is the expectation for a contact-free build: same process, same
    sim, so PhysX is bit-deterministic. Contacts break that determinism at the
    1e-7 level, which is why this returns a magnitude rather than a boolean.
    """
    if set(left) != set(right):
        return float("inf")
    worst = 0.0
    for key in left:
        a, b = float(left[key]), float(right[key])
        scale = max(abs(a), abs(b), 1e-12)
        worst = max(worst, abs(a - b) / scale)
    return worst


def _suite_mean(metrics, name):
    fixed = metrics.get("evaluation_fixed_" + name)
    uniform = metrics.get("evaluation_uniform_" + name)
    if fixed is None or uniform is None:
        return float("nan")
    return 0.5 * (float(fixed) + float(uniform))


def _suite_max(metrics, name):
    fixed = metrics.get("evaluation_fixed_" + name, float("nan"))
    uniform = metrics.get("evaluation_uniform_" + name, float("nan"))
    return max(float(fixed), float(uniform))


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


def derive(record, baseline):
    metrics, base = record["metrics"], baseline["metrics"]
    arm = _suite_mean(metrics, "mean_rms_position_error")
    hand = _suite_mean(metrics, "mean_rms_hand_position_error")
    arm_base = _suite_mean(base, "mean_rms_position_error")
    hand_base = _suite_mean(base, "mean_rms_hand_position_error")
    early = _suite_mean(metrics, "early_termination_fraction")
    early_base = _suite_mean(base, "early_termination_fraction")
    length = _suite_mean(metrics, "mean_episode_length")
    length_base = _suite_mean(base, "mean_episode_length")

    arm_ratio = arm / arm_base if arm_base > 0 else float("nan")
    hand_ratio = hand / hand_base if hand_base > 0 else float("nan")
    tracking = 0.5 * (math.log2(max(arm_ratio, 1e-9)) + math.log2(max(hand_ratio, 1e-9)))
    early_delta = early - early_base
    fragility = tracking + 4.0 * max(early_delta, 0.0)

    flags = []
    if record.get("diverged"):
        flags.append("diverged")
    # A condition that dies early reports the error of its few good steps and
    # can look better than the baseline; the flag keeps that from reading as
    # robustness.
    if length_base > 0 and length < 0.9 * length_base:
        flags.append("truncated")

    return {
        "arm_rms": arm,
        "hand_rms": hand,
        "arm_ratio": arm_ratio,
        "hand_ratio": hand_ratio,
        "early_fraction": early,
        "early_delta": early_delta,
        "episode_length": length,
        "peak_arm": _suite_max(metrics, "max_abs_position_error"),
        "tracking": tracking,
        "fragility": fragility,
        "flags": flags,
    }


def verdict_for(fragility, flags, noise_floor):
    if "diverged" in flags or "truncated" in flags:
        return "CRITICAL"
    if fragility < max(3.0 * noise_floor, 1e-6):
        return "below noise floor"
    if fragility < 0.15:
        return "negligible"
    if fragility < 1.0:
        return "randomize"
    return "CRITICAL"


RANDOMIZATION_HINTS = {
    "pd_gain": (
        "randomize the damping/stiffness ratio per environment -- a common "
        "factor on both is a no-op with gravity off and no contact"
    ),
    "latency": "randomize the action delay per episode",
    "obs_noise": "add per-step observation noise during training",
    "init_state": "perturb the reset state",
    "physics": "retrain with this physics setting enabled",
    "combo": "randomize these jointly, not one axis at a time",
}


def build_report(payloads, args):
    nominal = payloads["nominal"]
    baseline = next(c for c in nominal["conditions"] if c["name"] == "baseline")

    noise_floor = 0.0
    if "nominal_control" in payloads:
        control = next(
            c for c in payloads["nominal_control"]["conditions"] if c["name"] == "baseline"
        )
        noise_floor = abs(derive(control, baseline)["fragility"])

    rows = []
    gates = {"null": {}, "efficacy": {}}
    for record in nominal["conditions"]:
        if record["name"] in ("baseline", "baseline_final"):
            continue
        derived = derive(record, baseline)
        if record["gate"] == "null":
            gates["null"][record["name"]] = _metrics_equal(
                baseline["metrics"], record["metrics"]
            )
            continue
        if record["gate"] == "efficacy":
            gates["efficacy"][record["name"]] = {
                "passed": bool(
                    derived["arm_ratio"] >= 2.0 or derived["early_delta"] > 0.2
                ),
                "arm_ratio": derived["arm_ratio"],
                "early_delta": derived["early_delta"],
            }
            continue
        rows.append((record, derived))

    for variant, payload in payloads.items():
        if variant in ("nominal", "nominal_control"):
            continue
        record = next(c for c in payload["conditions"] if c["name"] == "baseline")
        record = dict(record, name=variant, family="physics", magnitude="on")
        rows.append((record, derive(record, baseline)))

    rows.sort(key=lambda item: item[1]["fragility"], reverse=True)

    families = {}
    for record, derived in rows:
        entry = families.setdefault(
            record["family"], {"max_fragility": -math.inf, "worst": None}
        )
        if derived["fragility"] > entry["max_fragility"]:
            entry["max_fragility"] = derived["fragility"]
            entry["worst"] = record["name"]
    for family, entry in families.items():
        entry["recommendation"] = (
            RANDOMIZATION_HINTS.get(family, "randomize")
            if entry["max_fragility"] >= 0.15
            else "skip"
        )

    repeat_noise = {}
    for variant, payload in payloads.items():
        first = next(c for c in payload["conditions"] if c["name"] == "baseline")
        last = next(
            (c for c in payload["conditions"] if c["name"] == "baseline_final"), None
        )
        # The same condition run twice inside one process. Zero for a
        # contact-free build; non-zero once contacts make PhysX
        # non-deterministic. Either way it is this variant's resolution.
        repeat_noise[variant] = (
            abs(derive(last, first)["fragility"]) if last is not None else 0.0
        )

    build_effect = {}
    nominal_step = nominal.get("mean_step_time_s") or float("nan")
    for variant, payload in payloads.items():
        if variant in ("nominal", "nominal_control"):
            continue
        step = payload.get("mean_step_time_s") or float("nan")
        record = next(c for c in payload["conditions"] if c["name"] == "baseline")
        derived = derive(record, baseline)
        # Cross-process comparisons cannot use exact equality, so a build flag
        # is only credible if it moved the metrics well past the rebuild noise
        # floor, or visibly changed the cost of a step.
        resolution = max(noise_floor, repeat_noise.get(variant, 0.0))
        moved_metrics = abs(derived["fragility"]) > max(3.0 * resolution, 0.05)
        ratio = step / nominal_step if nominal_step else float("nan")
        build_effect[variant] = {
            "step_time_s": step,
            "nominal_step_time_s": nominal_step,
            "ratio": ratio,
            "repeat_noise": repeat_noise.get(variant, 0.0),
            "verdict": (
                "flag took effect"
                if moved_metrics or ratio > 1.2
                else "INCONCLUSIVE: flag may not have taken effect"
            ),
        }

    return {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "build_effect": build_effect,
        "repeat_noise": repeat_noise,
        "protocol": {
            "seed": args.seed,
            "num_envs": args.num_envs,
            "episode_length": args.episode_length,
            "fixed_phases": args.fixed_phases,
            "sim_device": args.sim_device,
            "note": (
                "DeterministicEvaluator over two RSI cohorts; NOT comparable to "
                "scripts/evaluate.py numbers, which use one env over the full motion."
            ),
        },
        "diagnostics": nominal.get("diagnostics", {}),
        "noise_floor": noise_floor,
        "baseline": {"metrics": baseline["metrics"], "derived": derive(baseline, baseline)},
        "state_checks": {v: p["state_checks"] for v, p in payloads.items()},
        "gates": gates,
        "conditions": [
            dict(record, derived=derived, verdict=verdict_for(
                derived["fragility"], derived["flags"], noise_floor))
            for record, derived in rows
        ],
        "families": families,
        "not_probed": [
            {
                "name": "external_body_push",
                "reason": (
                    "The environment never acquires the rigid-body state tensor, so a "
                    "script cannot verify the force landed; DOF_MODE_POS also absorbs it "
                    "in PhysX's implicit joint PD. Needs a library change."
                ),
            },
            {
                "name": "sensing_latency",
                "reason": "Delay on the observation rather than the action; deferred.",
            },
            {
                "name": "link_mass_inertia",
                "reason": "Script-feasible, but with gravity off it acts only through inertia.",
            },
        ],
    }


def print_report(report):
    print()
    print("=" * 108)
    print("FRAGILITY PROBE  {}".format(report["checkpoint"]))
    print("=" * 108)

    diagnostics = report.get("diagnostics", {})
    gains = diagnostics.get("obs_q_rad_to_network_units")
    if gains:
        print(
            "q noise amplification into network units: min {:.1f} / median {:.1f} / "
            "max {:.1f} per rad".format(
                min(gains), float(np.median(gains)), max(gains)
            )
        )
    base = report["baseline"]["derived"]
    print(
        "baseline: arm_rms {:.6f} rad | hand_rms {:.6f} rad | early {:.3f} | "
        "episode {:.1f} steps".format(
            base["arm_rms"], base["hand_rms"], base["early_fraction"],
            base["episode_length"],
        )
    )
    print("rebuild noise floor (fragility): {:.4f}".format(report["noise_floor"]))
    print()

    null_gates = report["gates"]["null"]
    efficacy_gates = report["gates"]["efficacy"]
    print("gates  null (must reproduce baseline exactly): {}".format(
        ", ".join("{}={}".format(k, "OK" if v else "FAIL") for k, v in null_gates.items())
        or "none"))
    print("       efficacy (must break the policy):")
    for name, entry in efficacy_gates.items():
        print("         {:<20s} {:<4s} arm {:.2f}x  early +{:.3f}".format(
            name, "OK" if entry["passed"] else "FAIL",
            entry["arm_ratio"], entry["early_delta"]))
    for variant, noise in sorted(report.get("repeat_noise", {}).items()):
        if noise == 0.0:
            print("       {:<16s} repeats bit-identically".format(variant))
        else:
            print("       {:<16s} repeat noise {:.2e} on the fragility scale "
                  "(PhysX contacts are not bit-deterministic)".format(variant, noise))
    for variant, entry in report.get("build_effect", {}).items():
        print("       {:<16s} step time {:.2f} ms vs {:.2f} nominal ({:.2f}x) -> {}".format(
            variant, 1000 * entry["step_time_s"], 1000 * entry["nominal_step_time_s"],
            entry["ratio"], entry["verdict"]))
    print()

    header = "{:<20s} {:<11s} {:<17s} {:>9s} {:>7s} {:>9s} {:>7s} {:>7s} {:>7s}  {}"
    print(header.format("condition", "family", "magnitude", "arm_rms", "xnom",
                        "hand_rms", "xnom", "early%", "frag", "verdict"))
    print("-" * 118)
    for record in report["conditions"]:
        derived = record["derived"]
        flags = "".join(" [" + f + "]" for f in derived["flags"])
        print(header.format(
            record["name"], record["family"], record["magnitude"],
            "{:.6f}".format(derived["arm_rms"]),
            "{:.2f}x".format(derived["arm_ratio"]),
            "{:.6f}".format(derived["hand_rms"]),
            "{:.2f}x".format(derived["hand_ratio"]),
            "{:.1f}".format(100.0 * derived["early_fraction"]),
            "{:.2f}".format(derived["fragility"]),
            record["verdict"] + flags,
        ))
    print()
    print("by family:")
    for family, entry in sorted(
        report["families"].items(), key=lambda kv: -kv[1]["max_fragility"]
    ):
        print("  {:<12s} worst {:<20s} fragility {:>6.2f}   -> {}".format(
            family, entry["worst"], entry["max_fragility"], entry["recommendation"]))
    print()
    print("not probed:")
    for item in report["not_probed"]:
        print("  {}: {}".format(item["name"], item["reason"]))


def run_orchestrator(args):
    scratch = args.output.parent / ".probe_workers"
    scratch.mkdir(parents=True, exist_ok=True)
    payloads = {}
    try:
        for variant in args.variants:
            worker_output = scratch / "{}.json".format(variant)
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--worker", "--variant", variant,
                "--checkpoint", str(args.checkpoint),
                "--config", str(args.config),
                "--output", str(worker_output),
                "--num-envs", str(args.num_envs),
                "--episode-length", str(args.episode_length),
                "--seed", str(args.seed),
                "--sim-device", args.sim_device,
                "--fixed-phases", *[str(p) for p in args.fixed_phases],
            ]
            if variant != "nominal":
                command.append("--baseline-only")
            print("running variant {} ...".format(variant), flush=True)
            completed = subprocess.run(command)
            if completed.returncode != 0:
                raise RuntimeError(
                    "Variant {} failed with code {}".format(variant, completed.returncode)
                )
            with open(worker_output, "r", encoding="utf-8") as handle:
                payloads[variant] = json.load(handle)
    finally:
        pass

    report = build_report(payloads, args)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    # The aggregated report carries every worker metric, so the per-variant
    # scratch files are no longer worth keeping in the run directory.
    for worker_output in scratch.glob("*.json"):
        worker_output.unlink()
    scratch.rmdir()
    print_report(report)
    print()
    print("report: {}".format(args.output))

    failures = [k for k, v in report["gates"]["null"].items() if not v]
    failures += [k for k, v in report["gates"]["efficacy"].items() if not v["passed"]]
    # A contact-free build that fails to repeat itself means a shim leaked
    # state; a contact build is allowed its own scatter, which is reported
    # and used as that variant's resolution instead.
    failures += [
        "{}/repeatability".format(variant)
        for variant, noise in report.get("repeat_noise", {}).items()
        if noise > 0.0 and variant in ("nominal", "nominal_control", "gravity")
    ]
    if failures:
        print()
        print("GATES FAILED: {}".format(", ".join(failures)))
        print("A perturbation that does nothing is indistinguishable from a robust")
        print("policy, so the table above cannot be trusted for those families.")
        return 1
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--fixed-phases", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75]
    )
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument(
        "--variants", nargs="+", default=list(VARIANTS),
        choices=list(VARIANTS),
        help="Build variants to run, one subprocess each.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--variant", default="nominal", help=argparse.SUPPRESS)
    parser.add_argument("--baseline-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise SystemExit("Checkpoint not found: {}".format(args.checkpoint))
    if args.config is None:
        args.config = args.checkpoint.parent / "config.json"
    args.config = args.config.expanduser().resolve()
    if not args.config.is_file():
        raise SystemExit("Configuration not found: {}".format(args.config))
    if args.output is None:
        args.output = args.checkpoint.parent / "robustness_probe_{}.json".format(
            args.checkpoint.stem
        )
    args.output = args.output.expanduser().resolve()
    if "nominal" not in args.variants:
        raise SystemExit("The nominal variant is the baseline; it cannot be skipped")
    return args


def main():
    args = parse_args()
    if args.worker:
        run_worker(args)
        return 0
    return run_orchestrator(args)


if __name__ == "__main__":
    sys.exit(main())
