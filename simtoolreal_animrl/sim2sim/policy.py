"""Saved AnimRL configuration and deterministic actor checkpoint loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

from simtoolreal_animrl.runners.modules.normalizer import EmpiricalNormalization
from simtoolreal_animrl.runners.modules.policy import Policy

from .constants import ACTION_DIM, BASE_OBSERVATION_DIM


@dataclass(frozen=True)
class LoadedRun:
    checkpoint_path: Path
    config_path: Path
    repo_root: Path
    saved: Mapping[str, Any]

    @property
    def env_cfg(self) -> Mapping[str, Any]:
        return self.saved["env_cfg"]

    @property
    def train_cfg(self) -> Mapping[str, Any]:
        return self.saved["train_cfg"]


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "simtoolreal_animrl").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError("Could not find the simtoolreal_animrl repository root")


def load_saved_run(
    checkpoint_path: Path, config_path: Optional[Path] = None
) -> LoadedRun:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint))
    config = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else checkpoint.parent / "config.json"
    )
    if not config.is_file():
        raise FileNotFoundError("Saved configuration not found: {}".format(config))
    with config.open("r", encoding="utf-8") as stream:
        saved = json.load(stream)
    if "env_cfg" not in saved or "train_cfg" not in saved:
        raise ValueError("Saved configuration needs env_cfg and train_cfg")
    observation_dim = int(saved.get("observation_dim", saved["env_cfg"]["env"]["num_observations"]))
    action_dim = int(saved["env_cfg"]["env"]["num_actions"])
    if observation_dim != BASE_OBSERVATION_DIM or action_dim != ACTION_DIM:
        raise ValueError(
            "This MuJoCo runner requires the blind 108/26 contract, got {}/{}".format(
                observation_dim, action_dim
            )
        )
    if bool(saved["env_cfg"]["contact"].get("observe_fingertip_forces", False)):
        raise ValueError("This runner intentionally rejects force-observation checkpoints")
    return LoadedRun(
        checkpoint_path=checkpoint,
        config_path=config,
        repo_root=_find_repo_root(Path(__file__).resolve()),
        saved=saved,
    )


class AnimRLInferencePolicy:
    """The trained actor and its frozen empirical observation normalizer."""

    def __init__(self, run: LoadedRun, device: str = "cpu") -> None:
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for inference but is unavailable")
        self.run = run
        self.device = requested
        policy_cfg = run.train_cfg["policy"]
        self.actor = Policy(
            num_obs=BASE_OBSERVATION_DIM,
            num_actions=ACTION_DIM,
            hidden_dims=list(policy_cfg["actor_hidden_dims"]),
            activation=str(policy_cfg["activation"]),
            log_std_init=float(policy_cfg["log_std_init"]),
            max_action_std=float(policy_cfg["max_action_std"]),
            device=str(self.device),
        )
        self.normalizer = EmpiricalNormalization(
            shape=BASE_OBSERVATION_DIM
        ).to(self.device)
        checkpoint = self._read_checkpoint()
        self.actor.load_state_dict(checkpoint["policy_dict"], strict=True)
        self.normalizer.load_state_dict(
            checkpoint["actor_obs_normalizer"], strict=True
        )
        infos = checkpoint.get("infos")
        if isinstance(infos, dict):
            self.normalizer.count = int(infos.get("actor_normalizer_count", 0))
        self.actor.eval()
        self.normalizer.eval()
        self.infos = infos

    def _read_checkpoint(self) -> Mapping[str, Any]:
        try:
            return torch.load(
                str(self.run.checkpoint_path),
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            return torch.load(str(self.run.checkpoint_path), map_location=self.device)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (BASE_OBSERVATION_DIM,):
            raise ValueError(
                "Observation has shape {}, expected ({},)".format(
                    observation.shape, BASE_OBSERVATION_DIM
                )
            )
        tensor = torch.from_numpy(observation).to(self.device).unsqueeze(0)
        with torch.inference_mode():
            normalized = self.normalizer(tensor)
            action = self.actor.act_inference(normalized)
        return action[0].detach().cpu().numpy().astype(np.float32)
