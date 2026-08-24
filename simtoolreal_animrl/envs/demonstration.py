"""Validated discrete 60 Hz joint-demonstration loader."""

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Union

import numpy as np
import torch


class ReferenceSample(NamedTuple):
    q: torch.Tensor
    dq: torch.Tensor


@dataclass(frozen=True)
class JointDemonstration60Hz:
    path: Path
    timestamp: torch.Tensor
    monotonic_timestamp: torch.Tensor
    q: torch.Tensor
    dq: torch.Tensor
    frequency_hz: float

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: Union[str, torch.device],
        expected_hz: float = 60.0,
    ) -> "JointDemonstration60Hz":
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError("Demonstration not found: {}".format(resolved))

        required = {
            "timestamp",
            "monotonic_timestamp",
            "arm_q",
            "arm_dq",
            "hand_q_measured",
            "hand_dq_measured",
        }
        with np.load(str(resolved)) as archive:
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError("Demonstration is missing fields: {}".format(missing))
            arrays = {name: np.asarray(archive[name]) for name in required}

        timestamp = arrays["timestamp"].astype(np.float64, copy=False)
        monotonic = arrays["monotonic_timestamp"].astype(np.float64, copy=False)
        arm_q = arrays["arm_q"].astype(np.float32, copy=False)
        arm_dq = arrays["arm_dq"].astype(np.float32, copy=False)
        hand_q = arrays["hand_q_measured"].astype(np.float32, copy=False)
        hand_dq = arrays["hand_dq_measured"].astype(np.float32, copy=False)

        sample_count = len(timestamp)
        expected_shapes = {
            "monotonic_timestamp": (sample_count,),
            "arm_q": (sample_count, 6),
            "arm_dq": (sample_count, 6),
            "hand_q_measured": (sample_count, 20),
            "hand_dq_measured": (sample_count, 20),
        }
        for name, shape in expected_shapes.items():
            if arrays[name].shape != shape:
                raise ValueError(
                    "{} has shape {}, expected {}".format(
                        name, arrays[name].shape, shape
                    )
                )
        if sample_count < 3:
            raise ValueError("The demonstration must contain at least three samples")
        for name, values in arrays.items():
            if not np.all(np.isfinite(values)):
                raise ValueError("{} contains NaN or infinite values".format(name))

        relative_time = monotonic - monotonic[0]
        intervals = np.diff(relative_time)
        if np.any(intervals <= 0.0):
            raise ValueError("monotonic_timestamp must be strictly increasing")
        measured_hz = 1.0 / float(np.median(intervals))
        if not np.isclose(measured_hz, expected_hz, rtol=0.0, atol=0.05):
            raise ValueError(
                "Expected a {:.2f} Hz demonstration, measured {:.5f} Hz".format(
                    expected_hz, measured_hz
                )
            )

        q = np.concatenate((arm_q, hand_q), axis=1)
        dq = np.concatenate((arm_dq, hand_dq), axis=1)
        torch_device = torch.device(device)
        return cls(
            path=resolved,
            timestamp=torch.as_tensor(timestamp, dtype=torch.float64, device=torch_device),
            monotonic_timestamp=torch.as_tensor(
                monotonic, dtype=torch.float64, device=torch_device
            ),
            q=torch.as_tensor(q, dtype=torch.float32, device=torch_device),
            dq=torch.as_tensor(dq, dtype=torch.float32, device=torch_device),
            frequency_hz=measured_hz,
        )

    @property
    def sample_count(self) -> int:
        return int(self.q.shape[0])

    @property
    def last_index(self) -> int:
        return self.sample_count - 1

    def sample(self, indices: torch.Tensor) -> ReferenceSample:
        if indices.dtype != torch.long:
            indices = indices.long()
        if torch.any(indices < 0) or torch.any(indices > self.last_index):
            raise IndexError("Reference index outside demonstration")
        return ReferenceSample(q=self.q[indices], dq=self.dq[indices])
