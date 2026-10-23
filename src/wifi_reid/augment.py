"""Physics-grounded and vision-style control augmentations for CSI amplitude matrices."""

from __future__ import annotations

import math
import numpy as np


class ComposableAug:
    """Composable CSI augmentation policy."""

    VALID = ("gain", "dropout", "resample", "noise")
    INVALID = ("flip", "crop", "jitter")

    def __init__(self, transforms, strength: float = 1.0, name: str | None = None):
        self.transforms = tuple(transforms)
        self.strength = float(strength)
        self.name = name or ("+".join(self.transforms) if self.transforms else "none")

    def __call__(self, window: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        out = window.copy()
        strength = self.strength
        for transform in self.transforms:
            if transform == "gain":
                out = out * rng.uniform(1 - 0.1 * strength, 1 + 0.1 * strength)
            elif transform == "dropout":
                out = self._dropout(out, rng, strength)
            elif transform == "resample":
                out = self._resample(out, rng, strength)
            elif transform == "noise":
                out = self._noise(out, rng, strength)
            elif transform == "flip":
                if rng.rand() < min(0.5 * strength, 1.0):
                    out = out[:, ::-1].copy()
            elif transform == "crop":
                out = self._crop(out, rng, strength)
            elif transform == "jitter":
                out = out * (1.0 + rng.uniform(-0.3 * strength, 0.3 * strength))
            else:
                raise ValueError(f"Unknown transform: {transform}")
        return out.astype(np.float32, copy=False)

    @staticmethod
    def _dropout(window, rng, strength):
        kmax = max(1, int(round(5 * strength)))
        n_subcarriers = window.shape[1]
        k = rng.randint(1, min(kmax, n_subcarriers - 2) + 1)
        for carrier in rng.choice(n_subcarriers, size=k, replace=False):
            left = carrier - 1 if carrier > 0 else carrier + 2
            right = carrier + 1 if carrier < n_subcarriers - 1 else carrier - 2
            window[:, carrier] = 0.5 * (window[:, left] + window[:, right])
        return window

    @staticmethod
    def _resample(window, rng, strength):
        factor = rng.uniform(1 - 0.05 * strength, 1 + 0.05 * strength)
        time_steps, n_subcarriers = window.shape
        new_time = max(2, int(round(time_steps * factor)))
        result = np.empty((new_time, n_subcarriers), dtype=np.float32)
        source = np.linspace(0, time_steps - 1, new_time)
        base = np.arange(time_steps)
        for carrier in range(n_subcarriers):
            result[:, carrier] = np.interp(source, base, window[:, carrier])
        if new_time >= time_steps:
            start = rng.randint(0, new_time - time_steps + 1)
            return result[start : start + time_steps]
        return np.concatenate(
            [result, np.repeat(result[-1:], time_steps - new_time, axis=0)], axis=0
        )

    @staticmethod
    def _noise(window, rng, strength):
        snr_db = 30.0 / max(strength, 1e-6)
        noise_power = float(np.mean(window**2)) / (10 ** (snr_db / 10.0))
        noise = rng.normal(0, math.sqrt(noise_power), size=window.shape).astype(np.float32)
        return window + noise

    @staticmethod
    def _crop(window, rng, strength):
        time_steps, n_subcarriers = window.shape
        scale = rng.uniform(max(0.2, 1 - 0.2 * strength), 1.0)
        crop_t = max(2, int(time_steps * scale))
        crop_s = max(2, int(n_subcarriers * scale))
        t0 = rng.randint(0, time_steps - crop_t + 1)
        s0 = rng.randint(0, n_subcarriers - crop_s + 1)
        crop = window[t0 : t0 + crop_t, s0 : s0 + crop_s]

        temporal = np.empty((time_steps, crop.shape[1]), dtype=np.float32)
        time_index = np.linspace(0, crop.shape[0] - 1, time_steps)
        for j in range(crop.shape[1]):
            temporal[:, j] = np.interp(time_index, np.arange(crop.shape[0]), crop[:, j])

        output = np.empty((time_steps, n_subcarriers), dtype=np.float32)
        sub_index = np.linspace(0, crop.shape[1] - 1, n_subcarriers)
        for i in range(time_steps):
            output[i, :] = np.interp(sub_index, np.arange(crop.shape[1]), temporal[i, :])
        return output


def SubCarrierAug(strength: float = 1.0) -> ComposableAug:
    return ComposableAug(ComposableAug.VALID, strength, "SubCarrierAug")


def VisionAug(strength: float = 1.0) -> ComposableAug:
    return ComposableAug(ComposableAug.INVALID, strength, "VisionAug")
