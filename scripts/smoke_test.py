"""Fast synthetic smoke test; does not require Wi-PER81."""

import numpy as np
import torch

from wifi_reid.augment import ComposableAug, SubCarrierAug
from wifi_reid.model import SubCarrierReID


def main():
    rng = np.random.RandomState(0)
    window = rng.normal(size=(100, 52)).astype(np.float32)

    augmented = SubCarrierAug()(window, np.random.RandomState(1))
    assert augmented.shape == window.shape
    assert np.isfinite(augmented).all()

    flipped = ComposableAug(("flip",), strength=2.0)(window, np.random.RandomState(0))
    assert np.allclose(flipped, window[:, ::-1])

    model = SubCarrierReID(input_shape=(1, 100, 52), n_classes=81, dual_axis=True)
    x = torch.randn(2, 1, 100, 52)
    embedding = model.embed(x)
    assert embedding.shape == (2, 128)

    print("smoke test: PASS")
    print(f"embedding shape: {tuple(embedding.shape)}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
