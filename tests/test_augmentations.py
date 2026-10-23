import numpy as np

from wifi_reid.augment import ComposableAug, SubCarrierAug, VisionAug


def test_flip_reverses_subcarrier_axis():
    window = np.arange(100 * 52, dtype=np.float32).reshape(100, 52)
    output = ComposableAug(("flip",), strength=2.0)(window, np.random.RandomState(0))
    assert np.array_equal(output, window[:, ::-1])


def test_augmentations_preserve_shape_and_finite_values():
    window = np.random.RandomState(0).normal(size=(100, 52)).astype(np.float32)
    for policy in (SubCarrierAug(), VisionAug()):
        output = policy(window, np.random.RandomState(3))
        assert output.shape == window.shape
        assert np.isfinite(output).all()
