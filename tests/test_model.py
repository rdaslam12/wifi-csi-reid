import torch

from wifi_reid.model import SubCarrierReID


def test_embedding_shape():
    model = SubCarrierReID(input_shape=(1, 100, 52), n_classes=81, dual_axis=True)
    x = torch.randn(4, 1, 100, 52)
    assert model.embed(x).shape == (4, 128)


def test_siamese_forward_shapes():
    model = SubCarrierReID(input_shape=(1, 100, 52), n_classes=81)
    x = torch.randn(2, 1, 100, 52)
    z_i, z_j, logits_i, logits_j = model(x, x)
    assert z_i.shape == z_j.shape == (2, 128)
    assert logits_i.shape == logits_j.shape == (2, 81)
