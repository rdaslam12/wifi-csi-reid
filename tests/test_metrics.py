import numpy as np

from wifi_reid.metrics import cmc_and_map, macro_f1


def test_perfect_retrieval():
    similarity = np.eye(3, dtype=np.float32)
    ids = np.array([10, 20, 30])
    metrics = cmc_and_map(similarity, ids, ids, ranks=(1, 3))
    assert metrics["rank_1"] == 1.0
    assert metrics["rank_3"] == 1.0
    assert metrics["mAP"] == 1.0
    assert macro_f1(similarity, ids, ids) == 1.0
