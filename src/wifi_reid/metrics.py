"""Retrieval metrics for closed-set person re-identification."""

from __future__ import annotations

import numpy as np


def cmc_and_map(similarity, probe_ids, gallery_ids, ranks=(1, 5, 10)):
    hits = {rank: 0 for rank in ranks}
    average_precisions = []

    for i in range(similarity.shape[0]):
        order = np.argsort(-similarity[i])
        relevant = (gallery_ids[order] == probe_ids[i]).astype(int)
        if relevant.sum() == 0:
            continue
        for rank in ranks:
            if relevant[:rank].any():
                hits[rank] += 1
        cumulative = np.cumsum(relevant)
        average_precisions.append(
            float((cumulative / (np.arange(len(relevant)) + 1) * relevant).sum() / relevant.sum())
        )

    output = {f"rank_{rank}": hits[rank] / similarity.shape[0] for rank in ranks}
    output["mAP"] = float(np.mean(average_precisions)) if average_precisions else 0.0
    return output


def macro_f1(similarity, probe_ids, gallery_ids):
    predictions = gallery_ids[np.argmax(similarity, axis=1)]
    classes = np.unique(np.concatenate([probe_ids, gallery_ids]))
    f1_scores = []
    for cls in classes:
        tp = int(((predictions == cls) & (probe_ids == cls)).sum())
        fp = int(((predictions == cls) & (probe_ids != cls)).sum())
        fn = int(((predictions != cls) & (probe_ids == cls)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_scores.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return float(np.mean(f1_scores))


def bootstrap_rank1_ci(similarity, probe_ids, gallery_ids, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    n = similarity.shape[0]
    correct = np.array(
        [gallery_ids[np.argmax(similarity[i])] == probe_ids[i] for i in range(n)]
    )
    samples = [correct[rng.randint(0, n, n)].mean() for _ in range(n_boot)]
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))
