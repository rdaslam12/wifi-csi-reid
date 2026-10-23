"""Dynamic Siamese pair sampling and optional semi-hard negative mining."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .data import select_windows


class DynamicPairDataset(Dataset):
    def __init__(
        self,
        store,
        representation,
        n_pairs,
        window_indices,
        aug=None,
        seed=0,
        positive_ratio=0.5,
        random_offset=False,
        hard_neg_ratio=0.0,
    ):
        self.store = store
        self.representation = representation
        self.aug = aug
        self.n_pairs = n_pairs
        self.positive_ratio = positive_ratio
        self.random_offset = random_offset
        self.hard_neg_ratio = hard_neg_ratio
        self.rng = np.random.RandomState(seed)
        self.windows = select_windows(store, window_indices)
        self.by_pid = {}
        for i, (_, _, pid, _) in enumerate(self.windows):
            self.by_pid.setdefault(pid, []).append(i)
        self.all_pids = sorted(self.by_pid)
        self.hard_neg_pool = None
        self.resample()

    def set_hard_negatives(self, pairs):
        self.hard_neg_pool = pairs

    def resample(self):
        pairs = []
        n_positive = int(self.n_pairs * self.positive_ratio)
        for _ in range(n_positive):
            pid = self.all_pids[self.rng.randint(len(self.all_pids))]
            candidates = self.by_pid[pid]
            if len(candidates) < 2:
                continue
            i, j = self.rng.choice(len(candidates), 2, replace=False)
            pairs.append((candidates[i], candidates[j], 1))

        n_negative = self.n_pairs - n_positive
        n_hard = int(n_negative * self.hard_neg_ratio) if self.hard_neg_pool else 0
        for _ in range(n_hard):
            i, j = self.hard_neg_pool[self.rng.randint(len(self.hard_neg_pool))]
            pairs.append((i, j, 0))
        for _ in range(n_negative - n_hard):
            a, b = self.rng.choice(len(self.all_pids), 2, replace=False)
            candidates_a = self.by_pid[self.all_pids[a]]
            candidates_b = self.by_pid[self.all_pids[b]]
            pairs.append(
                (
                    candidates_a[self.rng.randint(len(candidates_a))],
                    candidates_b[self.rng.randint(len(candidates_b))],
                    0,
                )
            )
        self.rng.shuffle(pairs)
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def _prepare(self, window_index, rng):
        record_index, wi, _, _ = self.windows[window_index]
        x = self.store.get_window(
            record_index, wi, rng=rng, random_offset=self.random_offset
        )
        if self.aug is not None:
            x = self.aug(x, rng)
        return self.representation(x)

    def __getitem__(self, index):
        i, j, same = self.pairs[index]
        rng = np.random.RandomState(self.rng.randint(0, 2**31 - 1))
        return (
            self._prepare(i, rng),
            self._prepare(j, rng),
            self.store.pid_to_class[self.windows[i][2]],
            self.store.pid_to_class[self.windows[j][2]],
            float(same),
        )


@torch.no_grad()
def mine_semi_hard(
    model,
    dataset,
    rank_lo=50,
    rank_hi=500,
    batch=128,
    device=None,
):
    """Return a band of high-similarity different-identity pairs."""
    device = device or next(model.parameters()).device
    model.eval()
    xs = [dataset._prepare(i, np.random.RandomState(0)) for i in range(len(dataset.windows))]
    pids = np.array([w[2] for w in dataset.windows])

    embeddings = []
    for k in range(0, len(xs), batch):
        embeddings.append(model.embed(torch.stack(xs[k : k + batch]).to(device)).cpu())
    z = F.normalize(torch.cat(embeddings), dim=1)
    similarity = (z @ z.T).numpy()
    similarity = np.where(pids[:, None] != pids[None, :], similarity, -np.inf)

    upper = np.triu_indices(len(pids), k=1)
    scores = similarity[upper]
    finite = np.isfinite(scores)
    order = np.argsort(-scores[finite])
    band = order[rank_lo:rank_hi]
    if len(band) == 0:
        band = order[: max(1, rank_hi)]
    return list(zip(upper[0][finite][band].tolist(), upper[1][finite][band].tolist()))
