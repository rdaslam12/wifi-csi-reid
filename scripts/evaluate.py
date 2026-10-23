"""Evaluate a saved model on the Wi-PER81 closed-set session-B protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wifi_reid.data import Representation, WindowStore, build_cache, probe_gallery_masks, select_windows
from wifi_reid.metrics import bootstrap_rank1_ci, cmc_and_map, macro_f1
from wifi_reid.model import SubCarrierReID

TEST_WINDOWS = [4, 5, 6, 7, 8, 9]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--csv-tree", default="raw_amplitudes")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output", default=None, help="Optional JSON metrics path")
    return parser.parse_args()


@torch.no_grad()
def embed_windows(model, store, representation, window_indices, batch_size, device):
    selected = select_windows(store, window_indices)
    tensors = [representation(store.get_window(record_i, window_i)) for record_i, window_i, _, _ in selected]
    embeddings = []
    for start in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[start : start + batch_size]).to(device)
        embeddings.append(model.embed(batch).cpu())
    z = F.normalize(torch.cat(embeddings), dim=1)
    pids = np.array([item[2] for item in selected])
    return z, pids


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    index = build_cache(args.raw_dir, csv_tree=args.csv_tree, cache_dir=args.cache_dir)
    pids = blob.get("pids", sorted(index.pid.unique()))
    eval_store = WindowStore(index, pids, ["B"])

    representation = Representation(
        mean=np.asarray(blob["representation_mean"], dtype=np.float32),
        std=np.asarray(blob["representation_std"], dtype=np.float32),
    )
    cfg = blob.get("model_config", {})
    model = SubCarrierReID(
        input_shape=(1, 100, 52),
        n_classes=int(blob.get("n_classes", len(pids))),
        embed=int(cfg.get("embed", 128)),
        dual_axis=bool(cfg.get("dual_axis", True)),
        head=cfg.get("head", "flatten"),
    ).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()

    probe_mask, gallery_mask = probe_gallery_masks(eval_store, TEST_WINDOWS)
    z, ids = embed_windows(model, eval_store, representation, TEST_WINDOWS, args.batch_size, device)
    similarity = (z[probe_mask] @ z[gallery_mask].T).numpy()
    probe_ids = ids[probe_mask]
    gallery_ids = ids[gallery_mask]

    metrics = cmc_and_map(similarity, probe_ids, gallery_ids, ranks=(1, 5, 10))
    metrics["macro_f1"] = macro_f1(similarity, probe_ids, gallery_ids)
    ci_low, ci_high = bootstrap_rank1_ci(similarity, probe_ids, gallery_ids, n_boot=args.bootstrap, seed=0)
    metrics["rank_1_ci_95"] = [ci_low, ci_high]

    print(json.dumps(metrics, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2))
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
