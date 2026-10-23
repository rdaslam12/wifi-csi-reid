"""Train a WiFi-CSI ReID reference model on Wi-PER81."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from wifi_reid.augment import SubCarrierAug, VisionAug
from wifi_reid.data import Representation, WindowStore, build_cache
from wifi_reid.model import JointReIDLoss, SubCarrierReID
from wifi_reid.pairs import DynamicPairDataset
from wifi_reid.training import run_epoch, set_seed

TRAIN_WINDOWS = list(range(7))
VAL_WINDOWS = [7, 8, 9]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, help="Path to the extracted Wi-PER81 dataset")
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--csv-tree", default="raw_amplitudes")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--pairs", type=int, default=8000)
    parser.add_argument("--val-pairs", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--aug", choices=["none", "subcarrier", "vision"], default="subcarrier")
    parser.add_argument("--aug-strength", type=float, default=1.0)
    parser.add_argument("--dual-axis", action="store_true")
    parser.add_argument("--head", choices=["flatten", "pooled"], default="flatten")
    parser.add_argument("--output", default="checkpoints/model.pt")
    return parser.parse_args()


def fit_representation(store):
    windows = np.stack(
        [store.get_window(record_i, window_i) for record_i in range(store.n_captures()) for window_i in range(10)]
    )
    return Representation().fit(windows)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index = build_cache(args.raw_dir, csv_tree=args.csv_tree, cache_dir=args.cache_dir)
    pids = sorted(index.pid.unique())
    train_store = WindowStore(index, pids, ["A"])
    representation = fit_representation(train_store)

    if args.aug == "subcarrier":
        aug = SubCarrierAug(args.aug_strength)
    elif args.aug == "vision":
        aug = VisionAug(args.aug_strength)
    else:
        aug = None

    train_ds = DynamicPairDataset(train_store, representation, args.pairs, TRAIN_WINDOWS, aug=aug, seed=args.seed, random_offset=True)
    val_ds = DynamicPairDataset(train_store, representation, args.val_pairs, VAL_WINDOWS, aug=None, seed=args.seed + 100, random_offset=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SubCarrierReID(input_shape=(1, 100, 52), n_classes=len(pids), dual_axis=args.dual_axis, head=args.head).to(device)
    criterion = JointReIDLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        if epoch > 1:
            train_ds.resample()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        history.append(dict(epoch=epoch, train_loss=train_loss, train_pair_acc=train_acc, val_loss=val_loss, val_pair_acc=val_acc))
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        print(f"epoch {epoch:03d}/{args.epochs} | train {train_loss:.4f}/{train_acc:.3f} | val {val_loss:.4f}/{val_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "n_classes": len(pids), "pids": pids, "representation_mean": representation.mean, "representation_std": representation.std, "model_config": {"dual_axis": args.dual_axis, "head": args.head, "embed": 128}, "training_config": vars(args), "history": history}, output)
    history_path = output.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2))
    print(f"saved checkpoint: {output}")
    print(f"saved history:    {history_path}")


if __name__ == "__main__":
    main()
