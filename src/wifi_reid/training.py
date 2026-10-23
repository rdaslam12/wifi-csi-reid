"""Training helpers for the modular reference implementation."""

from __future__ import annotations

import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    correct = 0
    n = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x_i, x_j, class_i, class_j, same in loader:
            x_i, x_j = x_i.to(device), x_j.to(device)
            class_i = class_i.long().to(device)
            class_j = class_j.long().to(device)
            same = same.float().to(device)
            if train:
                optimizer.zero_grad()

            z_i, z_j, logits_i, logits_j = model(x_i, x_j)
            loss, _ = criterion(z_i, z_j, logits_i, logits_j, class_i, class_j, same)
            if train:
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                prediction = (F.cosine_similarity(z_i, z_j, dim=1) > 0.5).float()
                correct += (prediction == same).sum().item()
            total_loss += loss.item() * x_i.size(0)
            n += x_i.size(0)

    return total_loss / n, correct / n


def train_with_validation(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    best_loss = float("inf")
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)
