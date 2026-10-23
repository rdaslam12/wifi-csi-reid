"""Data loading, sanitisation, windowing, and normalization for Wi-PER81."""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

N_PACKETS = 1000
N_SUBCARRIERS = 52
WINDOW_SIZE = 100
N_WINDOWS = 10

FILENAME_RE = re.compile(
    r"(?P<pid>\d{4,5})\s*_?\s*(?P<gender>[mfMF])\s*_?\s*(?P<session>[ABab])"
)


def csi_to_amplitude(array: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(array):
        return np.abs(array).astype(np.float32)
    return array.astype(np.float32)


def load_csv(path: str | os.PathLike[str]) -> np.ndarray:
    df = pd.read_csv(path)
    timestamp_cols = [
        c for c in df.columns if "time" in str(c).lower() or "stamp" in str(c).lower()
    ]
    if timestamp_cols:
        df = df.drop(columns=timestamp_cols)
    elif df.shape[1] == N_SUBCARRIERS + 1:
        df = df.iloc[:, :N_SUBCARRIERS]

    arr = df.to_numpy()
    if arr.dtype == object:
        arr = np.vectorize(complex)(arr)
    arr = csi_to_amplitude(arr)

    if arr.shape != (N_PACKETS, N_SUBCARRIERS):
        raise ValueError(f"{Path(path).name}: expected {(N_PACKETS, N_SUBCARRIERS)}, got {arr.shape}")
    return arr


def iqr_sanitise(window: np.ndarray) -> np.ndarray:
    out = window.astype(np.float32).copy()
    q1 = np.percentile(out, 25, axis=0)
    q3 = np.percentile(out, 75, axis=0)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    bad = (out < lower) | (out > upper)

    for subcarrier in range(out.shape[1]):
        col = out[:, subcarrier]
        mask = bad[:, subcarrier]
        if not mask.any():
            continue
        good = np.where(~mask)[0]
        if good.size == 0:
            continue
        last = col[good[0]]
        for packet in range(out.shape[0]):
            if mask[packet]:
                col[packet] = last
            else:
                last = col[packet]
        out[:, subcarrier] = col
    return out


def build_cache(
    raw_dir: str | os.PathLike[str],
    csv_tree: str = "raw_amplitudes",
    cache_dir: str | os.PathLike[str] = "./cache",
) -> pd.DataFrame:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    all_csv = sorted(glob.glob(os.path.join(str(raw_dir), "**", "*.csv"), recursive=True))
    files = [f for f in all_csv if csv_tree in f.replace("\\", "/").split("/")]
    if not files:
        files = [f for f in all_csv if csv_tree in f]
    if not files:
        raise FileNotFoundError(f"No CSVs for {csv_tree!r} under {str(raw_dir)!r}")

    rows: list[dict[str, object]] = []
    for file_path in files:
        match = FILENAME_RE.search(Path(file_path).name)
        if match is None:
            continue
        pid = int(match.group("pid"))
        gender = match.group("gender").lower()
        session = match.group("session").upper()
        amplitude = load_csv(file_path)
        out = cache_path / f"{csv_tree}_{pid:05d}_{gender}_{session}.npy"
        np.save(out, amplitude)
        rows.append(
            {"pid": pid, "gender": gender, "session": session, "path": str(out), "src": file_path}
        )

    index = pd.DataFrame(rows).sort_values(["pid", "session"]).reset_index(drop=True)
    duplicates = index.groupby(["pid", "session"]).size()
    if (duplicates > 1).any():
        raise ValueError("Duplicate (pid, session) captures detected; select exactly one CSV tree.")
    return index


class WindowStore:
    def __init__(self, index: pd.DataFrame, pids: Sequence[int], sessions: Sequence[str]):
        selected = index[index.pid.isin(pids) & index.session.isin(sessions)]
        self.records = [
            (int(row.pid), row.session, np.load(row.path)) for _, row in selected.iterrows()
        ]
        self.pids = sorted({pid for pid, _, _ in self.records})
        self.pid_to_class = {pid: i for i, pid in enumerate(self.pids)}

    def n_captures(self) -> int:
        return len(self.records)

    def get_window(
        self,
        record_index: int,
        window_index: int,
        rng: np.random.RandomState | None = None,
        random_offset: bool = False,
    ) -> np.ndarray:
        _, _, amplitude = self.records[record_index]
        if random_offset and rng is not None:
            start = rng.randint(0, amplitude.shape[0] - WINDOW_SIZE + 1)
        else:
            start = window_index * WINDOW_SIZE
        return iqr_sanitise(amplitude[start : start + WINDOW_SIZE])

    def enumerate_windows(self) -> list[tuple[int, int, int, str]]:
        return [
            (record_index, window_index, pid, session)
            for record_index, (pid, session, _) in enumerate(self.records)
            for window_index in range(N_WINDOWS)
        ]


def select_windows(store: WindowStore, window_indices: Iterable[int]):
    keep = set(window_indices)
    return [w for w in store.enumerate_windows() if w[1] in keep]


def probe_gallery_masks(store: WindowStore, test_windows: Sequence[int]):
    selected = select_windows(store, test_windows)
    windows = np.array([item[1] for item in selected])
    half = len(test_windows) // 2
    return np.isin(windows, test_windows[:half]), np.isin(windows, test_windows[half:])


@dataclass
class Representation:
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, windows: np.ndarray) -> "Representation":
        flat = windows.reshape(-1, windows.shape[-1])
        self.mean = flat.mean(0).astype(np.float32)
        self.std = (flat.std(0) + 1e-6).astype(np.float32)
        return self

    def __call__(self, window: np.ndarray) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Representation.fit(...) must be called before normalization.")
        normalized = ((window - self.mean) / self.std).astype(np.float32)
        return torch.from_numpy(normalized).unsqueeze(0)
