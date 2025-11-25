from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import pandas as pd


def find_project_root(start: Path | None = None) -> Path:
    """
    Looks for folder where both src and README are. Utility to find project root.
    """
    if start is None:
        start = Path.cwd()

    candidates = [start] + list(start.parents)
    for p in candidates:
        if (p / "src").exists() and (p / "README.md").exists():
            return p
        if (p / "src").exists():  # fallback
            return p
    return Path.cwd()


def find_dataset_dir(project_root: Path | None = None) -> Path:
    """
    Finds dataset folder (the one that contains both train/ and test/).
    Assumes it exists at project root.
    """
    if project_root is None:
        project_root = find_project_root()

    # correct file name
    direct = project_root / "g2net-gravitational-wave-detection"
    # must contain both train/ and test/
    if (direct / "train").exists() and (direct / "test").exists():
        return direct

    # fallback
    for p in project_root.iterdir():
        if p.is_dir() and (p / "train").exists() and (p / "test").exists():
            return p

    raise FileNotFoundError(
        "Did not find dataset folder. Looking for something like <root>/g2net-gravitational-wave-detection containing both train/ and test/"
    )


def load_labels(dataset_dir: Path | None = None) -> pd.DataFrame:
    """
    Reads labels file. For this project, training_labels.csv.
    Returns DataFrame with columns: id, target.
    """
    if dataset_dir is None:
        dataset_dir = find_dataset_dir()

    labels_path = dataset_dir / "training_labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    return pd.read_csv(labels_path)


def sample_path(sample_id: str, split_dir: Path) -> Path:
    """
    Returns .npy sample path. 
    Supports:
        - nested: split/a/b/c/<id>.npy
        - flat:   split/<id>.npy
    """
    # nested
    p = split_dir / sample_id[0] / sample_id[1] / sample_id[2] / f"{sample_id}.npy"
    if p.exists():
        return p

    # flat fallback
    p2 = split_dir / f"{sample_id}.npy"
    return p2


def load_sample(sample_id: str, split: str = "train", dataset_dir: Path | None = None) -> np.ndarray:
    """
    Loads a sample .npy from train/ or test/.
    Works for nested folder structure (a/b/c/id.npy) and flat structure (id.npy).
    """
    if dataset_dir is None:
        dataset_dir = find_dataset_dir()

    split_dir = dataset_dir / split
    p = sample_path(sample_id, split_dir)

    if not p.exists():
        raise FileNotFoundError(f"Sample not found for id={sample_id}. Tried: {p}")

    return np.load(p)  # (3, 4096)
