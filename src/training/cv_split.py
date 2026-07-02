"""Cross-validation split helpers for ADMET datasets."""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import KFold, StratifiedKFold


def murcko_scaffold(smiles: str) -> str:
    """Return a Murcko scaffold SMILES string, or the molecule SMILES fallback."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return str(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return Chem.MolToSmiles(mol, canonical=True)
    return Chem.MolToSmiles(scaffold, canonical=True)


def assign_scaffold_folds(frame: pd.DataFrame, num_folds: int = 10, seed: int = 42) -> list[int]:
    """Assign scaffold groups to folds while keeping each scaffold in one fold."""
    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, smiles in enumerate(frame["smiles"].astype(str).tolist()):
        scaffold_to_indices[murcko_scaffold(smiles)].append(idx)

    rng = np.random.default_rng(seed)
    groups = list(scaffold_to_indices.values())
    groups.sort(key=len, reverse=True)
    same_size_start = 0
    while same_size_start < len(groups):
        same_size_end = same_size_start + 1
        while same_size_end < len(groups) and len(groups[same_size_end]) == len(groups[same_size_start]):
            same_size_end += 1
        block = groups[same_size_start:same_size_end]
        rng.shuffle(block)
        groups[same_size_start:same_size_end] = block
        same_size_start = same_size_end

    fold_sizes = [0 for _ in range(num_folds)]
    assignments = [-1 for _ in range(len(frame))]
    for group in groups:
        fold = int(np.argmin(fold_sizes))
        for idx in group:
            assignments[idx] = fold
        fold_sizes[fold] += len(group)
    return assignments


def assign_random_folds(frame: pd.DataFrame, num_folds: int = 10, seed: int = 42) -> list[int]:
    """Assign random KFold folds."""
    assignments = [-1 for _ in range(len(frame))]
    splitter = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    for fold, (_, test_idx) in enumerate(splitter.split(frame)):
        for idx in test_idx:
            assignments[int(idx)] = fold
    return assignments


def assign_stratified_folds(frame: pd.DataFrame, num_folds: int = 10, seed: int = 42) -> list[int]:
    """Assign stratified folds for binary classification labels."""
    assignments = [-1 for _ in range(len(frame))]
    labels = frame["y"].astype(int).to_numpy()
    splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    for fold, (_, test_idx) in enumerate(splitter.split(frame, labels)):
        for idx in test_idx:
            assignments[int(idx)] = fold
    return assignments


def make_cv_folds(
    frame: pd.DataFrame,
    num_folds: int = 10,
    split_type: str = "scaffold",
    seed: int = 42,
) -> list[dict[str, pd.DataFrame]]:
    """Return train/valid/test DataFrames for every fold."""
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")
    work = frame[["smiles", "y"]].dropna().reset_index(drop=True).copy()
    if len(work) < num_folds:
        raise ValueError("Number of rows must be >= num_folds.")

    if split_type == "scaffold":
        assignments = assign_scaffold_folds(work, num_folds, seed)
    elif split_type == "random":
        assignments = assign_random_folds(work, num_folds, seed)
    elif split_type == "stratified":
        try:
            assignments = assign_stratified_folds(work, num_folds, seed)
        except ValueError:
            assignments = assign_random_folds(work, num_folds, seed)
    else:
        raise ValueError(f"Unsupported split_type: {split_type}")

    work["fold"] = assignments
    folds: list[dict[str, pd.DataFrame]] = []
    for fold in range(num_folds):
        valid_fold = (fold + 1) % num_folds
        test = work[work["fold"] == fold][["smiles", "y"]].reset_index(drop=True)
        valid = work[work["fold"] == valid_fold][["smiles", "y"]].reset_index(drop=True)
        train = work[~work["fold"].isin([fold, valid_fold])][["smiles", "y"]].reset_index(drop=True)
        folds.append({"train": train, "valid": valid, "test": test})
    return folds


def parse_args() -> argparse.Namespace:
    """Parse CLI args for split smoke checks."""
    parser = argparse.ArgumentParser(description="Create CV folds for a standardized CSV.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--num-folds", type=int, default=10)
    parser.add_argument("--split-type", choices=["scaffold", "random", "stratified"], default="scaffold")
    return parser.parse_args()


def main() -> None:
    """Print fold sizes for a CSV containing smiles,y columns."""
    args = parse_args()
    frame = pd.read_csv(args.csv)
    folds = make_cv_folds(frame, args.num_folds, args.split_type)
    for idx, split in enumerate(folds):
        print(idx, len(split["train"]), len(split["valid"]), len(split["test"]))


if __name__ == "__main__":
    main()
