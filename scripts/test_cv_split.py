"""Smoke tests for cross-validation split helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.cv_split import make_cv_folds, murcko_scaffold


def test_scaffold_groups_stay_together() -> None:
    """Scaffold split should keep identical Murcko scaffolds in one test fold."""
    frame = pd.DataFrame(
        {
            "smiles": [
                "c1ccccc1Cl",
                "c1ccccc1F",
                "c1ccncc1",
                "c1ccncc1C",
                "CCO",
                "CCN",
                "CC(=O)O",
                "CCOC",
            ],
            "y": [0.1, 0.2, 1.0, 1.1, 0.3, 0.4, 0.5, 0.6],
        }
    )
    folds = make_cv_folds(frame, num_folds=4, split_type="scaffold", seed=7)

    scaffold_to_test_fold: dict[str, int] = {}
    for fold_idx, split in enumerate(folds):
        for smiles in split["test"]["smiles"]:
            scaffold = murcko_scaffold(smiles)
            if scaffold in scaffold_to_test_fold:
                assert scaffold_to_test_fold[scaffold] == fold_idx
            scaffold_to_test_fold[scaffold] = fold_idx

    for split in folds:
        assert set(split) == {"train", "valid", "test"}
        assert len(split["train"]) + len(split["valid"]) + len(split["test"]) == len(frame)


def test_random_split_runs() -> None:
    """Random split should create the requested number of folds."""
    frame = pd.DataFrame({"smiles": ["CCO", "CCN", "CCC", "CCCl", "CCBr", "CCF"], "y": [0, 1, 0, 1, 0, 1]})
    folds = make_cv_folds(frame, num_folds=3, split_type="random", seed=42)
    assert len(folds) == 3
    assert all(not split["test"].empty for split in folds)


if __name__ == "__main__":
    test_scaffold_groups_stay_together()
    test_random_split_runs()
    print("test_cv_split.py: ok")
