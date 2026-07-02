"""TDC dataset loading and scaffold split helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


ADME_DATASETS = {"Solubility_AqSolDB", "Lipophilicity_AstraZeneca", "BBB_Martins"}
TOX_DATASETS = {"hERG_Karim", "AMES"}


@dataclass(frozen=True)
class DatasetSplit:
    """Standardized train/valid/test split."""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def _standardize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename common TDC columns to smiles/y and drop invalid labels."""
    rename_map: dict[str, str] = {}
    if "Drug" in frame.columns:
        rename_map["Drug"] = "smiles"
    if "SMILES" in frame.columns:
        rename_map["SMILES"] = "smiles"
    if "Y" in frame.columns:
        rename_map["Y"] = "y"
    if "Label" in frame.columns:
        rename_map["Label"] = "y"
    standardized = frame.rename(columns=rename_map).copy()
    if "smiles" not in standardized.columns or "y" not in standardized.columns:
        raise ValueError(f"TDC split must contain SMILES/label columns. Columns found: {list(frame.columns)}")
    standardized = standardized[["smiles", "y"]].dropna()
    standardized["smiles"] = standardized["smiles"].astype(str)
    standardized["y"] = standardized["y"].astype(float)
    return standardized.reset_index(drop=True)


def load_tdc_dataset(dataset_name: str) -> DatasetSplit:
    """Load a TDC ADME/Tox dataset and return scaffold splits.

    Raises a friendly error when the dataset name is not available in the local
    PyTDC installation. Some TDC versions may use slightly different aliases.
    """
    try:
        from tdc.single_pred import ADME, Tox
    except Exception as exc:
        raise ImportError("PyTDC is required for training. Install it with `pip install PyTDC`.") from exc

    if dataset_name in ADME_DATASETS:
        dataset_cls = ADME
    elif dataset_name in TOX_DATASETS:
        dataset_cls = Tox
    else:
        supported = sorted(ADME_DATASETS | TOX_DATASETS)
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported datasets: {supported}")

    try:
        dataset = dataset_cls(name=dataset_name)
    except Exception as exc:
        raise ValueError(
            f"Could not load TDC dataset '{dataset_name}'. Check the exact dataset name for your PyTDC version."
        ) from exc

    try:
        split = dataset.get_split(method="scaffold")
    except Exception as exc:
        raise RuntimeError(f"TDC scaffold split failed for '{dataset_name}'.") from exc

    return DatasetSplit(
        train=_standardize_frame(split["train"]),
        valid=_standardize_frame(split["valid"]),
        test=_standardize_frame(split["test"]),
    )

