"""TDC dataset loading and scaffold split helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.errors import ParserError
from requests.exceptions import ConnectionError as RequestsConnectionError


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


def _remove_cached_dataset(dataset_name: str, data_dir: str) -> None:
    """Remove possibly corrupted local TDC cache files for a dataset."""
    path = Path(data_dir)
    stem = dataset_name.lower()
    for suffix in (".tab", ".csv", ".txt", ".pkl"):
        candidate = path / f"{stem}{suffix}"
        if candidate.exists():
            candidate.unlink()


def _friendly_load_error(dataset_name: str, data_dir: str, exc: Exception) -> Exception:
    """Convert common TDC loading failures into actionable messages."""
    message = str(exc)
    if isinstance(exc, RequestsConnectionError) or "NameResolutionError" in message or "getaddrinfo failed" in message:
        return RuntimeError(
            f"TDC dataset '{dataset_name}' 다운로드에 실패했습니다. "
            "dataverse.harvard.edu DNS/네트워크 연결을 확인한 뒤 다시 실행하세요. "
            f"이미 다운로드된 파일이 있다면 --tdc-data-dir {data_dir} 경로를 확인하세요."
        )
    if isinstance(exc, ParserError) or "EOF inside string" in message or "Error tokenizing data" in message:
        return RuntimeError(
            f"TDC dataset '{dataset_name}'의 로컬 캐시 파일이 깨진 것으로 보입니다. "
            f"`data/{dataset_name.lower()}.tab` 같은 부분 다운로드 파일을 삭제하거나 "
            "`--force-redownload` 옵션으로 다시 다운로드하세요."
        )
    return ValueError(
        f"Could not load TDC dataset '{dataset_name}'. "
        "Dataset name, local cache, and PyTDC version을 확인하세요."
    )


def load_tdc_dataset(dataset_name: str, data_dir: str = "./data", force_redownload: bool = False) -> DatasetSplit:
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

    if force_redownload:
        _remove_cached_dataset(dataset_name, data_dir)

    try:
        dataset = dataset_cls(name=dataset_name, path=data_dir)
    except Exception as exc:
        raise _friendly_load_error(dataset_name, data_dir, exc) from exc

    try:
        split = dataset.get_split(method="scaffold")
    except Exception as exc:
        raise RuntimeError(f"TDC scaffold split failed for '{dataset_name}'.") from exc

    return DatasetSplit(
        train=_standardize_frame(split["train"]),
        valid=_standardize_frame(split["valid"]),
        test=_standardize_frame(split["test"]),
    )
