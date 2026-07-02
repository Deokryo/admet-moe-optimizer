"""Common predictor interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rdkit import Chem


@dataclass(frozen=True)
class Prediction:
    """Single endpoint prediction."""

    endpoint: str
    task: str
    value: float
    unit: str | None
    confidence: float
    interpretation: str


class Predictor(Protocol):
    """Protocol for replaceable ADMET predictors."""

    name: str
    task: str

    def predict(self, mol: Chem.Mol, descriptors: dict[str, float]) -> Prediction:
        """Predict an endpoint for an RDKit molecule."""
        ...
