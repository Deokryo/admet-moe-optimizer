"""Descriptor-based heuristic ADMET predictors for the MVP."""

from __future__ import annotations

import math
from dataclasses import dataclass

from rdkit import Chem

from src.predictors.base import Prediction, Predictor


def _sigmoid(x: float) -> float:
    """Return a numerically stable logistic transform."""
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))


@dataclass
class HeuristicPredictor:
    """Small descriptor-based predictor compatible with future model replacement."""

    name: str
    task: str

    def predict(self, mol: Chem.Mol, descriptors: dict[str, float]) -> Prediction:
        """Predict using hand-built descriptor rules."""
        logp = descriptors["logp"]
        mw = descriptors["molecular_weight"]
        tpsa = descriptors["tpsa"]
        hbd = descriptors["hbd"]
        hba = descriptors["hba"]
        rotb = descriptors["rotatable_bonds"]

        if self.name == "Solubility Expert":
            value = 0.8 - 0.55 * logp - 0.006 * max(mw - 250.0, 0.0) + 0.012 * tpsa - 0.08 * rotb
            return Prediction(self.name, self.task, value, "heuristic logS", 0.55, "값이 높을수록 용해도가 높게 예측됩니다.")

        if self.name == "Lipophilicity Expert":
            return Prediction(self.name, self.task, logp, "LogP", 0.75, "일반적인 MVP 목표 범위는 약 1-3입니다.")

        if self.name == "BBB Expert":
            score = 1.6 * logp - 0.035 * tpsa - 0.45 * hbd - 0.12 * hba + 1.0
            value = _sigmoid(score)
            return Prediction(self.name, self.task, value, "probability", 0.6, "BBB 투과 가능성에 대한 확률형 추정값입니다.")

        if self.name == "hERG Expert":
            aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
            score = 0.85 * logp + 0.018 * mw + 0.12 * aromatic_atoms + 0.45 * hba - 6.8
            value = _sigmoid(score)
            return Prediction(self.name, self.task, value, "risk probability", 0.55, "값이 높을수록 hERG risk가 높게 예측됩니다.")

        if self.name == "AMES Expert":
            nitro = mol.HasSubstructMatch(Chem.MolFromSmarts("[$([NX3](=O)=O),$([NX3+](=O)[O-])]"))
            aniline = mol.HasSubstructMatch(Chem.MolFromSmarts("a[NH2,NH1,NH0]"))
            score = -2.2 + (1.7 if nitro else 0.0) + (1.2 if aniline else 0.0) + 0.2 * max(logp - 3.0, 0.0)
            value = _sigmoid(score)
            return Prediction(self.name, self.task, value, "risk probability", 0.5, "값이 높을수록 AMES risk가 높게 예측됩니다.")

        raise ValueError(f"Unsupported predictor: {self.name}")


def build_dummy_predictors() -> list[Predictor]:
    """Create the five MVP endpoint experts."""
    return [
        HeuristicPredictor("Solubility Expert", "regression"),
        HeuristicPredictor("Lipophilicity Expert", "regression"),
        HeuristicPredictor("BBB Expert", "binary classification"),
        HeuristicPredictor("hERG Expert", "binary classification"),
        HeuristicPredictor("AMES Expert", "binary classification"),
    ]
