"""Multi-objective candidate scoring."""

from __future__ import annotations

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from src.predictors.base import Prediction


def _range_desirability(value: float, low: float, high: float, softness: float = 1.0) -> float:
    """Return 1 inside a range and decay linearly outside it."""
    if low <= value <= high:
        return 1.0
    distance = low - value if value < low else value - high
    return max(0.0, 1.0 - distance / max(softness, 1e-6))


def _solubility_desirability(logs_value: float) -> float:
    """Map heuristic logS-like solubility to 0-1 desirability."""
    return max(0.0, min(1.0, (logs_value + 6.0) / 6.0))


def tanimoto_similarity(original: Chem.Mol, candidate: Chem.Mol) -> float:
    """Calculate Morgan fingerprint Tanimoto similarity."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp_a = generator.GetFingerprint(original)
    fp_b = generator.GetFingerprint(candidate)
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def score_candidate(
    original: Chem.Mol,
    candidate: Chem.Mol,
    descriptors: dict[str, float],
    predictions: dict[str, Prediction],
) -> tuple[float, float]:
    """Calculate the MVP multi-objective score and similarity."""
    sol = float(predictions["Solubility Expert"].value)
    logp = float(predictions["Lipophilicity Expert"].value)
    herg = float(predictions["hERG Expert"].value)
    ames = float(predictions["AMES Expert"].value)
    similarity = tanimoto_similarity(original, candidate)

    score = (
        0.20 * _solubility_desirability(sol)
        + 0.20 * _range_desirability(logp, 1.0, 3.0, softness=2.0)
        + 0.20 * float(descriptors["qed"])
        + 0.15 * similarity
        + 0.15 * (1.0 - herg)
        + 0.10 * (1.0 - ames)
    )
    return float(score), similarity
