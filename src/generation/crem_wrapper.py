"""Optional CReM candidate generation wrapper."""

from __future__ import annotations

from src.generation.rule_based import GeneratedCandidate
from src.utils.smiles import canonicalize_smiles


def generate_with_crem(smiles: str, max_candidates: int = 20) -> list[GeneratedCandidate]:
    """Generate candidates with CReM if the package and data are available.

    CReM usually requires fragment databases. For MVP portability this wrapper
    tries a conservative import and returns an empty list when unavailable.
    """
    try:
        from crem.crem import mutate_mol  # type: ignore
        from rdkit import Chem
    except Exception:
        return []

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    candidates: list[GeneratedCandidate] = []
    try:
        generated = mutate_mol(mol, db_name=None, max_size=3, return_mol=True)
    except Exception:
        return []

    original = canonicalize_smiles(smiles)
    for item in generated:
        try:
            candidate_smiles = canonicalize_smiles(Chem.MolToSmiles(item))
        except Exception:
            continue
        if candidate_smiles and candidate_smiles != original:
            candidates.append(GeneratedCandidate(candidate_smiles, "CReM mutation 후보"))
        if len(candidates) >= max_candidates:
            break
    return candidates

