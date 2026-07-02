"""Optional CReM candidate generation wrapper."""

from __future__ import annotations

from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from src.chemistry.scaffold import get_murcko_scaffold
from src.generation.rule_based import GeneratedCandidate
from src.utils.smiles import canonicalize_smiles


def is_crem_available() -> tuple[bool, str]:
    """Return whether CReM can be imported and a human-readable status."""
    try:
        from crem.crem import mutate_mol  # noqa: F401
    except Exception as exc:
        return False, f"CReM import failed: {exc}"
    return True, "CReM import available"


def _similarity(original: Chem.Mol, candidate: Chem.Mol) -> float:
    """Return Morgan fingerprint Tanimoto similarity."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    original_fp = generator.GetFingerprint(original)
    candidate_fp = generator.GetFingerprint(candidate)
    return float(DataStructs.TanimotoSimilarity(original_fp, candidate_fp))


def _passes_scaffold_filter(original: Chem.Mol, candidate: Chem.Mol, preserve_scaffold: bool) -> bool:
    """Return whether the candidate preserves the original Murcko scaffold."""
    if not preserve_scaffold:
        return True
    scaffold = get_murcko_scaffold(original)
    if scaffold is None:
        return True
    return bool(candidate.HasSubstructMatch(scaffold))


def generate_with_crem(
    smiles: str,
    max_candidates: int = 20,
    db_path: str | None = None,
    target_atom_indices: list[int] | None = None,
    min_similarity: float = 0.3,
    preserve_scaffold: bool = True,
) -> tuple[list[GeneratedCandidate], str, list[str]]:
    """Generate candidates with CReM and return candidates, status, and warnings."""
    warnings: list[str] = []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], "Invalid SMILES for CReM generation", warnings

    available, availability_status = is_crem_available()
    if not available:
        return [], availability_status, warnings

    normalized_db_path = (db_path or "").strip()
    if not normalized_db_path:
        return [], "CReM DB path is not provided", warnings
    db_file = Path(normalized_db_path)
    if not db_file.exists():
        return [], f"CReM DB path does not exist: {normalized_db_path}", warnings

    try:
        from crem.crem import mutate_mol
    except Exception as exc:
        return [], f"CReM import failed: {exc}", warnings

    try:
        # Current wrapper does not impose atom-index constraints directly because
        # CReM APIs differ by version. Target atoms are recorded in provenance.
        generated = mutate_mol(mol, db_name=str(db_file), max_size=3, return_mol=True)
    except Exception as exc:
        return [], f"CReM mutate_mol failed: {exc}", warnings

    original = canonicalize_smiles(smiles)
    unique: dict[str, GeneratedCandidate] = {}
    rejected_similarity = 0
    rejected_scaffold = 0
    rejected_invalid = 0

    for item in generated:
        try:
            candidate_mol = item if isinstance(item, Chem.Mol) else Chem.MolFromSmiles(str(item))
            if candidate_mol is None:
                rejected_invalid += 1
                continue
            Chem.SanitizeMol(candidate_mol)
            candidate_smiles = canonicalize_smiles(Chem.MolToSmiles(candidate_mol))
        except Exception:
            rejected_invalid += 1
            continue

        if not candidate_smiles or candidate_smiles == original:
            continue
        if _similarity(mol, candidate_mol) < min_similarity:
            rejected_similarity += 1
            continue
        if not _passes_scaffold_filter(mol, candidate_mol, preserve_scaffold):
            rejected_scaffold += 1
            continue

        target_note = f"target-informed generation; requested atoms={target_atom_indices}" if target_atom_indices else None
        unique.setdefault(
            candidate_smiles,
            GeneratedCandidate(
                smiles=candidate_smiles,
                note=target_note or "CReM mutation 후보",
                source="crem",
                generation_method="crem_mutate_mol",
                edited_region="target_informed" if target_atom_indices else "crem_mutation",
                target_atoms=target_atom_indices,
            ),
        )
        if len(unique) >= max_candidates:
            break

    if rejected_invalid:
        warnings.append(f"CReM invalid/sanitization rejected candidates: {rejected_invalid}")
    if rejected_similarity:
        warnings.append(f"CReM candidates below similarity threshold: {rejected_similarity}")
    if rejected_scaffold:
        warnings.append(f"CReM candidates rejected by scaffold preservation: {rejected_scaffold}")

    candidates = list(unique.values())[:max_candidates]
    status = f"CReM generated {len(candidates)} candidates using mutate_mol"
    return candidates, status, warnings
