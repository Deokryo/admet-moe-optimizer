"""SMILES validation helpers."""

from __future__ import annotations

from rdkit import Chem


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    """Parse a SMILES string into an RDKit molecule."""
    if not smiles or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip(), sanitize=True)
    except Exception:
        return None
    return mol


def validate_smiles(smiles: str) -> tuple[bool, str | None]:
    """Validate a SMILES string and return an error message if invalid."""
    if not smiles or not smiles.strip():
        return False, "SMILES is empty."
    mol = mol_from_smiles(smiles)
    if mol is None:
        return False, "SMILES could not be parsed or sanitized by RDKit."
    if mol.GetNumAtoms() == 0:
        return False, "SMILES contains no atoms."
    return True, None
