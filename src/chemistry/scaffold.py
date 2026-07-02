"""Murcko scaffold helpers."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_murcko_scaffold(mol: Chem.Mol) -> Chem.Mol | None:
    """Return the Murcko scaffold molecule if available."""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return None
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return None
    return scaffold


def get_scaffold_atom_indices(mol: Chem.Mol) -> set[int]:
    """Return atom indices in the original molecule that match the Murcko scaffold."""
    scaffold = get_murcko_scaffold(mol)
    if scaffold is None:
        return set()
    matches = mol.GetSubstructMatches(scaffold)
    if not matches:
        return set()
    return set(matches[0])
