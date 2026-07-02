"""SMILES utility functions."""

from __future__ import annotations

from rdkit import Chem


def canonicalize_smiles(smiles: str) -> str:
    """Return canonical RDKit SMILES or an empty string if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
