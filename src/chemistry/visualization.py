"""Molecule rendering helpers."""

from __future__ import annotations

from PIL.Image import Image
from rdkit import Chem
from rdkit.Chem import Draw


def mol_to_image(mol: Chem.Mol, size: tuple[int, int] = (320, 240)) -> Image:
    """Render an RDKit molecule as a PIL image."""
    display_mol = Chem.Mol(mol)
    Chem.rdDepictor.Compute2DCoords(display_mol)
    return Draw.MolToImage(display_mol, size=size)
