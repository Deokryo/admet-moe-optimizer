"""Molecule rendering helpers."""

from __future__ import annotations

from io import BytesIO

from PIL import Image as PILImage
from PIL.Image import Image
from rdkit import Chem
from rdkit.Chem import Draw


def mol_to_image(mol: Chem.Mol, size: tuple[int, int] = (320, 240)) -> Image:
    """Render an RDKit molecule as a PIL image."""
    display_mol = Chem.Mol(mol)
    Chem.rdDepictor.Compute2DCoords(display_mol)
    return Draw.MolToImage(display_mol, size=size)


def endpoint_color(endpoint: str) -> tuple[float, float, float]:
    """Return the default highlight color for an ADMET endpoint."""
    colors = {
        "solubility": (0.2, 0.72, 0.42),
        "lipophilicity": (0.95, 0.34, 0.18),
        "bbb": (0.18, 0.45, 0.95),
        "herg": (0.72, 0.2, 0.68),
        "ames": (0.92, 0.68, 0.18),
    }
    return colors.get(endpoint.lower(), (1.0, 0.4, 0.4))


def draw_saliency_molecule(
    smiles: str,
    atom_scores: dict[int, float],
    top_k: int = 8,
    width: int = 500,
    height: int = 400,
    color: tuple[float, float, float] = (1.0, 0.4, 0.4),
) -> Image | None:
    """Draw a molecule with atom saliency highlights scaled by normalized scores."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or not atom_scores:
            return None

        display_mol = Chem.Mol(mol)
        Chem.rdDepictor.Compute2DCoords(display_mol)

        ranked_atoms = sorted(atom_scores, key=atom_scores.get, reverse=True)[:top_k]
        if not ranked_atoms:
            return None

        max_score = max(float(atom_scores.get(atom_idx, 0.0)) for atom_idx in ranked_atoms)
        if max_score <= 0.0:
            return None

        highlight_atoms = [atom_idx for atom_idx in ranked_atoms if 0 <= atom_idx < display_mol.GetNumAtoms()]
        highlight_colors = {atom_idx: color for atom_idx in highlight_atoms}
        highlight_radii = {
            atom_idx: 0.18 + 0.42 * (float(atom_scores.get(atom_idx, 0.0)) / max_score)
            for atom_idx in highlight_atoms
        }
        if not highlight_atoms:
            return None

        drawer = Draw.MolDraw2DCairo(width, height)
        options = drawer.drawOptions()
        options.useBWAtomPalette()
        drawer.DrawMolecule(
            display_mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=highlight_colors,
            highlightAtomRadii=highlight_radii,
        )
        drawer.FinishDrawing()
        return PILImage.open(BytesIO(drawer.GetDrawingText()))
    except Exception:
        return None
