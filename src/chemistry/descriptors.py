"""RDKit descriptor calculation."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors


def estimate_sa_score_placeholder(mol: Chem.Mol) -> float:
    """Estimate synthetic accessibility on a rough 1-10 scale.

    The canonical RDKit SA score script is not bundled with RDKit. This MVP uses
    a transparent placeholder based on size, rings, hetero atoms, and stereocenters.
    Lower values are easier synthetic-accessibility estimates.
    """
    atoms = mol.GetNumAtoms()
    rings = rdMolDescriptors.CalcNumRings(mol)
    hetero = rdMolDescriptors.CalcNumHeteroatoms(mol)
    stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    score = 1.0 + 0.035 * atoms + 0.25 * rings + 0.08 * hetero + 0.25 * stereo
    return float(max(1.0, min(10.0, score)))


def calculate_descriptors(mol: Chem.Mol) -> dict[str, float]:
    """Calculate core molecular descriptors for MVP scoring and display."""
    return {
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hbd": float(Lipinski.NumHDonors(mol)),
        "hba": float(Lipinski.NumHAcceptors(mol)),
        "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
        "qed": float(QED.qed(mol)),
        "sa_score": estimate_sa_score_placeholder(mol),
    }
