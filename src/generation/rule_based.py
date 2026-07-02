"""Rule-based analogue generation fallback."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from src.utils.smiles import canonicalize_smiles


@dataclass(frozen=True)
class GeneratedCandidate:
    """Generated candidate molecule and provenance note."""

    smiles: str
    note: str


def _replace_atoms(mol: Chem.Mol, atomic_num_from: int, atomic_num_to: int, note: str) -> list[GeneratedCandidate]:
    """Generate candidates by replacing one atom type with another."""
    candidates: list[GeneratedCandidate] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != atomic_num_from:
            continue
        rw_mol = Chem.RWMol(mol)
        rw_mol.GetAtomWithIdx(atom.GetIdx()).SetAtomicNum(atomic_num_to)
        try:
            Chem.SanitizeMol(rw_mol)
        except Exception:
            continue
        smiles = canonicalize_smiles(Chem.MolToSmiles(rw_mol))
        candidates.append(GeneratedCandidate(smiles=smiles, note=note))
    return candidates


def _methyl_to_hydroxymethyl(mol: Chem.Mol) -> list[GeneratedCandidate]:
    """Convert terminal methyl groups into hydroxymethyl-like substituents."""
    candidates: list[GeneratedCandidate] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.GetDegree() != 1:
            continue
        if atom.GetTotalNumHs() < 2:
            continue
        rw_mol = Chem.RWMol(mol)
        oxygen_idx = rw_mol.AddAtom(Chem.Atom(8))
        rw_mol.AddBond(atom.GetIdx(), oxygen_idx, Chem.BondType.SINGLE)
        try:
            Chem.SanitizeMol(rw_mol)
        except Exception:
            continue
        candidates.append(
            GeneratedCandidate(
                smiles=canonicalize_smiles(Chem.MolToSmiles(rw_mol)),
                note="terminal methyl에 hydroxyl 추가",
            )
        )
    return candidates


def _carboxylic_acid_to_methyl_ester(mol: Chem.Mol) -> list[GeneratedCandidate]:
    """Generate methyl ester analogues from carboxylic acids where possible."""
    pattern = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
    candidates: list[GeneratedCandidate] = []
    if pattern is None:
        return candidates
    for match in mol.GetSubstructMatches(pattern):
        oxygen_idx = match[2]
        rw_mol = Chem.RWMol(mol)
        carbon_idx = rw_mol.AddAtom(Chem.Atom(6))
        rw_mol.AddBond(oxygen_idx, carbon_idx, Chem.BondType.SINGLE)
        try:
            Chem.SanitizeMol(rw_mol)
        except Exception:
            continue
        candidates.append(
            GeneratedCandidate(
                smiles=canonicalize_smiles(Chem.MolToSmiles(rw_mol)),
                note="carboxylic acid의 methyl ester analogue",
            )
        )
    return candidates


def generate_rule_based(smiles: str, max_candidates: int = 20) -> list[GeneratedCandidate]:
    """Generate valid analogue candidates using simple medicinal chemistry rules."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    candidates: list[GeneratedCandidate] = []
    candidates.extend(_replace_atoms(mol, 17, 9, "Cl을 F로 치환"))
    candidates.extend(_replace_atoms(mol, 35, 9, "Br을 F로 치환"))
    candidates.extend(_replace_atoms(mol, 53, 9, "I를 F로 치환"))
    candidates.extend(_methyl_to_hydroxymethyl(mol))
    candidates.extend(_carboxylic_acid_to_methyl_ester(mol))

    unique: dict[str, GeneratedCandidate] = {}
    original = canonicalize_smiles(smiles)
    for candidate in candidates:
        if candidate.smiles and candidate.smiles != original:
            unique.setdefault(candidate.smiles, candidate)
    return list(unique.values())[:max_candidates]

