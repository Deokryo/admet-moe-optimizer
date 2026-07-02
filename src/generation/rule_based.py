"""Rule-based analogue generation fallback."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from src.utils.smiles import canonicalize_smiles


@dataclass(frozen=True)
class GeneratedCandidate:
    """Generated candidate molecule and provenance metadata."""

    smiles: str
    note: str | None = None
    source: str = "rule_based"
    generation_method: str = "rule_based"
    edited_region: str | None = None
    target_atoms: list[int] | None = None


def _target_note(target_atom_indices: list[int] | None) -> str | None:
    """Return a concise target atom note for provenance."""
    if not target_atom_indices:
        return None
    return f"target-informed generation; requested atoms={target_atom_indices}"


def _replace_atoms(
    mol: Chem.Mol,
    atomic_num_from: int,
    atomic_num_to: int,
    note: str,
    method: str,
    target_atom_indices: list[int] | None = None,
) -> list[GeneratedCandidate]:
    """Generate candidates by replacing one atom type with another."""
    candidates: list[GeneratedCandidate] = []
    target_set = set(target_atom_indices or [])
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
        edited_region = "target_atom" if atom.GetIdx() in target_set else "global_atom_replacement"
        target_note = _target_note(target_atom_indices)
        candidates.append(
            GeneratedCandidate(
                smiles=smiles,
                note=f"{note}; {target_note}" if target_note else note,
                source="rule_based",
                generation_method=method,
                edited_region=edited_region,
                target_atoms=[atom.GetIdx()],
            )
        )
    return candidates


def _methyl_to_hydroxymethyl(
    mol: Chem.Mol,
    target_atom_indices: list[int] | None = None,
) -> list[GeneratedCandidate]:
    """Convert terminal methyl groups into hydroxymethyl-like substituents."""
    candidates: list[GeneratedCandidate] = []
    target_set = set(target_atom_indices or [])
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
        note = "terminal methyl에 hydroxyl 추가"
        target_note = _target_note(target_atom_indices)
        candidates.append(
            GeneratedCandidate(
                smiles=canonicalize_smiles(Chem.MolToSmiles(rw_mol)),
                note=f"{note}; {target_note}" if target_note else note,
                source="rule_based",
                generation_method="terminal_methyl_hydroxylation",
                edited_region="target_side_chain" if atom.GetIdx() in target_set else "terminal_methyl",
                target_atoms=[atom.GetIdx()],
            )
        )
    return candidates


def _carboxylic_acid_to_methyl_ester(
    mol: Chem.Mol,
    target_atom_indices: list[int] | None = None,
) -> list[GeneratedCandidate]:
    """Generate methyl ester analogues from carboxylic acids where possible."""
    pattern = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
    candidates: list[GeneratedCandidate] = []
    if pattern is None:
        return candidates
    target_set = set(target_atom_indices or [])
    for match in mol.GetSubstructMatches(pattern):
        oxygen_idx = match[2]
        rw_mol = Chem.RWMol(mol)
        carbon_idx = rw_mol.AddAtom(Chem.Atom(6))
        rw_mol.AddBond(oxygen_idx, carbon_idx, Chem.BondType.SINGLE)
        try:
            Chem.SanitizeMol(rw_mol)
        except Exception:
            continue
        note = "carboxylic acid를 methyl ester analogue로 변환"
        target_note = _target_note(target_atom_indices)
        candidates.append(
            GeneratedCandidate(
                smiles=canonicalize_smiles(Chem.MolToSmiles(rw_mol)),
                note=f"{note}; {target_note}" if target_note else note,
                source="rule_based",
                generation_method="carboxylic_acid_methyl_ester",
                edited_region="target_substructure" if target_set.intersection(match) else "carboxylic_acid",
                target_atoms=list(match),
            )
        )
    return candidates


def generate_rule_based(
    smiles: str,
    max_candidates: int = 20,
    target_atom_indices: list[int] | None = None,
    target_substructure: dict[str, object] | None = None,
) -> list[GeneratedCandidate]:
    """Generate valid analogue candidates using simple medicinal chemistry rules."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    effective_targets = target_atom_indices
    if not effective_targets and target_substructure:
        raw_atoms = target_substructure.get("Atom indices") or target_substructure.get("atom_indices")
        if isinstance(raw_atoms, str):
            effective_targets = [int(item.strip()) for item in raw_atoms.split(",") if item.strip().isdigit()]
        elif isinstance(raw_atoms, list):
            effective_targets = [int(item) for item in raw_atoms if isinstance(item, int)]

    candidates: list[GeneratedCandidate] = []
    candidates.extend(_replace_atoms(mol, 17, 9, "Cl을 F로 치환", "halogen_replacement_Cl_to_F", effective_targets))
    candidates.extend(_replace_atoms(mol, 35, 9, "Br을 F로 치환", "halogen_replacement_Br_to_F", effective_targets))
    candidates.extend(_replace_atoms(mol, 53, 9, "I를 F로 치환", "halogen_replacement_I_to_F", effective_targets))
    candidates.extend(_methyl_to_hydroxymethyl(mol, effective_targets))
    candidates.extend(_carboxylic_acid_to_methyl_ester(mol, effective_targets))

    unique: dict[str, GeneratedCandidate] = {}
    original = canonicalize_smiles(smiles)
    for candidate in candidates:
        if candidate.smiles and candidate.smiles != original:
            unique.setdefault(candidate.smiles, candidate)
    return list(unique.values())[:max_candidates]
