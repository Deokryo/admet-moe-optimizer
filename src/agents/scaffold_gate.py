"""Scaffold gate decisions for saliency targets."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from src.agents.saliency import SaliencyTarget
from src.chemistry.scaffold import get_scaffold_atom_indices


@dataclass(frozen=True)
class ScaffoldDecision:
    """Editable/protected/scaffold-hopping decision for one saliency target."""

    substructure_name: str
    atom_indices: list[int]
    location: str
    decision: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Serialize for display."""
        return {
            "Substructure": self.substructure_name,
            "Atom index": ", ".join(str(idx) for idx in self.atom_indices),
            "위치": self.location,
            "판단": self.decision,
            "사유": self.reason,
        }


class ScaffoldGate:
    """Determine whether targets are scaffold core or editable R-groups."""

    def evaluate(self, mol: Chem.Mol, targets: list[SaliencyTarget]) -> list[ScaffoldDecision]:
        """Return scaffold decisions for saliency targets."""
        scaffold_atoms = get_scaffold_atom_indices(mol)
        decisions: list[ScaffoldDecision] = []
        for target in targets:
            target_atoms = set(target.atom_indices)
            in_scaffold = bool(scaffold_atoms and target_atoms.issubset(scaffold_atoms))
            if target.toxic_alert and in_scaffold:
                decision = "scaffold_hopping_candidate"
                reason = "독성 alert가 Murcko scaffold와 겹칩니다."
            elif in_scaffold:
                decision = "protected"
                reason = "해당 target이 Murcko scaffold core에 포함됩니다."
            else:
                decision = "editable"
                reason = "해당 target이 R-group 또는 peripheral motif로 보입니다."
            decisions.append(
                ScaffoldDecision(
                    substructure_name=target.substructure_name,
                    atom_indices=target.atom_indices,
                    location="scaffold_core" if in_scaffold else "r_group_or_side_chain",
                    decision=decision,
                    reason=reason,
                )
            )
        return decisions

