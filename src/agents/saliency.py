"""Heuristic saliency analyzer based on SMARTS alerts."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from src.chemistry.alerts import ALERT_PATTERNS


@dataclass(frozen=True)
class SaliencyTarget:
    """A problem substructure candidate."""

    atom_indices: list[int]
    substructure_name: str
    reason: str
    toxic_alert: bool = False

    def to_dict(self) -> dict[str, object]:
        """Serialize for display."""
        return {
            "Substructure": self.substructure_name,
            "Atom index": ", ".join(str(idx) for idx in self.atom_indices),
            "독성 alert": self.toxic_alert,
            "사유": self.reason,
        }


class HeuristicSaliencyAnalyzer:
    """Find substructure targets using RDKit SMARTS patterns."""

    def find_targets(self, mol: Chem.Mol) -> list[SaliencyTarget]:
        """Return matched substructure targets."""
        targets: list[SaliencyTarget] = []
        seen: set[tuple[str, tuple[int, ...]]] = set()
        for alert in ALERT_PATTERNS:
            pattern = Chem.MolFromSmarts(alert.smarts)
            if pattern is None:
                continue
            for match in mol.GetSubstructMatches(pattern):
                key = (alert.name, tuple(match))
                if key in seen:
                    continue
                seen.add(key)
                targets.append(
                    SaliencyTarget(
                        atom_indices=list(match),
                        substructure_name=alert.name,
                        reason=alert.reason,
                        toxic_alert=alert.toxic_alert,
                    )
                )
        return targets

