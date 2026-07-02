"""Saliency analyzers for endpoint-specific molecular explanations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rdkit import Chem

from src.chemistry.alerts import ALERT_PATTERNS
from src.training.featurizer import smiles_to_data


@dataclass(frozen=True)
class SaliencyTarget:
    """A problem substructure candidate used by scaffold gating."""

    atom_indices: list[int]
    substructure_name: str
    reason: str
    toxic_alert: bool = False

    def to_dict(self) -> dict[str, object]:
        """Serialize for display."""
        return {
            "Substructure": self.substructure_name,
            "Atom index": ", ".join(str(idx) for idx in self.atom_indices),
            "Toxic alert": self.toxic_alert,
            "Reason": self.reason,
        }


@dataclass(frozen=True)
class SaliencySubstructure:
    """A scored substructure explanation for one endpoint."""

    name: str
    atom_indices: list[int]
    score: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Serialize for Streamlit tables."""
        return {
            "Substructure": self.name,
            "Atom indices": ", ".join(str(idx) for idx in self.atom_indices),
            "Score": round(float(self.score), 4),
            "Reason": self.reason,
        }


@dataclass(frozen=True)
class SaliencyResult:
    """Endpoint saliency in a predictor-agnostic structure."""

    endpoint: str
    source: str
    atom_scores: dict[int, float]
    top_atoms: list[int]
    substructures: list[SaliencySubstructure]

    def to_dict(self) -> dict[str, object]:
        """Serialize using the shared saliency result schema."""
        return {
            "endpoint": self.endpoint,
            "source": self.source,
            "atom_scores": self.atom_scores,
            "top_atoms": self.top_atoms,
            "substructures": [item.to_dict() for item in self.substructures],
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

    def explain(self, mol: Chem.Mol, endpoint: str, top_k: int = 8) -> SaliencyResult:
        """Return endpoint-specific heuristic atom scores and motif explanations."""
        atom_scores: dict[int, float] = {}
        substructures: list[SaliencySubstructure] = []

        for target in self.find_targets(mol):
            score = _heuristic_score(endpoint, target.substructure_name, target.toxic_alert)
            for atom_idx in target.atom_indices:
                atom_scores[atom_idx] = max(atom_scores.get(atom_idx, 0.0), score)
            substructures.append(
                SaliencySubstructure(
                    name=target.substructure_name,
                    atom_indices=target.atom_indices,
                    score=score,
                    reason=_endpoint_reason(endpoint, target.substructure_name, target.reason),
                )
            )

        top_atoms = sorted(atom_scores, key=atom_scores.get, reverse=True)[:top_k]
        substructures.sort(key=lambda item: item.score, reverse=True)
        return SaliencyResult(
            endpoint=endpoint,
            source="heuristic",
            atom_scores=atom_scores,
            top_atoms=top_atoms,
            substructures=substructures[:top_k],
        )


class GNNSaliencyAnalyzer:
    """Gradient-norm atom saliency for checkpoint-backed GNN predictors."""

    def explain(self, smiles: str, endpoint: str, predictor: Any, top_k: int = 8) -> SaliencyResult:
        """Compute atom feature gradient saliency for a single SMILES."""
        data = smiles_to_data(smiles)
        if data is None:
            raise ValueError(f"Invalid SMILES for GNN saliency: {smiles}")
        if not hasattr(predictor, "model") or not hasattr(predictor, "torch"):
            raise TypeError("GNN saliency requires a loaded GNNPredictor instance.")

        torch = predictor.torch
        device = getattr(predictor, "device", "cpu")
        model = predictor.model
        model.eval()

        data = data.to(device)
        data.x = data.x.detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)

        output = model(data).view(-1)[0]
        objective = torch.sigmoid(output) if getattr(predictor, "task", "") == "binary classification" else output
        objective.backward()

        if data.x.grad is None:
            raise RuntimeError("GNN saliency gradient was empty.")

        scores_tensor = data.x.grad.detach().norm(p=2, dim=1).cpu()
        max_score = float(scores_tensor.max().item()) if scores_tensor.numel() else 0.0
        if max_score <= 0.0:
            raise RuntimeError("GNN saliency gradients were all zero.")

        atom_scores = {
            int(atom_idx): float(score.item()) / max_score
            for atom_idx, score in enumerate(scores_tensor)
            if float(score.item()) > 0.0
        }
        top_atoms = sorted(atom_scores, key=atom_scores.get, reverse=True)[:top_k]
        mol = Chem.MolFromSmiles(smiles)
        substructures = _substructures_from_scores(mol, endpoint, atom_scores, top_atoms, top_k) if mol else []
        return SaliencyResult(
            endpoint=endpoint,
            source="gnn",
            atom_scores=atom_scores,
            top_atoms=top_atoms,
            substructures=substructures,
        )


def explain_endpoint_saliency(
    smiles: str,
    mol: Chem.Mol,
    endpoint: str,
    predictor: Any | None = None,
    prefer_gnn: bool = True,
    top_k: int = 8,
) -> SaliencyResult:
    """Use GNN saliency when available, otherwise return heuristic saliency."""
    if prefer_gnn and predictor is not None:
        try:
            result = GNNSaliencyAnalyzer().explain(smiles, endpoint, predictor, top_k=top_k)
            if result.atom_scores:
                return result
        except Exception:
            pass
    return HeuristicSaliencyAnalyzer().explain(mol, endpoint, top_k=top_k)


def _heuristic_score(endpoint: str, motif: str, toxic_alert: bool) -> float:
    """Score a motif by endpoint relevance for MVP explanations."""
    endpoint_key = endpoint.lower()
    motif_scores: dict[str, dict[str, float]] = {
        "solubility": {"long_alkyl_chain": 0.85, "aromatic_ring": 0.62, "halogen": 0.58},
        "lipophilicity": {"long_alkyl_chain": 0.82, "halogen": 0.78, "aromatic_ring": 0.68},
        "bbb": {"basic_amine": 0.78, "aromatic_ring": 0.58, "halogen": 0.52},
        "herg": {"basic_amine": 0.92, "halogen": 0.78, "aromatic_ring": 0.72, "long_alkyl_chain": 0.62},
        "ames": {"nitro_group": 0.96, "aniline_like": 0.9, "aromatic_ring": 0.55},
    }
    score = motif_scores.get(endpoint_key, {}).get(motif, 0.45)
    if toxic_alert:
        score = max(score, 0.86)
    return float(min(score, 1.0))


def _endpoint_reason(endpoint: str, motif: str, fallback: str) -> str:
    """Create a concise endpoint-specific saliency rationale."""
    endpoint_key = endpoint.lower()
    reasons: dict[str, dict[str, str]] = {
        "solubility": {
            "long_alkyl_chain": "Hydrophobic alkyl motif may contribute to low predicted solubility.",
            "aromatic_ring": "Aromatic surface can reduce aqueous solubility in the heuristic view.",
            "halogen": "Halogen substitution may increase hydrophobicity and lower solubility.",
        },
        "lipophilicity": {
            "long_alkyl_chain": "Hydrophobic chain is a strong contributor to high predicted lipophilicity.",
            "halogen": "Halogen atoms often increase predicted lipophilicity.",
            "aromatic_ring": "Aromatic ring area can contribute to elevated LogP.",
        },
        "bbb": {
            "basic_amine": "Basic amine can affect BBB permeability prediction depending on CNS context.",
            "aromatic_ring": "Aromatic surface can influence predicted BBB penetration.",
            "halogen": "Halogen substitution may shift CNS exposure-related properties.",
        },
        "herg": {
            "basic_amine": "Basic amine is highlighted as a possible hERG risk contributor.",
            "halogen": "Halogenated hydrophobic groups can contribute to predicted hERG liability.",
            "aromatic_ring": "Aromatic hydrophobic surface may contribute to predicted hERG risk.",
            "long_alkyl_chain": "Hydrophobic chain may contribute to predicted hERG risk.",
        },
        "ames": {
            "nitro_group": "Nitro group is a structural alert associated with predicted AMES risk.",
            "aniline_like": "Aniline-like motif is a structural alert associated with predicted AMES risk.",
            "aromatic_ring": "Aromatic motif is highlighted as context around predicted AMES risk.",
        },
    }
    return reasons.get(endpoint_key, {}).get(motif, fallback)


def _substructures_from_scores(
    mol: Chem.Mol,
    endpoint: str,
    atom_scores: dict[int, float],
    top_atoms: list[int],
    top_k: int,
) -> list[SaliencySubstructure]:
    """Attach SMARTS motif names to gradient-highlighted atoms where possible."""
    substructures: list[SaliencySubstructure] = []
    top_atom_set = set(top_atoms)
    for target in HeuristicSaliencyAnalyzer().find_targets(mol):
        if not top_atom_set.intersection(target.atom_indices):
            continue
        score = max(atom_scores.get(atom_idx, 0.0) for atom_idx in target.atom_indices)
        substructures.append(
            SaliencySubstructure(
                name=target.substructure_name,
                atom_indices=target.atom_indices,
                score=score,
                reason=_endpoint_reason(endpoint, target.substructure_name, target.reason),
            )
        )

    if not substructures and top_atoms:
        substructures.append(
            SaliencySubstructure(
                name="top gradient atoms",
                atom_indices=top_atoms,
                score=max(atom_scores.get(atom_idx, 0.0) for atom_idx in top_atoms),
                reason=f"Highest atom-feature gradient contribution for predicted {endpoint} output.",
            )
        )

    substructures.sort(key=lambda item: item.score, reverse=True)
    return substructures[:top_k]
