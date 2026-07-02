"""Candidate generator facade."""

from __future__ import annotations

from dataclasses import dataclass

from src.generation.crem_wrapper import generate_with_crem, is_crem_available
from src.generation.rule_based import GeneratedCandidate, generate_rule_based


@dataclass(frozen=True)
class GenerationResult:
    """Candidate generation output with provenance and status metadata."""

    candidates: list[GeneratedCandidate]
    used_crem: bool
    used_rule_based: bool
    crem_available: bool
    crem_status: str
    rule_based_status: str
    warnings: list[str]


class CandidateGenerator:
    """Generate candidates using optional CReM with rule-based fallback."""

    def generate(
        self,
        smiles: str,
        max_candidates: int = 20,
        use_crem: bool = False,
        crem_db_path: str | None = None,
        target_atom_indices: list[int] | None = None,
        target_substructure: dict[str, object] | None = None,
        preserve_scaffold: bool = True,
        min_similarity: float = 0.3,
    ) -> GenerationResult:
        """Generate valid, deduplicated candidates with status metadata."""
        crem_available, availability_status = is_crem_available()
        warnings: list[str] = []

        if use_crem:
            crem_candidates, crem_status, crem_warnings = generate_with_crem(
                smiles=smiles,
                max_candidates=max_candidates,
                db_path=crem_db_path,
                target_atom_indices=target_atom_indices,
                min_similarity=min_similarity,
                preserve_scaffold=preserve_scaffold,
            )
            warnings.extend(crem_warnings)
        else:
            crem_candidates = []
            crem_status = "CReM disabled"

        if not crem_available and use_crem:
            crem_status = availability_status

        remaining = max(0, max_candidates - len(crem_candidates))
        if remaining > 0:
            rule_candidates = generate_rule_based(
                smiles,
                max_candidates=max_candidates,
                target_atom_indices=target_atom_indices,
                target_substructure=target_substructure,
            )
            used_rule_based = True
            rule_based_status = f"Rule-based fallback generated {len(rule_candidates)} candidates"
        else:
            rule_candidates = []
            used_rule_based = False
            rule_based_status = "Rule-based fallback skipped because CReM filled max_candidates"

        unique: dict[str, GeneratedCandidate] = {}
        for candidate in [*crem_candidates, *rule_candidates]:
            unique.setdefault(candidate.smiles, candidate)

        return GenerationResult(
            candidates=list(unique.values())[:max_candidates],
            used_crem=bool(crem_candidates),
            used_rule_based=used_rule_based,
            crem_available=crem_available,
            crem_status=crem_status,
            rule_based_status=rule_based_status,
            warnings=warnings,
        )
