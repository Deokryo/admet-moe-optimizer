"""Candidate generator facade."""

from __future__ import annotations

from src.generation.crem_wrapper import generate_with_crem
from src.generation.rule_based import GeneratedCandidate, generate_rule_based


class CandidateGenerator:
    """Generate candidates using optional CReM with rule-based fallback."""

    def generate(self, smiles: str, max_candidates: int = 20) -> list[GeneratedCandidate]:
        """Generate valid, deduplicated candidates."""
        candidates = generate_with_crem(smiles, max_candidates=max_candidates)
        if len(candidates) < max_candidates:
            candidates.extend(generate_rule_based(smiles, max_candidates=max_candidates))

        unique: dict[str, GeneratedCandidate] = {}
        for candidate in candidates:
            unique.setdefault(candidate.smiles, candidate)
        return list(unique.values())[:max_candidates]
