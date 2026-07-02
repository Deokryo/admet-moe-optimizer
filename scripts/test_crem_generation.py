"""Smoke test optional CReM generation and rule-based fallback."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.generator import CandidateGenerator
from src.generation.crem_wrapper import is_crem_available


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Test CReM candidate generation with fallback.")
    parser.add_argument("--smiles", required=True, help="Input molecule SMILES.")
    parser.add_argument("--db-path", default=None, help="CReM fragment database path.")
    parser.add_argument("--max-candidates", type=int, default=10, help="Maximum number of candidates.")
    parser.add_argument("--min-similarity", type=float, default=0.3, help="Minimum Tanimoto similarity.")
    parser.add_argument("--no-preserve-scaffold", action="store_true", help="Disable scaffold preservation filter.")
    return parser.parse_args()


def main() -> None:
    """Run the generation smoke test."""
    args = parse_args()
    available, status = is_crem_available()
    print(f"CReM available: {available}")
    print(f"CReM availability status: {status}")

    result = CandidateGenerator().generate(
        smiles=args.smiles,
        max_candidates=args.max_candidates,
        use_crem=True,
        crem_db_path=args.db_path,
        min_similarity=args.min_similarity,
        preserve_scaffold=not args.no_preserve_scaffold,
    )

    print(f"CReM status: {result.crem_status}")
    print(f"Rule-based status: {result.rule_based_status}")
    print(f"CReM used: {result.used_crem}")
    print(f"Rule-based used: {result.used_rule_based}")
    print(f"Final candidate count: {len(result.candidates)}")
    for warning in result.warnings:
        print(f"WARNING: {warning}")

    for idx, candidate in enumerate(result.candidates, start=1):
        print(
            f"{idx}\t{candidate.source}\t{candidate.generation_method}\t"
            f"{candidate.smiles}\t{candidate.note or ''}"
        )


if __name__ == "__main__":
    main()
