"""Smoke test the closed-loop optimization agent."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.abnormality_gate import AbnormalityConfig, AbnormalityGate
from src.agents.optimization_loop import OptimizationLoop
from src.agents.saliency import HeuristicSaliencyAnalyzer
from src.agents.scaffold_gate import ScaffoldGate
from src.generation.generator import CandidateGenerator
from src.predictors.dummy_predictors import build_dummy_predictors
from src.predictors.scoring import score_candidate


def main() -> None:
    """Run a two-iteration dummy-predictor smoke test."""
    loop = OptimizationLoop(
        predictors=build_dummy_predictors(),
        abnormality_gate=AbnormalityGate(AbnormalityConfig(is_cns_target=False)),
        saliency_analyzer=HeuristicSaliencyAnalyzer(),
        scaffold_gate=ScaffoldGate(),
        candidate_generator=CandidateGenerator(),
        scorer=score_candidate,
        max_iterations=2,
        max_candidates_per_step=10,
        min_improvement=0.0,
        use_crem=False,
        preserve_scaffold=True,
        min_similarity=0.3,
    )
    result = loop.run("CC(=O)Oc1ccccc1C(=O)O")
    print(f"steps={len(result.steps)}")
    print(f"final_smiles={result.final_smiles}")
    print(f"success={result.success}")
    print(f"stop_reason={result.stop_reason}")
    for step in result.steps:
        print(
            f"step={step.iteration} endpoint={step.selected_endpoint} "
            f"selected={step.selected_candidate_smiles} improvement={step.improvement} "
            f"stop={step.stop_reason}"
        )


if __name__ == "__main__":
    main()
