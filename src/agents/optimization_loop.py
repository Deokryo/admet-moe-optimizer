"""Closed-loop iterative molecular optimization agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from rdkit import Chem

from src.agents.abnormality_gate import AbnormalityGate
from src.agents.saliency import HeuristicSaliencyAnalyzer, SaliencyResult, explain_endpoint_saliency
from src.agents.scaffold_gate import ScaffoldGate
from src.chemistry.descriptors import calculate_descriptors
from src.chemistry.validation import mol_from_smiles
from src.generation.generator import CandidateGenerator, GenerationResult
from src.generation.rule_based import GeneratedCandidate
from src.predictors.base import Prediction, Predictor
from src.predictors.scoring import score_candidate, score_molecule
from src.utils.smiles import canonicalize_smiles


ENDPOINT_TO_PREDICTOR = {
    "Solubility": "Solubility Expert",
    "Lipophilicity": "Lipophilicity Expert",
    "BBB": "BBB Expert",
    "hERG": "hERG Expert",
    "AMES": "AMES Expert",
}

ENDPOINT_PRIORITY = {
    "hERG": 0,
    "AMES": 1,
    "Lipophilicity": 2,
    "Solubility": 3,
    "BBB": 4,
    "QED": 5,
}


@dataclass
class OptimizationStep:
    """One predict-diagnose-edit-validate iteration."""

    iteration: int
    input_smiles: str
    canonical_input_smiles: str
    predictions: dict[str, Prediction]
    abnormal_endpoints: list[Any]
    selected_endpoint: str | None
    saliency_result: dict[str, object] | None
    scaffold_decision: dict[str, object] | None
    generation_result: GenerationResult | None
    generated_candidates: list[dict[str, object]]
    selected_candidate: dict[str, object] | None
    selected_candidate_smiles: str | None
    selected_candidate_score: float | None
    improvement: float | None
    edit_reason: str
    stop_reason: str | None


@dataclass
class OptimizationLoopResult:
    """Complete closed-loop optimization result."""

    initial_smiles: str
    final_smiles: str
    steps: list[OptimizationStep]
    success: bool
    stop_reason: str
    total_iterations: int


class OptimizationLoop:
    """Greedy closed-loop optimizer using ADMET experts and candidate generation."""

    def __init__(
        self,
        predictors: list[Predictor],
        abnormality_gate: AbnormalityGate,
        saliency_analyzer: HeuristicSaliencyAnalyzer,
        scaffold_gate: ScaffoldGate,
        candidate_generator: CandidateGenerator,
        scorer: Callable[..., tuple[float, float]] = score_candidate,
        max_iterations: int = 5,
        max_candidates_per_step: int = 20,
        min_improvement: float = 0.01,
        use_crem: bool = False,
        crem_db_path: str | None = None,
        preserve_scaffold: bool = True,
        min_similarity: float = 0.3,
    ) -> None:
        """Initialize optimizer dependencies and loop settings."""
        self.predictors = predictors
        self.predictors_by_name = {predictor.name: predictor for predictor in predictors}
        self.abnormality_gate = abnormality_gate
        self.saliency_analyzer = saliency_analyzer
        self.scaffold_gate = scaffold_gate
        self.candidate_generator = candidate_generator
        self.scorer = scorer
        self.max_iterations = max_iterations
        self.max_candidates_per_step = max_candidates_per_step
        self.min_improvement = min_improvement
        self.use_crem = use_crem
        self.crem_db_path = crem_db_path
        self.preserve_scaffold = preserve_scaffold
        self.min_similarity = min_similarity

    def run(self, smiles: str, context: dict[str, object] | None = None) -> OptimizationLoopResult:
        """Run iterative optimization until endpoints pass or stopping criteria are met."""
        del context
        initial_smiles = smiles
        current_smiles = canonicalize_smiles(smiles)
        if not current_smiles:
            return OptimizationLoopResult(
                initial_smiles=initial_smiles,
                final_smiles=smiles,
                steps=[],
                success=False,
                stop_reason="Invalid initial SMILES",
                total_iterations=0,
            )

        steps: list[OptimizationStep] = []
        visited = {current_smiles}
        stop_reason = "Maximum iterations reached"
        success = False

        for iteration in range(1, self.max_iterations + 1):
            step = self._run_step(iteration, current_smiles)
            steps.append(step)

            if step.stop_reason:
                stop_reason = step.stop_reason
                success = step.stop_reason == "All endpoints are within acceptable ranges"
                break
            if not step.selected_candidate_smiles:
                stop_reason = "No candidate selected"
                break
            if step.selected_candidate_smiles == current_smiles:
                stop_reason = "Selected candidate is identical to current molecule"
                step.stop_reason = stop_reason
                break
            if step.selected_candidate_smiles in visited:
                stop_reason = "Repeated molecule detected"
                step.stop_reason = stop_reason
                break
            if step.improvement is None or step.improvement < self.min_improvement:
                stop_reason = "No meaningful improvement"
                step.stop_reason = stop_reason
                break

            visited.add(step.selected_candidate_smiles)
            current_smiles = step.selected_candidate_smiles
        else:
            stop_reason = "Maximum iterations reached"

        return OptimizationLoopResult(
            initial_smiles=initial_smiles,
            final_smiles=current_smiles,
            steps=steps,
            success=success,
            stop_reason=stop_reason,
            total_iterations=len(steps),
        )

    def _run_step(self, iteration: int, smiles: str) -> OptimizationStep:
        """Run a single optimization step."""
        mol = mol_from_smiles(smiles)
        if mol is None:
            return self._empty_step(iteration, smiles, "Invalid SMILES during optimization")

        canonical_smiles = canonicalize_smiles(smiles)
        descriptors = calculate_descriptors(mol)
        predictions = {predictor.name: predictor.predict(mol, descriptors) for predictor in self.predictors}
        abnormalities = self.abnormality_gate.evaluate(descriptors, predictions)
        current_score = score_molecule(descriptors, predictions, similarity_to_seed=1.0)

        if not abnormalities:
            return OptimizationStep(
                iteration=iteration,
                input_smiles=smiles,
                canonical_input_smiles=canonical_smiles,
                predictions=predictions,
                abnormal_endpoints=abnormalities,
                selected_endpoint=None,
                saliency_result=None,
                scaffold_decision=None,
                generation_result=None,
                generated_candidates=[],
                selected_candidate=None,
                selected_candidate_smiles=None,
                selected_candidate_score=current_score,
                improvement=0.0,
                edit_reason="All endpoints are within acceptable ranges",
                stop_reason="All endpoints are within acceptable ranges",
            )

        selected_endpoint = self._select_endpoint(abnormalities)
        saliency = self._explain(mol, canonical_smiles, selected_endpoint)
        target_atoms = saliency.top_atoms if saliency else None
        target_substructure = saliency.substructures[0].to_dict() if saliency and saliency.substructures else None

        scaffold_decisions = self.scaffold_gate.evaluate(mol, self.saliency_analyzer.find_targets(mol))
        scaffold_decision = self._select_scaffold_decision(scaffold_decisions, target_atoms)

        generation_result = self.candidate_generator.generate(
            canonical_smiles,
            max_candidates=self.max_candidates_per_step,
            use_crem=self.use_crem,
            crem_db_path=self.crem_db_path,
            target_atom_indices=target_atoms,
            target_substructure=target_substructure,
            preserve_scaffold=self.preserve_scaffold,
            min_similarity=self.min_similarity,
        )

        candidate_records = self._evaluate_candidates(mol, generation_result.candidates)
        if not candidate_records:
            return OptimizationStep(
                iteration=iteration,
                input_smiles=smiles,
                canonical_input_smiles=canonical_smiles,
                predictions=predictions,
                abnormal_endpoints=abnormalities,
                selected_endpoint=selected_endpoint,
                saliency_result=saliency.to_dict() if saliency else None,
                scaffold_decision=scaffold_decision,
                generation_result=generation_result,
                generated_candidates=[],
                selected_candidate=None,
                selected_candidate_smiles=None,
                selected_candidate_score=None,
                improvement=None,
                edit_reason="No valid candidates were generated",
                stop_reason="No valid candidates generated",
            )

        selected = candidate_records[0]
        selected_score = float(selected["score"])
        improvement = selected_score - current_score
        edit_reason = (
            f"Selected top-1 candidate for {selected_endpoint}; "
            f"score {selected_score:.3f} vs current {current_score:.3f}"
        )

        return OptimizationStep(
            iteration=iteration,
            input_smiles=smiles,
            canonical_input_smiles=canonical_smiles,
            predictions=predictions,
            abnormal_endpoints=abnormalities,
            selected_endpoint=selected_endpoint,
            saliency_result=saliency.to_dict() if saliency else None,
            scaffold_decision=scaffold_decision,
            generation_result=generation_result,
            generated_candidates=candidate_records,
            selected_candidate=selected,
            selected_candidate_smiles=str(selected["smiles"]),
            selected_candidate_score=selected_score,
            improvement=improvement,
            edit_reason=edit_reason,
            stop_reason=None,
        )

    def _evaluate_candidates(self, original_mol: Chem.Mol, candidates: list[GeneratedCandidate]) -> list[dict[str, object]]:
        """Evaluate and score generated candidates."""
        records: list[dict[str, object]] = []
        for candidate in candidates:
            candidate_mol = mol_from_smiles(candidate.smiles)
            if candidate_mol is None:
                continue
            descriptors = calculate_descriptors(candidate_mol)
            predictions = {predictor.name: predictor.predict(candidate_mol, descriptors) for predictor in self.predictors}
            score, similarity = self.scorer(original_mol, candidate_mol, descriptors, predictions)
            records.append(
                {
                    "smiles": candidate.smiles,
                    "generation_note": candidate.note,
                    "source": candidate.source,
                    "generation_method": candidate.generation_method,
                    "edited_region": candidate.edited_region,
                    "target_atoms": candidate.target_atoms,
                    "descriptors": descriptors,
                    "predictions": predictions,
                    "score": score,
                    "similarity": similarity,
                    "mol": candidate_mol,
                }
            )
        records.sort(key=lambda item: float(item["score"]), reverse=True)
        return records

    def _select_endpoint(self, abnormalities: list[Any]) -> str:
        """Select the highest-priority abnormal endpoint."""
        return min(abnormalities, key=lambda item: ENDPOINT_PRIORITY.get(getattr(item, "endpoint", ""), 99)).endpoint

    def _explain(self, mol: Chem.Mol, smiles: str, endpoint: str) -> SaliencyResult | None:
        """Explain the selected endpoint with GNN saliency when possible."""
        predictor_name = ENDPOINT_TO_PREDICTOR.get(endpoint)
        predictor = self.predictors_by_name.get(predictor_name or "")
        return explain_endpoint_saliency(smiles, mol, endpoint, predictor=predictor, prefer_gnn=bool(predictor), top_k=8)

    @staticmethod
    def _select_scaffold_decision(scaffold_decisions: list[Any], target_atoms: list[int] | None) -> dict[str, object] | None:
        """Return the scaffold decision overlapping target atoms, if any."""
        if not scaffold_decisions:
            return None
        if not target_atoms:
            return scaffold_decisions[0].to_dict()
        target_set = set(target_atoms)
        for decision in scaffold_decisions:
            if target_set.intersection(decision.atom_indices):
                return decision.to_dict()
        return scaffold_decisions[0].to_dict()

    @staticmethod
    def _empty_step(iteration: int, smiles: str, stop_reason: str) -> OptimizationStep:
        """Build a stopped step for invalid inputs."""
        return OptimizationStep(
            iteration=iteration,
            input_smiles=smiles,
            canonical_input_smiles=smiles,
            predictions={},
            abnormal_endpoints=[],
            selected_endpoint=None,
            saliency_result=None,
            scaffold_decision=None,
            generation_result=None,
            generated_candidates=[],
            selected_candidate=None,
            selected_candidate_smiles=None,
            selected_candidate_score=None,
            improvement=None,
            edit_reason=stop_reason,
            stop_reason=stop_reason,
        )
