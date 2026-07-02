"""Plain-language report generator."""

from __future__ import annotations

from src.agents.abnormality_gate import Abnormality
from src.agents.saliency import SaliencyTarget
from src.agents.scaffold_gate import ScaffoldDecision
from src.predictors.base import Prediction
from src.agents.optimization_loop import OptimizationLoopResult


def build_report(
    original_smiles: str,
    descriptors: dict[str, float],
    predictions: dict[str, Prediction],
    abnormalities: list[Abnormality],
    saliency_targets: list[SaliencyTarget],
    scaffold_decisions: list[ScaffoldDecision],
    candidate_records: list[dict[str, object]],
) -> str:
    """Create a concise demo report for the current optimization run."""
    lines = [
        "ADMET-MoE 분자 최적화 MVP 리포트",
        "",
        f"입력 SMILES: {original_smiles}",
        f"핵심 물성: MW={descriptors['molecular_weight']:.2f}, LogP={descriptors['logp']:.2f}, "
        f"TPSA={descriptors['tpsa']:.2f}, QED={descriptors['qed']:.3f}, SA placeholder={descriptors['sa_score']:.2f}",
        "",
        "Endpoint 예측 요약:",
    ]
    for prediction in predictions.values():
        lines.append(f"- {prediction.endpoint}: {prediction.value:.3f} ({prediction.interpretation})")

    lines.append("")
    if abnormalities:
        lines.append("감지된 이상 endpoint:")
        for item in abnormalities:
            lines.append(f"- {item.endpoint}: {item.reason} 예측값={item.value:.3f}")
    else:
        lines.append("현재 gate에서 이상 endpoint가 감지되지 않았습니다.")

    lines.append("")
    if saliency_targets:
        lines.append("휴리스틱 saliency target:")
        for target in saliency_targets[:8]:
            lines.append(f"- {target.substructure_name}, atom {target.atom_indices}: {target.reason}")
    else:
        lines.append("휴리스틱 substructure target이 발견되지 않았습니다.")

    lines.append("")
    if scaffold_decisions:
        lines.append("Scaffold gate 판단:")
        for decision in scaffold_decisions[:8]:
            lines.append(f"- {decision.substructure_name}: {decision.decision} ({decision.reason})")

    lines.append("")
    if candidate_records:
        lines.append("아래 후보는 MVP scoring function 기준으로 상대적으로 낮은 ADMET risk가 예측된 추천 후보입니다:")
        for rank, record in enumerate(candidate_records[:5], start=1):
            lines.append(f"- #{rank} score={float(record['score']):.3f}: {record['smiles']} [{record['generation_note']}]")
    else:
        lines.append("유효한 후보가 생성되지 않았습니다.")

    lines.extend(
        [
            "",
            "Disclaimer: 본 결과는 제안서/시연용 ADMET risk 추정입니다. "
            "특정 분자가 안전하거나 유효하다는 근거가 아니며, 실험 검증을 대체하지 않습니다.",
        ]
    )
    return "\n".join(lines)


def generate_optimization_loop_report(result: OptimizationLoopResult) -> str:
    """Create a markdown summary for an iterative optimization loop."""
    lines = [
        "# 반복형 ADMET-MoE 최적화 리포트",
        "",
        f"- 초기 SMILES: `{result.initial_smiles}`",
        f"- 최종 SMILES: `{result.final_smiles}`",
        f"- 성공 여부: {result.success}",
        f"- 종료 이유: {result.stop_reason}",
        f"- 총 iteration 수: {result.total_iterations}",
        "",
        "## Iteration 요약",
    ]
    for step in result.steps:
        abnormal_names = [getattr(item, "endpoint", "-") for item in step.abnormal_endpoints]
        top_substructure = "-"
        saliency_source = "-"
        if step.saliency_result:
            saliency_source = str(step.saliency_result.get("source", "-"))
            substructures = step.saliency_result.get("substructures", [])
            if isinstance(substructures, list) and substructures:
                first = substructures[0]
                if isinstance(first, dict):
                    top_substructure = str(first.get("Substructure", first.get("name", "-")))

        lines.extend(
            [
                "",
                f"### Step {step.iteration}",
                f"- 입력 분자: `{step.canonical_input_smiles}`",
                f"- 이상 endpoint: {', '.join(abnormal_names) if abnormal_names else '없음'}",
                f"- 선택 endpoint: {step.selected_endpoint or '-'}",
                f"- saliency source: {saliency_source}",
                f"- 주요 substructure: {top_substructure}",
                f"- 선택 후보: `{step.selected_candidate_smiles or '-'}`",
                f"- 선택 후보 score: {step.selected_candidate_score if step.selected_candidate_score is not None else '-'}",
                f"- improvement: {step.improvement if step.improvement is not None else '-'}",
                f"- 편집 근거: {step.edit_reason}",
            ]
        )
        if step.stop_reason:
            lines.append(f"- 종료 이유: {step.stop_reason}")

    lines.extend(
        [
            "",
            "## Disclaimer",
            "본 결과는 계산 기반 ADMET risk prioritization이며 실제 효능과 안전성을 보장하지 않습니다. "
            "각 단계의 saliency는 모델/휴리스틱 기반 설명으로, 화학적 인과관계를 확정하지 않습니다. "
            "실제 신약 개발에는 후속 wet lab 검증과 전문가 검토가 필요합니다.",
        ]
    )
    return "\n".join(lines)
