"""Plain-language report generator."""

from __future__ import annotations

from src.agents.abnormality_gate import Abnormality
from src.agents.saliency import SaliencyTarget
from src.agents.scaffold_gate import ScaffoldDecision
from src.predictors.base import Prediction


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

