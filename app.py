"""Streamlit app for the ADMET-MoE Molecular Optimizer MVP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.agents.abnormality_gate import AbnormalityConfig, AbnormalityGate
from src.agents.optimization_loop import OptimizationLoop, OptimizationLoopResult
from src.agents.report_agent import build_report, generate_optimization_loop_report
from src.agents.saliency import HeuristicSaliencyAnalyzer, explain_endpoint_saliency
from src.agents.scaffold_gate import ScaffoldGate
from src.chemistry.descriptors import calculate_descriptors
from src.chemistry.validation import mol_from_smiles, validate_smiles
from src.chemistry.visualization import draw_saliency_molecule, endpoint_color, mol_to_image
from src.dashboard.training_dashboard import render_training_dashboard
from src.generation.generator import CandidateGenerator
from src.predictors.base import Predictor
from src.predictors.dummy_predictors import build_dummy_predictors
from src.predictors.gnn_predictor import ENDPOINT_TO_DATASET, GNNPredictor
from src.predictors.scoring import score_candidate
from src.utils.smiles import canonicalize_smiles


DISCLAIMER = (
    "이 MVP는 제안서/시연용 메디시널 케미스트리 및 GNN 기반 ADMET risk 예측 도구입니다. "
    "실험 검증, 임상 근거, 전문가 검토를 대체하지 않습니다."
)

SALIENCY_DISCLAIMER = (
    "이 saliency는 해당 endpoint 예측에 크게 기여한 atom/substructure를 보여주는 "
    "gradient/heuristic 기반 설명이며, 화학적 인과관계를 확정하는 것은 아닙니다. "
    "후보 수정 방향 제안을 위한 참고 정보입니다."
)

ENDPOINT_LABELS = {
    "Solubility Expert": "용해도 Expert",
    "Lipophilicity Expert": "지용성 Expert",
    "BBB Expert": "BBB Expert",
    "hERG Expert": "hERG Expert",
    "AMES Expert": "AMES Expert",
}

TASK_LABELS = {
    "regression": "회귀",
    "binary classification": "이진 분류",
}

ABNORMAL_ENDPOINT_TO_PREDICTOR = {
    "Solubility": "Solubility Expert",
    "Lipophilicity": "Lipophilicity Expert",
    "BBB": "BBB Expert",
    "hERG": "hERG Expert",
    "AMES": "AMES Expert",
}


def build_predictors(predictor_mode: str, checkpoint_root: Path = Path("checkpoints")) -> tuple[list[Predictor], dict[str, str]]:
    """Build endpoint predictors with optional GNN checkpoint loading."""
    dummy_by_name = {predictor.name: predictor for predictor in build_dummy_predictors()}
    sources = {name: "Dummy / Heuristic" for name in dummy_by_name}
    if predictor_mode != "GNN Checkpoint":
        return list(dummy_by_name.values()), sources

    predictors: list[Predictor] = []
    for endpoint_name, fallback in dummy_by_name.items():
        dataset_name = ENDPOINT_TO_DATASET.get(endpoint_name)
        if dataset_name is None:
            predictors.append(fallback)
            continue
        checkpoint_path = checkpoint_root / dataset_name / "best.pt"
        try:
            predictor = GNNPredictor(dataset_name=dataset_name, checkpoint_path=checkpoint_path)
        except Exception as exc:
            st.warning(
                f"{ENDPOINT_LABELS.get(endpoint_name, endpoint_name)} checkpoint 로드 실패: "
                f"{exc} heuristic predictor로 대체합니다."
            )
            predictors.append(fallback)
            continue
        predictors.append(predictor)
        sources[endpoint_name] = "GNN Checkpoint"
    return predictors, sources


def _prediction_frame(predictions: dict[str, Any], predictor_sources: dict[str, str]) -> pd.DataFrame:
    """Convert endpoint predictions into a display table."""
    rows = []
    for name, prediction in predictions.items():
        rows.append(
            {
                "Endpoint": ENDPOINT_LABELS.get(name, name),
                "모델": predictor_sources.get(name, "-"),
                "태스크": TASK_LABELS.get(prediction.task, prediction.task),
                "예측값": round(float(prediction.value), 4),
                "단위": prediction.unit or "-",
                "신뢰도": round(float(prediction.confidence), 3),
                "해석": prediction.interpretation,
            }
        )
    return pd.DataFrame(rows)


def _descriptor_frame(descriptors: dict[str, float]) -> pd.DataFrame:
    """Convert descriptors into a vertical display table."""
    return pd.DataFrame([{"Descriptor": key, "값": round(float(value), 4)} for key, value in descriptors.items()])


def _candidate_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a compact Top-K candidate table."""
    rows = []
    for rank, record in enumerate(records, start=1):
        desc = record["descriptors"]
        preds = record["predictions"]
        rows.append(
            {
                "순위": rank,
                "SMILES": record["smiles"],
                "점수": round(float(record["score"]), 4),
                "MW": round(desc["molecular_weight"], 2),
                "LogP": round(desc["logp"], 2),
                "TPSA": round(desc["tpsa"], 2),
                "QED": round(desc["qed"], 3),
                "용해도": round(float(preds["Solubility Expert"].value), 3),
                "hERG Risk": round(float(preds["hERG Expert"].value), 3),
                "AMES Risk": round(float(preds["AMES Expert"].value), 3),
                "원본 유사도": round(float(record["similarity"]), 3),
                "Source": record["source"],
                "Generation Method": record["generation_method"],
                "Edited Region": record["edited_region"] or "-",
                "Target Atoms": ", ".join(str(atom) for atom in (record["target_atoms"] or [])) or "-",
                "Note": record["generation_note"] or "-",
            }
        )
    return pd.DataFrame(rows)


def _delta_table(original: dict[str, float], records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build original-vs-candidate descriptor deltas."""
    rows = []
    keys = ["molecular_weight", "logp", "tpsa", "hbd", "hba", "rotatable_bonds", "qed", "sa_score"]
    for record in records:
        row: dict[str, Any] = {"SMILES": record["smiles"]}
        desc = record["descriptors"]
        for key in keys:
            row[f"변화량 {key}"] = round(float(desc[key] - original[key]), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def _loop_prediction_frame(predictions: dict[str, Any]) -> pd.DataFrame:
    """Convert loop step predictions into a display table."""
    rows = []
    for name, prediction in predictions.items():
        rows.append(
            {
                "Endpoint": ENDPOINT_LABELS.get(name, name),
                "Task": TASK_LABELS.get(prediction.task, prediction.task),
                "Value": round(float(prediction.value), 4),
                "Unit": prediction.unit or "-",
                "Interpretation": prediction.interpretation,
            }
        )
    return pd.DataFrame(rows)


def _abnormality_frame(abnormalities: list[Any]) -> pd.DataFrame:
    """Convert abnormalities to a DataFrame."""
    return pd.DataFrame([item.to_dict() for item in abnormalities]) if abnormalities else pd.DataFrame()


def _loop_summary_rows(result: OptimizationLoopResult) -> list[dict[str, Any]]:
    """Build top summary rows for a loop result."""
    final_step = result.steps[-1] if result.steps else None
    final_abnormal = []
    final_score = None
    if final_step:
        final_abnormal = [getattr(item, "endpoint", "-") for item in final_step.abnormal_endpoints]
        final_score = final_step.selected_candidate_score
    return [
        {"항목": "초기 SMILES", "값": result.initial_smiles},
        {"항목": "최종 SMILES", "값": result.final_smiles},
        {"항목": "성공 여부", "값": result.success},
        {"항목": "종료 이유", "값": result.stop_reason},
        {"항목": "총 iteration 수", "값": result.total_iterations},
        {"항목": "최종 score", "값": round(float(final_score), 4) if final_score is not None else "-"},
        {"항목": "최종 abnormal endpoint", "값": ", ".join(final_abnormal) if final_abnormal else "없음"},
    ]


def _render_loop_result(result: OptimizationLoopResult) -> None:
    """Render iterative optimization results."""
    st.subheader("반복형 최적화 요약")
    st.dataframe(pd.DataFrame(_loop_summary_rows(result)), use_container_width=True, hide_index=True)

    if result.steps:
        path_rows = []
        for step in result.steps:
            path_rows.append(
                {
                    "Iteration": step.iteration,
                    "Input SMILES": step.canonical_input_smiles,
                    "Selected endpoint": step.selected_endpoint or "-",
                    "Selected candidate": step.selected_candidate_smiles or "-",
                    "Score": round(float(step.selected_candidate_score), 4) if step.selected_candidate_score is not None else "-",
                    "Improvement": round(float(step.improvement), 4) if step.improvement is not None else "-",
                    "Stop reason": step.stop_reason or "-",
                }
            )
        st.subheader("분자 변화 경로")
        st.dataframe(pd.DataFrame(path_rows), use_container_width=True, hide_index=True)

        metric_rows = []
        for step in result.steps:
            preds = step.predictions
            if not preds:
                continue
            metric_rows.append(
                {
                    "iteration": step.iteration,
                    "Solubility": float(preds["Solubility Expert"].value),
                    "Lipophilicity": float(preds["Lipophilicity Expert"].value),
                    "BBB": float(preds["BBB Expert"].value),
                    "hERG": float(preds["hERG Expert"].value),
                    "AMES": float(preds["AMES Expert"].value),
                }
            )
        if metric_rows:
            st.subheader("iteration별 endpoint 변화")
            st.line_chart(pd.DataFrame(metric_rows).set_index("iteration"))

    for step in result.steps:
        title = f"Step {step.iteration}: {step.selected_endpoint or '정상 범위'}"
        if step.selected_candidate_smiles:
            title += f" -> {step.selected_candidate_smiles}"
        with st.expander(title, expanded=step.iteration == 1):
            col_in, col_out = st.columns(2)
            input_mol = mol_from_smiles(step.canonical_input_smiles)
            if input_mol is not None:
                with col_in:
                    st.markdown("**입력 분자**")
                    st.image(mol_to_image(input_mol, size=(320, 240)), caption=step.canonical_input_smiles)
            if step.selected_candidate_smiles:
                selected_mol = mol_from_smiles(step.selected_candidate_smiles)
                if selected_mol is not None:
                    with col_out:
                        st.markdown("**선택 후보**")
                        st.image(mol_to_image(selected_mol, size=(320, 240)), caption=step.selected_candidate_smiles)

            if step.predictions:
                st.markdown("**5 Expert 예측**")
                st.dataframe(_loop_prediction_frame(step.predictions), use_container_width=True, hide_index=True)
            if step.abnormal_endpoints:
                st.markdown("**이상 endpoint**")
                st.dataframe(_abnormality_frame(step.abnormal_endpoints), use_container_width=True, hide_index=True)
            else:
                st.success("모든 endpoint가 허용 범위에 들어와 최적화를 종료했습니다.")

            st.markdown(f"**이번 단계에서 선택한 이상 endpoint:** {step.selected_endpoint or '-'}")
            if step.saliency_result:
                atom_scores = step.saliency_result.get("atom_scores", {})
                top_atoms = step.saliency_result.get("top_atoms", [])
                source = step.saliency_result.get("source", "-")
                st.markdown(f"**편집 근거 saliency:** {source}, top atoms {top_atoms}")
                if isinstance(atom_scores, dict) and atom_scores:
                    image = draw_saliency_molecule(
                        step.canonical_input_smiles,
                        {int(key): float(value) for key, value in atom_scores.items()},
                        color=endpoint_color(step.selected_endpoint or ""),
                    )
                    if image is not None:
                        st.image(image, caption="Saliency highlighted molecule")
                substructures = step.saliency_result.get("substructures", [])
                if isinstance(substructures, list) and substructures:
                    st.dataframe(pd.DataFrame(substructures), use_container_width=True, hide_index=True)

            if step.scaffold_decision:
                st.markdown("**Scaffold gate decision**")
                st.dataframe(pd.DataFrame([step.scaffold_decision]), use_container_width=True, hide_index=True)

            if step.generation_result:
                st.markdown("**후보 생성 상태**")
                status_rows = [
                    {"항목": "CReM used", "값": step.generation_result.used_crem},
                    {"항목": "Rule-based used", "값": step.generation_result.used_rule_based},
                    {"항목": "CReM status", "값": step.generation_result.crem_status},
                    {"항목": "Rule-based status", "값": step.generation_result.rule_based_status},
                    {"항목": "Generated candidates", "값": len(step.generation_result.candidates)},
                ]
                st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)
                for warning in step.generation_result.warnings:
                    st.warning(warning)

            if step.generated_candidates:
                st.markdown("**후보 Top-K**")
                st.dataframe(_candidate_table(step.generated_candidates[:10]), use_container_width=True, hide_index=True)

            st.markdown(f"**선택 이유:** {step.edit_reason}")
            st.markdown(f"**Improvement:** {step.improvement if step.improvement is not None else '-'}")
            if step.stop_reason:
                st.markdown(f"**종료 이유:** {step.stop_reason}")

    report = generate_optimization_loop_report(result)
    st.subheader("반복형 최적화 리포트")
    st.markdown(report)
    st.download_button("리포트 다운로드", report, file_name="optimization_loop_report.md", mime="text/markdown")
    st.caption(
        "본 결과는 계산 기반 ADMET risk prioritization이며 실제 효능과 안전성을 보장하지 않습니다. "
        "각 단계의 saliency는 모델/휴리스틱 기반 설명으로, 화학적 인과관계를 확정하지 않습니다. "
        "실제 신약 개발에는 후속 wet lab 검증과 전문가 검토가 필요합니다."
    )


def _saliency_endpoint_options(abnormalities: list[Any]) -> list[str]:
    """Return abnormal endpoints that can be explained with atom saliency."""
    options: list[str] = []
    for abnormality in abnormalities:
        endpoint = getattr(abnormality, "endpoint", "")
        if endpoint in ABNORMAL_ENDPOINT_TO_PREDICTOR and endpoint not in options:
            options.append(endpoint)
    return options


def _source_label(source: str) -> str:
    """Return a UI label for the saliency source."""
    return "GNN Saliency" if source == "gnn" else "Heuristic Saliency"


def _extract_target_atoms(target_substructure: dict[str, object] | None) -> list[int] | None:
    """Extract atom indices from a stored saliency target dictionary."""
    if not target_substructure:
        return None
    raw_atoms = target_substructure.get("Atom indices") or target_substructure.get("Atom index") or target_substructure.get("atom_indices")
    if isinstance(raw_atoms, str):
        atoms = [int(item.strip()) for item in raw_atoms.split(",") if item.strip().isdigit()]
        return atoms or None
    if isinstance(raw_atoms, list):
        atoms = [int(item) for item in raw_atoms if isinstance(item, int)]
        return atoms or None
    return None


def _stored_generation_target() -> tuple[list[int] | None, dict[str, object] | None, str]:
    """Return the user-selected saliency target from session state if available."""
    selected = st.session_state.get("selected_saliency_target")
    if not isinstance(selected, dict):
        return None, None, "선택된 saliency target 없음"
    substructure = selected.get("substructure")
    target_substructure = substructure if isinstance(substructure, dict) else None
    target_atoms = selected.get("top_atoms")
    if not isinstance(target_atoms, list) or not target_atoms:
        target_atoms = _extract_target_atoms(target_substructure)
    target_atoms = [int(atom) for atom in target_atoms] if target_atoms else None
    endpoint = selected.get("endpoint", "-")
    source = selected.get("source", "-")
    name = target_substructure.get("Substructure", "-") if target_substructure else "-"
    label = f"{endpoint} / {name} / atoms {target_atoms or []} / source: {source}"
    return target_atoms, target_substructure, label


def _render_saliency_section(result: dict[str, Any]) -> None:
    """Render endpoint-specific atom saliency visualization."""
    st.subheader("이상 endpoint atom saliency")
    endpoints = _saliency_endpoint_options(result["abnormalities"])
    if not endpoints:
        st.info("시각화할 수 있는 비정상 ADMET endpoint가 없습니다.")
        return

    endpoint = st.selectbox("Saliency를 확인할 endpoint", endpoints)
    predictor_name = ABNORMAL_ENDPOINT_TO_PREDICTOR[endpoint]
    predictor = result.get("predictor_objects", {}).get(predictor_name)
    prefer_gnn = result.get("predictor_sources", {}).get(predictor_name) == "GNN Checkpoint"

    try:
        saliency = explain_endpoint_saliency(
            smiles=str(result["original_smiles"]),
            mol=result["mol"],
            endpoint=endpoint,
            predictor=predictor,
            prefer_gnn=prefer_gnn,
            top_k=8,
        )
    except Exception as exc:
        st.warning(f"Saliency 계산 실패: {exc}")
        return

    if not saliency.atom_scores:
        st.info("시각화 가능한 saliency가 없습니다.")
        return

    image = draw_saliency_molecule(
        smiles=str(result["original_smiles"]),
        atom_scores=saliency.atom_scores,
        top_k=8,
        width=520,
        height=400,
        color=endpoint_color(endpoint),
    )
    if image is None:
        st.warning("RDKit saliency drawing에 실패했습니다.")
    else:
        st.image(image, caption=f"{endpoint} - {_source_label(saliency.source)}")

    top_substructure = saliency.substructures[0] if saliency.substructures else None
    info_rows = [
        {"항목": "Endpoint", "값": endpoint},
        {"항목": "Source", "값": _source_label(saliency.source)},
        {"항목": "Top atoms", "값": ", ".join(str(idx) for idx in saliency.top_atoms) or "-"},
        {"항목": "Highlighted atoms", "값": len(saliency.top_atoms)},
        {"항목": "Top substructure", "값": top_substructure.name if top_substructure else "-"},
        {"항목": "Reason", "값": top_substructure.reason if top_substructure else "-"},
    ]
    st.dataframe(pd.DataFrame(info_rows), use_container_width=True, hide_index=True)

    if saliency.substructures:
        st.markdown("**상위 substructure**")
        st.dataframe(
            pd.DataFrame([item.to_dict() for item in saliency.substructures]),
            use_container_width=True,
            hide_index=True,
        )
        labels = [
            f"{idx + 1}. {item.name} ({', '.join(str(atom) for atom in item.atom_indices)})"
            for idx, item in enumerate(saliency.substructures)
        ]
        selected_label = st.selectbox("수정 대상으로 사용할 substructure", labels)
        selected_idx = labels.index(selected_label)
        if st.button("이 substructure를 수정 대상으로 사용"):
            selected = saliency.substructures[selected_idx]
            st.session_state["selected_saliency_target"] = {
                "endpoint": saliency.endpoint,
                "source": saliency.source,
                "top_atoms": saliency.top_atoms,
                "substructure": selected.to_dict(),
            }
            st.success("선택한 substructure를 후보 생성 수정 target으로 저장했습니다.")

    st.caption(SALIENCY_DISCLAIMER)


def run_analysis(
    smiles: str,
    is_cns_target: bool,
    top_k: int,
    herg_threshold: float,
    ames_threshold: float,
    min_solubility: float,
    min_logp: float,
    max_logp: float,
    use_crem: bool,
    crem_db_path: str | None,
    min_similarity: float,
    preserve_scaffold: bool,
    selected_target_atom_indices: list[int] | None,
    selected_target_substructure: dict[str, object] | None,
    predictor_mode: str,
) -> dict[str, Any]:
    """Run the complete parse-predict-generate-rank workflow."""
    valid, error = validate_smiles(smiles)
    if not valid:
        raise ValueError(error or "유효하지 않은 SMILES입니다.")

    original_smiles = canonicalize_smiles(smiles)
    mol = mol_from_smiles(original_smiles)
    if mol is None:
        raise ValueError("RDKit이 입력 SMILES를 파싱하지 못했습니다.")

    descriptors = calculate_descriptors(mol)
    predictors, predictor_sources = build_predictors(predictor_mode)
    predictor_objects = {predictor.name: predictor for predictor in predictors}
    predictions = {predictor.name: predictor.predict(mol, descriptors) for predictor in predictors}

    abnormality_gate = AbnormalityGate(
        AbnormalityConfig(
            is_cns_target=is_cns_target,
            herg_threshold=herg_threshold,
            ames_threshold=ames_threshold,
            min_solubility=min_solubility,
            min_logp=min_logp,
            max_logp=max_logp,
        )
    )
    abnormalities = abnormality_gate.evaluate(descriptors, predictions)

    saliency_analyzer = HeuristicSaliencyAnalyzer()
    saliency_targets = saliency_analyzer.find_targets(mol)
    target_atom_indices = selected_target_atom_indices
    target_substructure = selected_target_substructure
    target_label = "선택된 saliency target 없음"
    if target_atom_indices:
        target_label = f"UI selected target atoms {target_atom_indices}"
    elif saliency_targets:
        first_target = saliency_targets[0]
        target_atom_indices = first_target.atom_indices
        target_substructure = first_target.to_dict()
        target_label = f"자동 saliency target: {first_target.substructure_name} / atoms {first_target.atom_indices}"

    scaffold_gate = ScaffoldGate()
    scaffold_decisions = scaffold_gate.evaluate(mol, saliency_targets)

    generator = CandidateGenerator()
    generation_result = generator.generate(
        original_smiles,
        max_candidates=max(top_k * 4, 12),
        use_crem=use_crem,
        crem_db_path=crem_db_path,
        target_atom_indices=target_atom_indices,
        target_substructure=target_substructure,
        preserve_scaffold=preserve_scaffold,
        min_similarity=min_similarity,
    )

    records: list[dict[str, Any]] = []
    for candidate in generation_result.candidates:
        candidate_mol = mol_from_smiles(candidate.smiles)
        if candidate_mol is None:
            continue
        candidate_desc = calculate_descriptors(candidate_mol)
        candidate_preds = {predictor.name: predictor.predict(candidate_mol, candidate_desc) for predictor in predictors}
        score, similarity = score_candidate(mol, candidate_mol, candidate_desc, candidate_preds)
        records.append(
            {
                "smiles": candidate.smiles,
                "generation_note": candidate.note,
                "source": candidate.source,
                "generation_method": candidate.generation_method,
                "edited_region": candidate.edited_region,
                "target_atoms": candidate.target_atoms,
                "descriptors": candidate_desc,
                "predictions": candidate_preds,
                "score": score,
                "similarity": similarity,
                "mol": candidate_mol,
            }
        )

    records.sort(key=lambda item: float(item["score"]), reverse=True)
    top_records = records[:top_k]

    report = build_report(
        original_smiles=original_smiles,
        descriptors=descriptors,
        predictions=predictions,
        abnormalities=abnormalities,
        saliency_targets=saliency_targets,
        scaffold_decisions=scaffold_decisions,
        candidate_records=top_records,
    )

    return {
        "original_smiles": original_smiles,
        "mol": mol,
        "target_config": {
            "Solubility minimum": min_solubility,
            "LogP minimum": min_logp,
            "LogP maximum": max_logp,
            "hERG threshold": herg_threshold,
            "AMES threshold": ames_threshold,
        },
        "descriptors": descriptors,
        "predictions": predictions,
        "predictor_objects": predictor_objects,
        "predictor_sources": predictor_sources,
        "abnormalities": abnormalities,
        "saliency_targets": saliency_targets,
        "generation_target_label": target_label,
        "generation_result": generation_result,
        "scaffold_decisions": scaffold_decisions,
        "candidate_records": top_records,
        "report": report,
    }


def run_optimization_loop(
    smiles: str,
    is_cns_target: bool,
    herg_threshold: float,
    ames_threshold: float,
    min_solubility: float,
    min_logp: float,
    max_logp: float,
    predictor_mode: str,
    max_iterations: int,
    max_candidates_per_step: int,
    min_improvement: float,
    use_crem: bool,
    crem_db_path: str | None,
    min_similarity: float,
    preserve_scaffold: bool,
) -> OptimizationLoopResult:
    """Run the closed-loop iterative optimizer."""
    predictors, _predictor_sources = build_predictors(predictor_mode)
    abnormality_gate = AbnormalityGate(
        AbnormalityConfig(
            is_cns_target=is_cns_target,
            herg_threshold=herg_threshold,
            ames_threshold=ames_threshold,
            min_solubility=min_solubility,
            min_logp=min_logp,
            max_logp=max_logp,
        )
    )
    loop = OptimizationLoop(
        predictors=predictors,
        abnormality_gate=abnormality_gate,
        saliency_analyzer=HeuristicSaliencyAnalyzer(),
        scaffold_gate=ScaffoldGate(),
        candidate_generator=CandidateGenerator(),
        scorer=score_candidate,
        max_iterations=max_iterations,
        max_candidates_per_step=max_candidates_per_step,
        min_improvement=min_improvement,
        use_crem=use_crem,
        crem_db_path=crem_db_path,
        preserve_scaffold=preserve_scaffold,
        min_similarity=min_similarity,
    )
    return loop.run(smiles)


def render_molecule_optimizer() -> None:
    """Render the molecule optimizer page."""
    st.title("ADMET-MoE 분자 최적화 MVP")
    st.caption(DISCLAIMER)

    with st.sidebar:
        st.header("입력")
        smiles = st.text_input("초기 SMILES", value="CC(C)Oc1ccc(Cl)cc1C(=O)O")
        target_options = ["비-CNS 타깃", "CNS 타깃"]
        target_context = st.radio("타깃 맥락", target_options, index=0)
        predictor_mode = st.radio("Predictor mode", ["Dummy / Heuristic", "GNN Checkpoint"], index=0)
        execution_mode = st.radio("실행 모드", ["단일 후보 생성", "반복형 최적화"], index=0)
        top_k = st.slider("추천 후보 수 Top-K", min_value=3, max_value=20, value=8, step=1)
        st.markdown("**목표 물성 범위**")
        min_solubility = st.slider("Solubility minimum", -8.0, 1.0, -3.5, 0.1)
        min_logp, max_logp = st.slider("LogP 목표 범위", -1.0, 6.0, (1.0, 3.0), 0.1)
        herg_threshold = st.slider("hERG risk 임계값", 0.1, 0.9, 0.55, 0.05)
        ames_threshold = st.slider("AMES risk 임계값", 0.1, 0.9, 0.50, 0.05)
        st.markdown("**후보 생성 엔진**")
        use_crem = st.checkbox("CReM 후보 생성 사용", value=False)
        crem_db_path = st.text_input("CReM fragment DB 경로", value="")
        min_similarity = st.slider("Minimum similarity to original", 0.0, 1.0, 0.3, 0.05)
        preserve_scaffold = st.checkbox("Scaffold 보존", value=True)
        if execution_mode == "반복형 최적화":
            st.markdown("**반복형 최적화 설정**")
            max_iterations = st.slider("max_iterations", 1, 10, 5, 1)
            max_candidates_per_step = st.slider("max_candidates_per_step", 5, 50, 20, 1)
            min_improvement = st.number_input("min_improvement", min_value=0.0, max_value=1.0, value=0.01, step=0.01)
        else:
            max_iterations = 1
            max_candidates_per_step = max(top_k * 4, 12)
            min_improvement = 0.01
        run_button = st.button("최적화 실행", type="primary")

    if run_button:
        try:
            if execution_mode == "반복형 최적화":
                loop_result = run_optimization_loop(
                    smiles=smiles,
                    is_cns_target=target_context == target_options[1],
                    herg_threshold=herg_threshold,
                    ames_threshold=ames_threshold,
                    min_solubility=min_solubility,
                    min_logp=min_logp,
                    max_logp=max_logp,
                    predictor_mode=predictor_mode,
                    max_iterations=max_iterations,
                    max_candidates_per_step=max_candidates_per_step,
                    min_improvement=min_improvement,
                    use_crem=use_crem,
                    crem_db_path=crem_db_path,
                    min_similarity=min_similarity,
                    preserve_scaffold=preserve_scaffold,
                )
                st.session_state["optimization_loop_result"] = loop_result
                st.session_state["last_execution_mode"] = execution_mode
                result = loop_result
            else:
                selected_atoms, selected_substructure, _selected_label = _stored_generation_target()
                result = run_analysis(
                    smiles=smiles,
                    is_cns_target=target_context == target_options[1],
                    top_k=top_k,
                    herg_threshold=herg_threshold,
                    ames_threshold=ames_threshold,
                    min_solubility=min_solubility,
                    min_logp=min_logp,
                    max_logp=max_logp,
                    use_crem=use_crem,
                    crem_db_path=crem_db_path,
                    min_similarity=min_similarity,
                    preserve_scaffold=preserve_scaffold,
                    selected_target_atom_indices=selected_atoms,
                    selected_target_substructure=selected_substructure,
                    predictor_mode=predictor_mode,
                )
                st.session_state["optimizer_result"] = result
                st.session_state["last_execution_mode"] = execution_mode
        except Exception as exc:
            st.error(f"분석 실패: {exc}")
            return
    elif st.session_state.get("last_execution_mode") == "반복형 최적화" and "optimization_loop_result" in st.session_state:
        result = st.session_state["optimization_loop_result"]
    elif "optimizer_result" in st.session_state:
        result = st.session_state["optimizer_result"]
        if "generation_result" not in result:
            st.info("후보 생성 엔진 설정이 변경되었습니다. 최적화를 다시 실행하세요.")
            return
    else:
        st.info("초기 SMILES를 입력한 뒤 최적화를 실행하세요.")
        return

    if isinstance(result, OptimizationLoopResult):
        _render_loop_result(result)
        return

    left, right = st.columns([1, 2])
    with left:
        st.subheader("원본 분자")
        st.image(mol_to_image(result["mol"], size=(360, 280)), caption=result["original_smiles"])
    with right:
        st.subheader("RDKit 분자 descriptor")
        st.dataframe(_descriptor_frame(result["descriptors"]), use_container_width=True, hide_index=True)

    st.subheader("현재 목표값 / 임계값")
    st.dataframe(
        pd.DataFrame(
            [{"항목": key, "값": round(float(value), 4)} for key, value in result["target_config"].items()]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("ADMET endpoint 예측")
    st.dataframe(_prediction_frame(result["predictions"], result["predictor_sources"]), use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("이상 endpoint")
        abnormalities = result["abnormalities"]
        if abnormalities:
            st.dataframe(pd.DataFrame([item.to_dict() for item in abnormalities]), use_container_width=True, hide_index=True)
        else:
            st.success("현재 gate에서 이상 endpoint가 감지되지 않았습니다.")

    with col_b:
        st.subheader("문제 substructure 후보")
        targets = result["saliency_targets"]
        if targets:
            st.dataframe(pd.DataFrame([target.to_dict() for target in targets]), use_container_width=True, hide_index=True)
        else:
            st.info("휴리스틱 substructure target이 발견되지 않았습니다.")

    _render_saliency_section(result)

    st.subheader("Scaffold gate 판단")
    decisions = result["scaffold_decisions"]
    if decisions:
        st.dataframe(pd.DataFrame([decision.to_dict() for decision in decisions]), use_container_width=True, hide_index=True)
    else:
        st.info("생성된 scaffold gate 판단이 없습니다.")

    generation_result = result["generation_result"]
    st.subheader("후보 생성 상태")
    status_rows = [
        {"항목": "CReM available", "값": generation_result.crem_available},
        {"항목": "CReM used", "값": generation_result.used_crem},
        {"항목": "Rule-based used", "값": generation_result.used_rule_based},
        {"항목": "CReM status", "값": generation_result.crem_status},
        {"항목": "Rule-based status", "값": generation_result.rule_based_status},
        {"항목": "Generation target", "값": result["generation_target_label"]},
        {"항목": "Final candidates", "값": len(generation_result.candidates)},
    ]
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)
    for warning in generation_result.warnings:
        st.warning(warning)

    st.subheader("생성 후보 Top-K")
    records = result["candidate_records"]
    if records:
        st.dataframe(_candidate_table(records), use_container_width=True, hide_index=True)
        st.subheader("원본 대비 후보 물성 변화")
        st.dataframe(_delta_table(result["descriptors"], records), use_container_width=True, hide_index=True)

        st.subheader("후보 구조")
        cols = st.columns(3)
        for idx, record in enumerate(records):
            with cols[idx % 3]:
                st.image(mol_to_image(record["mol"], size=(300, 220)), caption=f"{idx + 1}. {record['smiles']}")
    else:
        st.warning(
            "유효한 후보가 생성되지 않았습니다. Cl, Br, alkyl, ester 등 편집 가능한 치환기가 있는 분자를 시도해보세요."
        )

    st.subheader("자동 생성 리포트")
    st.text_area("리포트", value=result["report"], height=280)
    st.caption(DISCLAIMER)


def main() -> None:
    """Render the selected Streamlit page."""
    st.set_page_config(page_title="ADMET-MoE Molecular Optimizer", layout="wide")
    with st.sidebar:
        page = st.radio("페이지", ["Molecule Optimizer", "GNN Training Dashboard"], index=0)

    if page == "GNN Training Dashboard":
        render_training_dashboard()
    else:
        render_molecule_optimizer()


if __name__ == "__main__":
    main()
