"""Streamlit app for the ADMET-MoE Molecular Optimizer MVP."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.agents.abnormality_gate import AbnormalityConfig, AbnormalityGate
from src.agents.report_agent import build_report
from src.agents.saliency import HeuristicSaliencyAnalyzer
from src.agents.scaffold_gate import ScaffoldGate
from src.chemistry.descriptors import calculate_descriptors
from src.chemistry.validation import mol_from_smiles, validate_smiles
from src.chemistry.visualization import mol_to_image
from src.dashboard.training_dashboard import render_training_dashboard
from src.generation.generator import CandidateGenerator
from src.predictors.base import Predictor
from src.predictors.dummy_predictors import build_dummy_predictors
from src.predictors.gnn_predictor import ENDPOINT_TO_DATASET, GNNPredictor
from src.predictors.scoring import score_candidate
from src.utils.smiles import canonicalize_smiles


DISCLAIMER = (
    "이 MVP는 제안서/시연용 휴리스틱 및 GNN 기반 ADMET risk 예측 도구입니다. "
    "실험 검증, 임상 근거, 전문가 검토를 대체하지 않습니다."
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
            st.warning(f"{ENDPOINT_LABELS.get(endpoint_name, endpoint_name)} checkpoint 로드 실패: {exc} heuristic predictor로 대체합니다.")
            predictors.append(fallback)
            continue
        predictors.append(predictor)
        sources[endpoint_name] = "GNN Checkpoint"
    return predictors, sources


def _prediction_frame(predictions: dict[str, object], predictor_sources: dict[str, str]) -> pd.DataFrame:
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


def _candidate_table(records: list[dict[str, object]]) -> pd.DataFrame:
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
                "생성 규칙": record["generation_note"],
            }
        )
    return pd.DataFrame(rows)


def _delta_table(original: dict[str, float], records: list[dict[str, object]]) -> pd.DataFrame:
    """Build original-vs-candidate descriptor deltas."""
    rows = []
    keys = ["molecular_weight", "logp", "tpsa", "hbd", "hba", "rotatable_bonds", "qed", "sa_score"]
    for record in records:
        row: dict[str, object] = {"SMILES": record["smiles"]}
        desc = record["descriptors"]
        for key in keys:
            row[f"변화량 {key}"] = round(float(desc[key] - original[key]), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def run_analysis(
    smiles: str,
    is_cns_target: bool,
    top_k: int,
    herg_threshold: float,
    ames_threshold: float,
    predictor_mode: str,
) -> dict[str, object]:
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
    predictions = {predictor.name: predictor.predict(mol, descriptors) for predictor in predictors}

    abnormality_gate = AbnormalityGate(
        AbnormalityConfig(
            is_cns_target=is_cns_target,
            herg_threshold=herg_threshold,
            ames_threshold=ames_threshold,
        )
    )
    abnormalities = abnormality_gate.evaluate(descriptors, predictions)

    saliency_analyzer = HeuristicSaliencyAnalyzer()
    saliency_targets = saliency_analyzer.find_targets(mol)

    scaffold_gate = ScaffoldGate()
    scaffold_decisions = scaffold_gate.evaluate(mol, saliency_targets)

    generator = CandidateGenerator()
    generated = generator.generate(original_smiles, max_candidates=max(top_k * 4, 12))

    records: list[dict[str, object]] = []
    for candidate in generated:
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
        "descriptors": descriptors,
        "predictions": predictions,
        "predictor_sources": predictor_sources,
        "abnormalities": abnormalities,
        "saliency_targets": saliency_targets,
        "scaffold_decisions": scaffold_decisions,
        "candidate_records": top_records,
        "report": report,
    }


def render_molecule_optimizer() -> None:
    """Render the molecule optimizer page."""
    st.title("ADMET-MoE 분자 최적화 MVP")
    st.caption(DISCLAIMER)

    with st.sidebar:
        st.header("입력")
        smiles = st.text_input("초기 SMILES", value="CC(C)Oc1ccc(Cl)cc1C(=O)O")
        target_context = st.radio("타깃 맥락", ["비-CNS 타깃", "CNS 타깃"], index=0)
        predictor_mode = st.radio("Predictor mode", ["Dummy / Heuristic", "GNN Checkpoint"], index=0)
        top_k = st.slider("추천 후보 수 Top-K", min_value=3, max_value=20, value=8, step=1)
        herg_threshold = st.slider("hERG risk 임계값", 0.1, 0.9, 0.55, 0.05)
        ames_threshold = st.slider("AMES risk 임계값", 0.1, 0.9, 0.50, 0.05)
        run_button = st.button("최적화 실행", type="primary")

    if not run_button:
        st.info("초기 SMILES를 입력한 뒤 최적화를 실행하세요.")
        return

    try:
        result = run_analysis(
            smiles=smiles,
            is_cns_target=target_context == "CNS 타깃",
            top_k=top_k,
            herg_threshold=herg_threshold,
            ames_threshold=ames_threshold,
            predictor_mode=predictor_mode,
        )
    except Exception as exc:
        st.error(f"분석 실패: {exc}")
        return

    left, right = st.columns([1, 2])
    with left:
        st.subheader("원본 분자")
        st.image(mol_to_image(result["mol"], size=(360, 280)), caption=result["original_smiles"])
    with right:
        st.subheader("RDKit 물성 descriptor")
        st.dataframe(_descriptor_frame(result["descriptors"]), use_container_width=True, hide_index=True)

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

    st.subheader("Scaffold gate 판단")
    decisions = result["scaffold_decisions"]
    if decisions:
        st.dataframe(pd.DataFrame([decision.to_dict() for decision in decisions]), use_container_width=True, hide_index=True)
    else:
        st.info("생성된 scaffold gate 판단이 없습니다.")

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
        st.warning("유효한 후보가 생성되지 않았습니다. Cl, Br, alkyl, ester 등 편집 가능한 치환기가 있는 분자를 시도해보세요.")

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
