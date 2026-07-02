# ADMET-MoE Molecular Optimizer

Python/Streamlit 기반 AI 신약개발 MVP입니다. 초기 SMILES를 입력하면 RDKit descriptor를 계산하고, 5개 ADMET endpoint expert로 이상 물성을 탐지한 뒤, rule-based 또는 optional CReM 기반 analogue 후보를 생성하고 Top-K 후보를 추천합니다.

이 프로젝트는 대회 제안서/본선 시연용 프로토타입입니다. 출력은 "ADMET risk가 낮게 예측되는 후보"에 대한 모델/휴리스틱 추천이며, 실제 실험 검증이나 전문가 검토를 대체하지 않습니다.

## 설치

권장 환경은 Python 3.11입니다. Windows에서는 RDKit을 conda로 설치하는 편이 안정적입니다.

```powershell
conda create -n admet-moe python=3.11 -y
conda activate admet-moe
conda install -c conda-forge rdkit -y
```

CUDA 12.8 PyTorch 예시:

```powershell
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

프로젝트 기본 의존성:

```powershell
pip install --no-cache-dir -r requirements.txt
```

TDC 데이터셋 학습용 PyTDC는 Windows에서 `cellxgene-census/tiledbsoma` 의존성 때문에 wheel build가 실패할 수 있습니다. 이 프로젝트는 TDC의 ADME/Tox single prediction dataset만 사용하므로 PyTDC를 별도로 설치합니다.

```powershell
pip install --no-cache-dir --no-deps -r requirements-tdc.txt
pip install huggingface-hub
```

설치 확인:

```powershell
python -c "import rdkit, torch, torch_geometric, sklearn, tdc; print('ok'); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## 실행

```powershell
streamlit run app.py
```

sidebar의 `페이지`에서 다음 화면을 선택할 수 있습니다.

- `Molecule Optimizer`: 분자 최적화 workflow
- `GNN Training Dashboard`: GNN 학습 결과, checkpoint 상태, config, metric 시각화

`Molecule Optimizer`의 `Predictor mode`에서 `Dummy / Heuristic` 또는 `GNN Checkpoint`를 선택할 수 있습니다. GNN checkpoint가 없는 endpoint는 앱이 중단되지 않고 dummy predictor로 fallback됩니다.

## Atom Saliency Visualization

Optimizer는 abnormal endpoint가 감지되면 해당 endpoint 기준 atom saliency를 분자 구조 위에 highlight합니다.

- GNN checkpoint가 있는 endpoint: atom feature gradient norm 기반 `GNN Saliency`를 우선 사용합니다.
- checkpoint가 없거나 GNN saliency 계산이 실패한 endpoint: SMARTS motif 기반 `Heuristic Saliency`로 fallback합니다.
- 표시 정보: endpoint, source, top atoms, highlight atom 수, top substructure, reason, substructure table
- `이 substructure를 수정 대상으로 사용` 버튼을 누르면 선택한 target이 `st.session_state["selected_saliency_target"]`에 저장되어 후보 생성 로직 확장에 사용할 수 있습니다.

Saliency는 예측 모델 또는 휴리스틱이 크게 반응한 atom/substructure를 보여주는 설명 도구입니다. 화학적 인과관계를 확정하거나 실제 독성/효능을 보증하지 않으며, 후보 수정 방향을 잡기 위한 참고 정보로만 사용해야 합니다.

## GNN 학습

빠른 smoke test는 epoch를 5-10 정도로 낮춰 실행할 수 있습니다.

```powershell
python -m src.training.train --dataset Solubility_AqSolDB --task regression --epochs 10
python -m src.training.train --dataset Lipophilicity_AstraZeneca --task regression --epochs 10
python -m src.training.train --dataset BBB_Martins --task classification --epochs 10
python -m src.training.train --dataset hERG_Karim --task classification --epochs 10
python -m src.training.train --dataset AMES --task classification --epochs 10
```

학습 결과는 아래 위치에 저장됩니다.

```text
checkpoints/{dataset_name}/best.pt
checkpoints/{dataset_name}/config.json
checkpoints/{dataset_name}/metrics.json
```

### TDC 다운로드/캐시 문제 해결

`dataverse.harvard.edu` DNS 오류가 나면 네트워크 연결을 확인한 뒤 다시 실행합니다.

로컬 `data/*.tab` 파일이 부분 다운로드되어 `EOF inside string` 또는 `Error tokenizing data`가 나면 `--force-redownload`를 붙여 다시 받습니다.

```powershell
python -m src.training.train --dataset AMES --task classification --epochs 1 --force-redownload
python -m src.training.train --dataset hERG_Karim --task classification --epochs 1 --force-redownload
```

데이터 저장 위치를 바꾸고 싶으면:

```powershell
python -m src.training.train --dataset AMES --task classification --epochs 1 --tdc-data-dir ./data
```

## GNN Training Dashboard

학습 명령을 실행한 뒤 Streamlit sidebar에서 `GNN Training Dashboard`를 선택합니다.

Dashboard에서 확인할 수 있는 항목:

- 5개 모델 비교 summary table
- Dataset별 checkpoint/config/metrics file 존재 여부
- Best epoch, best validation metric
- Test metric 요약
- Train loss / valid loss 학습 곡선
- Regression: valid MAE, valid RMSE, valid R2
- Classification: valid AUROC, valid AUPRC, valid F1, valid Accuracy

Plotly가 설치되어 있으면 interactive chart를 사용하고, import가 실패하면 Streamlit 기본 `st.line_chart`로 fallback됩니다.

## metrics.json 구조

학습 스크립트는 다음 구조로 `metrics.json`을 저장합니다.

```json
{
  "dataset": "Solubility_AqSolDB",
  "task": "regression",
  "history": [
    {
      "epoch": 1,
      "train_loss": 0.8,
      "valid_loss": 0.7,
      "valid_mae": 0.5,
      "valid_rmse": 0.8,
      "valid_r2": 0.31
    }
  ],
  "best_epoch": 12,
  "best_valid_metric": 0.45,
  "test_metrics": {
    "mae": 0.47,
    "rmse": 0.72,
    "r2": 0.31
  }
}
```

Classification task는 `valid_auroc`, `valid_auprc`, `valid_f1`, `valid_accuracy`와 test metric을 저장합니다. Dashboard는 이전 구조의 `metrics.json`도 최대한 읽도록 하위 호환 helper를 포함합니다.

## Project Structure

```text
admet_moe_optimizer/
  app.py
  README.md
  requirements.txt
  requirements-tdc.txt
  src/
    agents/
    chemistry/
    dashboard/
    generation/
    predictors/
    training/
    utils/
```

## Disclaimer

This MVP provides heuristic or learned ADMET risk estimates for proposal/demo use only. It does not claim that any molecule is a safe drug and does not replace experimental validation, clinical evidence, or expert medicinal chemistry review.
