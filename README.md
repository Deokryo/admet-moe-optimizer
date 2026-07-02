# ADMET-MoE Molecular Optimizer

Python/Streamlit 기반 AI 신약개발 MVP입니다. 초기 SMILES를 입력하면 RDKit descriptor를 계산하고, 5개 ADMET endpoint expert로 후보의 이상 물성을 탐지한 뒤, rule-based 또는 optional CReM 기반 analogue 후보를 생성하고 Top-K 후보를 추천합니다.

이 프로젝트는 대회 제안서/본선 시연용 프로토타입입니다. 출력은 "ADMET risk가 낮게 예측되는 후보"에 대한 모델/휴리스틱 추천이며, 실제 실험 검증이나 전문가 검토를 대체하지 않습니다.

## 설치

Windows에서는 PyTDC가 `cellxgene-census/tiledbsoma`를 끌고 오며 wheel build에 실패할 수 있습니다. 이 프로젝트는 TDC의 ADME/Tox single prediction dataset만 사용하므로 PyTDC는 별도 `--no-deps` 방식으로 설치합니다.

권장 환경은 Python 3.11입니다.

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

TDC 데이터셋 학습용 PyTDC:

```powershell
pip install --no-cache-dir --no-deps -r requirements-tdc.txt
```

설치 확인:

```powershell
python -c "import rdkit, torch, torch_geometric, sklearn, tdc; print('ok'); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## 실행

```powershell
streamlit run app.py
```

앱 sidebar의 `페이지`에서 다음 화면을 선택할 수 있습니다.

- `Molecule Optimizer`: 분자 최적화 workflow
- `GNN Training Dashboard`: GNN 학습 결과, checkpoint 상태, config, metric 시각화

`Molecule Optimizer`의 `Predictor mode`에서 `Dummy / Heuristic` 또는 `GNN Checkpoint`를 선택할 수 있습니다. GNN checkpoint가 없는 endpoint는 앱이 중단되지 않고 dummy predictor로 fallback합니다.

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

## GNN 학습 결과 Dashboard

학습 후 앱을 실행하고 sidebar에서 `GNN Training Dashboard`를 선택합니다.

Dashboard에서 확인할 수 있는 항목:

- 5개 모델 비교 summary table
- Dataset별 checkpoint/config/metrics file 존재 여부
- Best epoch, best validation metric
- Test metric 요약
- Train loss / valid loss 학습 곡선
- Regression: valid MAE, valid RMSE, valid R2
- Classification: valid AUROC, valid AUPRC, valid F1, valid Accuracy

Plotly가 설치되어 있으면 interactive chart를 사용하고, import에 실패하면 Streamlit 기본 `st.line_chart`로 fallback합니다.

## metrics.json 구조

새 학습 스크립트는 다음 구조로 `metrics.json`을 저장합니다.

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
    chemistry/
    predictors/
    training/
    dashboard/
    agents/
    generation/
    utils/
```

## Disclaimer

This MVP provides heuristic or learned ADMET risk estimates for proposal/demo use only. It does not claim that any molecule is a safe drug and does not replace experimental validation, clinical evidence, or expert medicinal chemistry review.
