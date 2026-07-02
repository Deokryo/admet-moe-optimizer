# ADMET-MoE Molecular Optimizer

## 1. Overview

ADMET-MoE Molecular Optimizer는 초기 SMILES를 입력하면 RDKit 기반 descriptor 계산, ADMET endpoint 예측, abnormal endpoint 탐지, atom saliency 시각화, 후보 분자 생성, 후보 재평가를 한 번에 수행하는 Python/Streamlit 기반 AI 신약개발 MVP입니다.

이 프로젝트는 대회 제안서 및 본선 시연용 프로토타입입니다. 출력은 “ADMET risk가 낮게 예측되는 후보”에 대한 모델/휴리스틱 기반 추천이며, 실제 실험 검증이나 전문가 검토를 대체하지 않습니다.

주요 기능:

- RDKit descriptor 계산: MW, LogP, TPSA, HBD, HBA, rotatable bonds, QED 등
- 5개 ADMET endpoint expert: Solubility, Lipophilicity, BBB, hERG, AMES
- Dummy heuristic predictor와 checkpoint 기반 GNN predictor fallback 구조
- GNN/heuristic atom saliency visualization
- Rule-based 및 optional CReM 기반 후보 생성
- 반복형 closed-loop molecular optimization
- TDC 기반 GNN 학습 및 10-fold CV benchmark
- Streamlit GNN Training Dashboard와 live 10-fold monitor

## 2. Motivation

신약개발 초기 단계에서는 후보 분자의 효능뿐 아니라 용해도, 지용성, BBB 투과성, hERG risk, AMES mutagenicity 같은 ADMET risk를 빠르게 확인해야 합니다. 하지만 본격적인 실험 검증 전에는 많은 후보를 빠르게 비교하고, 어떤 부분 구조가 문제 예측에 기여했는지 설명 가능한 형태로 확인하는 도구가 필요합니다.

이 MVP의 목표는 SOTA 성능을 주장하는 완성형 플랫폼이 아니라, 다음 흐름이 실제로 동작하는 end-to-end prototype을 만드는 것입니다.

```text
Initial SMILES
-> RDKit molecular parsing/descriptors
-> ADMET endpoint prediction
-> Abnormality gate
-> GNN/heuristic saliency
-> Scaffold gate
-> Candidate generation
-> Candidate re-evaluation
-> Top-K recommendation
```

## 3. Architecture

```text
admet_moe_optimizer/
  app.py
  requirements.txt
  requirements-tdc.txt
  scripts/
    run_10fold_gnn_cv.py
    test_*.py
  src/
    agents/
      abnormality_gate.py
      optimization_loop.py
      saliency.py
      scaffold_gate.py
      report_agent.py
    chemistry/
      descriptors.py
      validation.py
      scaffold.py
      visualization.py
      alerts.py
    dashboard/
      training_dashboard.py
    generation/
      generator.py
      rule_based.py
      crem_wrapper.py
    predictors/
      base.py
      dummy_predictors.py
      gnn_predictor.py
      scoring.py
    reporting/
      cv_tables.py
    training/
      dataset_loader.py
      featurizer.py
      model.py
      train.py
      cv_split.py
      cv_summary.py
      live_logging.py
      run_status.py
      metrics.py
```

Core flow:

1. `app.py`에서 Streamlit UI를 렌더링합니다.
2. `chemistry/` 모듈이 SMILES validation, descriptor 계산, scaffold 추출, 분자 이미지 생성을 담당합니다.
3. `predictors/` 모듈이 dummy heuristic predictor 또는 GNN checkpoint predictor를 제공합니다.
4. `agents/` 모듈이 abnormality 판단, saliency 설명, scaffold gate, 반복형 최적화 loop를 처리합니다.
5. `generation/` 모듈이 CReM 후보 생성 또는 rule-based fallback 후보 생성을 수행합니다.
6. `training/` 모듈이 TDC dataset loading, molecular graph featurization, GNN 학습, 10-fold CV, live logging을 담당합니다.
7. `dashboard/` 모듈이 single-run 학습 결과, 10-fold 비교 결과, live CV monitor를 시각화합니다.

GNN model variants:

- `gine`: GINEConv 기반 edge-aware molecular GNN
- `attentivefp`: GATv2Conv 기반 AttentiveFP-style baseline
- `dmpnn`: NNConv 기반 message passing baseline
- `cmpnn`: residual/complement-style message passing baseline

## 4. Tech Stack

Main stack:

- Python 3.11
- Streamlit
- RDKit
- PyTorch
- PyTorch Geometric
- pandas / numpy
- scikit-learn
- Pillow
- Plotly optional fallback

Training data:

- TDC ADME/Toxicity datasets
- `Solubility_AqSolDB`
- `Lipophilicity_AstraZeneca`
- `BBB_Martins`
- `hERG_Karim`
- `AMES`

Optional:

- CReM for fragment-based molecular mutation
- `streamlit-autorefresh` for smoother dashboard refresh

## 5. Results

This repository focuses on a reproducible MVP workflow rather than claiming experimentally validated drug safety.

Implemented outputs:

- Original molecule descriptor table
- 5-endpoint ADMET prediction table
- Abnormal endpoint list
- Endpoint-specific atom saliency visualization
- Scaffold/R-group editability judgment
- Generated candidate table
- Original vs candidate property delta table
- Top-K multi-objective candidate ranking
- Auto-generated report text
- GNN training metric curves
- 10-fold CV comparison table
- Live fold/epoch training monitor

Training artifacts are written locally:

```text
checkpoints/{dataset_name}/best.pt
checkpoints/{dataset_name}/config.json
checkpoints/{dataset_name}/metrics.json
```

10-fold CV artifacts are also local and intentionally excluded from Git:

```text
checkpoints_cv/{dataset_name}/{model_type}/run_status.json
checkpoints_cv/{dataset_name}/{model_type}/fold_{fold}/live_metrics.jsonl
checkpoints_cv/{dataset_name}/{model_type}/fold_{fold}/metrics.json
checkpoints_cv/{dataset_name}/{model_type}/fold_{fold}/best.pt
checkpoints_cv/{dataset_name}/{model_type}/fold_{fold}/config.json
checkpoints_cv/{dataset_name}/{model_type}/cv_summary.json
experiments/gnn_10fold_comparison.csv
```

`.gitignore` excludes model checkpoints and generated CV results:

```gitignore
checkpoints/
checkpoints_cv/
*.pt
*.pth
*.ckpt
experiments/gnn_10fold_comparison.csv
```

## 6. How to Run

### 6.1 Create Environment

Windows에서는 RDKit을 conda로 설치하는 편이 안정적입니다.

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

CPU 환경이라면 PyTorch 공식 안내에 맞는 CPU wheel을 설치해도 됩니다.

### 6.2 Install Dependencies

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

### 6.3 Run Streamlit App

```powershell
streamlit run app.py
```

Sidebar의 `페이지`에서 다음 화면을 선택할 수 있습니다.

- `Molecule Optimizer`: 분자 최적화 workflow
- `GNN Training Dashboard`: GNN 학습 결과, checkpoint 상태, 10-fold CV, live monitor

`Molecule Optimizer`의 `Predictor mode`:

- `Dummy / Heuristic`
- `GNN Checkpoint`
- `Auto best per endpoint`

GNN checkpoint가 없는 endpoint는 앱이 중단되지 않고 dummy predictor로 fallback됩니다.

### 6.4 Train Single GNN Expert

빠른 smoke test는 epoch를 5-10 정도로 낮춰 실행할 수 있습니다.

```powershell
python -m src.training.train --dataset Solubility_AqSolDB --task regression --epochs 10
python -m src.training.train --dataset Lipophilicity_AstraZeneca --task regression --epochs 10
python -m src.training.train --dataset BBB_Martins --task classification --epochs 10
python -m src.training.train --dataset hERG_Karim --task classification --epochs 10
python -m src.training.train --dataset AMES --task classification --epochs 10
```

### 6.5 Run 10-Fold CV Benchmark

빠른 smoke test:

```powershell
python scripts/run_10fold_gnn_cv.py --dataset Solubility_AqSolDB --model gine --epochs 1 --device cuda --limit 200 --skip-existing
```

단일 dataset/model 10-fold 실행:

```powershell
python scripts/run_10fold_gnn_cv.py --dataset Solubility_AqSolDB --model gine --epochs 10 --device cuda --skip-existing
```

전체 benchmark 실행:

```powershell
python scripts/run_10fold_gnn_cv.py --all --epochs 10 --device cuda --skip-existing
```

CPU 실행:

```powershell
python scripts/run_10fold_gnn_cv.py --dataset BBB_Martins --model cmpnn --epochs 5 --device cpu --skip-existing
```

주요 옵션:

- `--device auto|cuda|cpu`
- `--split-type scaffold|random|stratified`
- `--num-folds 10`
- `--limit 200`
- `--skip-existing`
- `--continue-on-error`

### 6.6 Monitor Training in Dashboard

Streamlit app에서 `GNN Training Dashboard`를 열면 다음을 확인할 수 있습니다.

- Single-run checkpoint/config/metrics status
- Train loss / valid loss curve
- Regression metrics: MAE, RMSE, R2
- Classification metrics: AUROC, AUPRC, F1, Accuracy
- 10-fold CV comparison table
- Fold별 completed test metrics
- Live `run_status.json` status card
- Live `live_metrics.jsonl` epoch curve
- Interim mean ± std
- Final `cv_summary.json`

Dashboard의 `Start 10-fold CV Run` 버튼은 학습을 별도 Python process로 실행합니다. Streamlit app은 직접 학습 loop를 소유하지 않고 checkpoint/log 파일을 읽어 모니터링합니다.

### 6.7 TDC Download Troubleshooting

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

## 7. What I Learned

이 프로젝트를 통해 다음 설계 포인트를 정리했습니다.

- 신약개발 MVP에서는 모델 성능만큼 end-to-end workflow의 안정성이 중요합니다.
- GNN checkpoint가 없을 때도 dummy predictor로 fallback하면 시연 가능한 UX를 유지할 수 있습니다.
- ADMET 예측은 “안전한 약물” 판정이 아니라 risk prioritization으로 표현해야 합니다.
- Saliency는 인과관계 확정이 아니라 후보 수정 방향을 제안하는 설명 보조 도구입니다.
- 10-fold CV 결과는 단일 split보다 모델 비교를 설득력 있게 보여주지만, scaffold split과 데이터 누수 방지가 중요합니다.
- Streamlit app이 학습 loop를 직접 소유하지 않고 file-based monitoring을 수행하면 긴 학습 중에도 UI 안정성을 유지할 수 있습니다.
- Windows 환경에서는 RDKit, PyTorch, PyTorch Geometric, PyTDC 의존성 조합을 명확히 분리해 안내하는 것이 필요합니다.

## Disclaimer

This MVP provides heuristic or learned ADMET risk estimates for proposal/demo use only. It does not claim that any molecule is a safe drug and does not replace experimental validation, clinical evidence, or expert medicinal chemistry review.
