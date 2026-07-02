"""Checkpoint-backed GNN predictors compatible with the Predictor interface."""

from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem

from src.predictors.base import Prediction
from src.training.featurizer import smiles_to_data
from src.training.model import MolecularGNN


DATASET_TO_ENDPOINT = {
    "Solubility_AqSolDB": ("Solubility Expert", "regression", "heuristic/learned logS", "GNN regression output"),
    "Lipophilicity_AstraZeneca": ("Lipophilicity Expert", "regression", "LogP", "GNN regression output"),
    "BBB_Martins": ("BBB Expert", "binary classification", "probability", "GNN probability estimate"),
    "hERG_Karim": ("hERG Expert", "binary classification", "risk probability", "GNN risk probability estimate"),
    "AMES": ("AMES Expert", "binary classification", "risk probability", "GNN risk probability estimate"),
}
ENDPOINT_TO_DATASET = {value[0]: key for key, value in DATASET_TO_ENDPOINT.items()}


class GNNPredictor:
    """Load a trained MolecularGNN checkpoint and predict one molecule at a time."""

    def __init__(self, dataset_name: str, checkpoint_path: str | Path, device: str | None = None) -> None:
        """Load model config and weights from checkpoint directory."""
        try:
            import torch
        except Exception as exc:
            raise ImportError("PyTorch is required for GNN checkpoint inference.") from exc

        if dataset_name not in DATASET_TO_ENDPOINT:
            raise ValueError(f"Unsupported GNN dataset: {dataset_name}")

        self.dataset_name = dataset_name
        self.name, self.task, self.unit, self.interpretation = DATASET_TO_ENDPOINT[dataset_name]
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        config_path = self.checkpoint_path.parent / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch = torch

        self.model = MolecularGNN(
            atom_feature_dim=int(self.config["atom_feature_dim"]),
            bond_feature_dim=int(self.config["bond_feature_dim"]),
            hidden_dim=int(self.config.get("hidden_dim", 128)),
            num_layers=int(self.config.get("num_layers", 3)),
            dropout=float(self.config.get("dropout", 0.1)),
            output_dim=int(self.config.get("output_dim", 1)),
        ).to(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_smiles(self, smiles: str) -> float:
        """Predict a raw regression value or probability for one SMILES."""
        data = smiles_to_data(smiles)
        if data is None:
            raise ValueError(f"Invalid SMILES for GNN inference: {smiles}")
        data = data.to(self.device)
        with self.torch.no_grad():
            output = float(self.model(data).view(-1)[0].detach().cpu().item())
        if self.task == "binary classification":
            return float(self.torch.sigmoid(self.torch.tensor(output)).item())
        return output

    def predict_many(self, smiles_list: list[str]) -> list[float]:
        """Predict values for a list of SMILES."""
        return [self.predict_smiles(smiles) for smiles in smiles_list]

    def predict(self, mol: Chem.Mol, descriptors: dict[str, float]) -> Prediction:
        """Predict an endpoint for an RDKit molecule."""
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        value = self.predict_smiles(smiles)
        return Prediction(
            endpoint=self.name,
            task=self.task,
            value=value,
            unit=self.unit,
            confidence=0.8,
            interpretation=self.interpretation,
        )

