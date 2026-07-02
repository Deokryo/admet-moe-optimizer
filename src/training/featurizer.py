"""RDKit-to-PyTorch-Geometric molecular graph featurization."""

from __future__ import annotations

from typing import Iterable

from rdkit import Chem


HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
CHIRAL_TAGS = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
STEREO_TYPES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]

ATOM_FEATURE_DIM = 5 + len(HYBRIDIZATIONS) + len(CHIRAL_TAGS)
BOND_FEATURE_DIM = len(BOND_TYPES) + 2 + len(STEREO_TYPES)


def _one_hot(value: object, choices: Iterable[object]) -> list[float]:
    """Return a one-hot vector with an all-zero fallback for unknown values."""
    return [1.0 if value == choice else 0.0 for choice in choices]


def atom_features(atom: Chem.Atom) -> list[float]:
    """Create numeric atom features."""
    return [
        float(atom.GetAtomicNum()) / 100.0,
        float(atom.GetTotalDegree()) / 6.0,
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs()) / 4.0,
        *_one_hot(atom.GetHybridization(), HYBRIDIZATIONS),
        float(atom.GetIsAromatic()),
        *_one_hot(atom.GetChiralTag(), CHIRAL_TAGS),
    ]


def bond_features(bond: Chem.Bond) -> list[float]:
    """Create numeric bond features."""
    return [
        *_one_hot(bond.GetBondType(), BOND_TYPES),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        *_one_hot(bond.GetStereo(), STEREO_TYPES),
    ]


def smiles_to_data(smiles: str, y: float | None = None):
    """Convert a SMILES string to a PyG Data object, returning None if invalid."""
    try:
        import torch
        from torch_geometric.data import Data
    except Exception as exc:
        raise ImportError("torch and torch_geometric are required for graph featurization.") from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)
    edge_indices: list[list[int]] = []
    edge_attrs: list[list[float]] = []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        features = bond_features(bond)
        edge_indices.extend([[start, end], [end, start]])
        edge_attrs.extend([features, features])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
    if y is not None:
        data.y = torch.tensor([float(y)], dtype=torch.float)
    return data


def dataframe_to_graphs(frame) -> list:
    """Convert a standardized smiles/y DataFrame to PyG graphs, skipping invalid SMILES."""
    graphs = []
    skipped = 0
    for row in frame.itertuples(index=False):
        data = smiles_to_data(row.smiles, row.y)
        if data is None:
            skipped += 1
            continue
        graphs.append(data)
    if skipped:
        print(f"Skipped {skipped} invalid SMILES during featurization.")
    return graphs
