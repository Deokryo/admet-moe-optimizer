"""GINEConv molecular GNN model."""

from __future__ import annotations


class MolecularGNN:  # pragma: no cover - real implementation is injected when torch is installed.
    """Placeholder overwritten below when PyTorch dependencies are available."""

    def __init__(self, *args, **kwargs) -> None:
        """Raise a friendly dependency error."""
        raise ImportError("torch and torch_geometric are required to instantiate MolecularGNN.")


try:
    import torch
    from torch import nn
    from torch_geometric.nn import GINEConv, global_mean_pool
except Exception:
    torch = None
else:

    class MolecularGNN(nn.Module):
        """Edge-aware molecular GNN using GINEConv layers."""

        def __init__(
            self,
            atom_feature_dim: int,
            bond_feature_dim: int,
            hidden_dim: int = 128,
            num_layers: int = 3,
            dropout: float = 0.1,
            output_dim: int = 1,
        ) -> None:
            """Initialize encoders, GINE layers, and graph-level head."""
            super().__init__()
            self.node_encoder = nn.Linear(atom_feature_dim, hidden_dim)
            self.edge_encoder = nn.Linear(bond_feature_dim, hidden_dim)
            self.convs = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            self.dropout = nn.Dropout(dropout)

            for _ in range(num_layers):
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.convs.append(GINEConv(mlp, train_eps=True))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, data):
            """Return graph-level regression values or classification logits."""
            x = self.node_encoder(data.x)
            edge_attr = self.edge_encoder(data.edge_attr)
            batch = getattr(data, "batch", None)
            if batch is None:
                batch = x.new_zeros(x.size(0), dtype=torch.long)

            for conv, batch_norm in zip(self.convs, self.batch_norms):
                residual = x
                x = conv(x, data.edge_index, edge_attr)
                x = batch_norm(x)
                x = torch.relu(x)
                x = self.dropout(x)
                x = x + residual

            graph_embedding = global_mean_pool(x, batch)
            return self.head(graph_embedding).view(-1)

