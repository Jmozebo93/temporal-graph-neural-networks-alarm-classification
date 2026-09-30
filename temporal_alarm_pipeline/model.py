from __future__ import annotations

# Temporal GNN model definition.
# This model combines node features with positional encodings and uses graph
# attention layers to reason over the temporal event graph.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool


class TemporalEventGNN(nn.Module):
    # Graph attention network for temporal alarm classification.
    # Each graph represents a window of events; the model aggregates node
    # embeddings into a graph-level representation used for multiclass prediction.
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, pos_dim: int = 16, heads: int = 4, dropout: float = 0.25):
        super().__init__()
        self.pos_dim = pos_dim
        self.dropout = dropout

        self.input_proj = nn.Linear(input_dim + pos_dim, hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _positional_encoding(self, positions: torch.Tensor) -> torch.Tensor:
        # Encode relative event positions so the model can capture temporal order.
        if positions.dim() == 1:
            positions = positions.unsqueeze(1)

        half_dim = self.pos_dim // 2
        if half_dim == 0:
            return torch.zeros(positions.size(0), 0, device=positions.device)

        div_term = torch.exp(
            torch.arange(0, half_dim, device=positions.device, dtype=torch.float32)
            * -(math.log(10000.0) / max(half_dim, 1))
        )
        angles = positions * div_term
        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if pe.size(1) < self.pos_dim:
            pad = torch.zeros(pe.size(0), self.pos_dim - pe.size(1), device=positions.device)
            pe = torch.cat([pe, pad], dim=1)
        return pe

    def forward(self, data: Data) -> torch.Tensor:
        # Project features into hidden space, apply graph attention over the event
        # graph, and classify the window with a graph-level head.
        x = data.x
        pos = getattr(data, "position", torch.zeros(x.size(0), 1, device=x.device))
        if not torch.is_tensor(pos):
            pos = torch.tensor(pos, dtype=torch.float32, device=x.device)
        pos = pos.to(x.device)

        x = torch.cat([x, self._positional_encoding(pos)], dim=1)
        h = F.relu(self.input_proj(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.gat1(h, data.edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.gat2(h, data.edge_index))
        graph_embedding = global_mean_pool(h, data.batch)
        return self.classifier(graph_embedding)
