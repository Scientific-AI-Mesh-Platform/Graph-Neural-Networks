import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

import logging

logger = logging.getLogger(__name__)


class CustomWearMessagePassingLayer(MessagePassing):
    """
    Custom Message Passing layer that incorporates physics-aware edge attributes
    into the message passing scheme.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        logger.debug(
            f"Initializing CustomWearMessagePassingLayer with {in_channels}->{out_channels}"
        )
        # We use 'add' aggregation for physical summation of forces/wear
        super(CustomWearMessagePassingLayer, self).__init__(aggr="add")

        self.lin_node = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(2, out_channels)  # Edge features: distance, stiffness

        # Combine node and edge features
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        # Add self-loops to the graph
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Create self-loop edge attributes (zeros for distance and stiffness)
        self_loop_attr = torch.zeros((x.size(0), 2), device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)

        # Linearly transform node features
        x = self.lin_node(x)

        # Compute normalization (degree)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Start message passing
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)

    def message(
        self, x_j: torch.Tensor, edge_attr: torch.Tensor, norm: torch.Tensor
    ) -> torch.Tensor:
        # Process edge attributes
        edge_features = self.lin_edge(edge_attr)

        # Combine neighbor node features with edge features
        combined = torch.cat([x_j, edge_features], dim=-1)

        # Pass through MLP
        out = self.update_mlp(combined)

        # Apply normalization
        return norm.view(-1, 1) * out
