import pytest
import torch
from mesh_ml.models import PhysicsInformedGNN


def test_physics_informed_gnn() -> None:
    num_features = 8
    hidden_channels = 16
    model = PhysicsInformedGNN(num_features, hidden_channels)

    num_nodes = 5
    num_edges = 6
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
    edge_attr = torch.randn(num_edges, 2)

    out = model(x, edge_index, edge_attr)
    assert out.size(0) == num_nodes
