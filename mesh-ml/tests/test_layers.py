import pytest
import torch
from mesh_ml.layers import CustomWearMessagePassingLayer


def test_custom_layer_forward() -> None:
    in_channels = 8
    out_channels = 16
    layer = CustomWearMessagePassingLayer(in_channels, out_channels)

    num_nodes = 5
    num_edges = 6
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
    edge_attr = torch.randn(num_edges, 2)

    out = layer(x, edge_index, edge_attr)
    assert out.size(0) == num_nodes
    assert out.size(1) == out_channels
