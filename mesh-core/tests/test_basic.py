import pytest
import numpy as np
from mesh_core.structures import MeshGraphData


def test_mesh_graph_data() -> None:
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    elements = np.array([[0, 1, 2]])
    mesh = MeshGraphData(nodes=nodes, elements=elements)
    assert mesh.num_nodes == 3
    assert mesh.num_elements == 1
