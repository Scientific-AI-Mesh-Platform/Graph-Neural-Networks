import torch
import numpy as np

class MeshGraphData:
    """Core data structure representing a structural mesh."""
    def __init__(self, nodes, elements, thickness=None, youngs_modulus=None, density=None, roller_paths=None, wear=None):
        self.nodes = nodes
        self.elements = elements
        self.thickness = thickness
        self.youngs_modulus = youngs_modulus
        self.density = density
        self.roller_paths = roller_paths
        self.wear = wear

    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def num_elements(self):
        return len(self.elements)
