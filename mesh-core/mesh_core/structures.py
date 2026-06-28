import logging
from typing import Optional, List, Dict, Any, Union
import numpy as np
import torch

logger = logging.getLogger(__name__)


class MeshGraphData:
    """
    Core data structure representing a structural mesh.
    Provides standard interfaces for querying adjacency, curvature, and physical properties.
    """

    def __init__(
        self,
        nodes: Union[np.ndarray, torch.Tensor],
        elements: Union[np.ndarray, torch.Tensor],
        thickness: Optional[Union[np.ndarray, torch.Tensor]] = None,
        youngs_modulus: Optional[Union[np.ndarray, torch.Tensor]] = None,
        density: Optional[Union[np.ndarray, torch.Tensor]] = None,
        roller_paths: Optional[Union[np.ndarray, torch.Tensor]] = None,
        wear: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> None:
        logger.debug(
            f"Initializing MeshGraphData with {len(nodes)} nodes and {len(elements)} elements"
        )
        self.nodes = nodes
        self.elements = elements
        self.thickness = thickness
        self.youngs_modulus = youngs_modulus
        self.density = density
        self.roller_paths = roller_paths
        self.wear = wear

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the mesh."""
        return len(self.nodes)

    @property
    def num_elements(self) -> int:
        """Return the number of elements in the mesh."""
        return len(self.elements)
