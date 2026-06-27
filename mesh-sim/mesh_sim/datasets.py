import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from typing import Optional, Callable, List

class WearMeshDataset(Dataset):
    """
    Dataset for loading structural mesh data for wear prediction.
    Expects data in the format defined in the README.
    """
    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._find_samples()

    def _find_samples(self) -> List[str]:
        """Find all sample directories in the root_dir."""
        if not os.path.exists(self.root_dir):
            return []

        samples = []
        for d in os.listdir(self.root_dir):
            sample_dir = os.path.join(self.root_dir, d)
            if os.path.isdir(sample_dir):
                # Verify required files exist
                required_files = ['nodes.csv', 'elements.csv', 'properties.csv', 'wear.csv']
                if all(os.path.exists(os.path.join(sample_dir, f)) for f in required_files):
                    samples.append(sample_dir)

        return sorted(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        sample_dir = self.samples[idx]

        # Load data
        nodes_df = pd.read_csv(os.path.join(sample_dir, 'nodes.csv'))
        elements_df = pd.read_csv(os.path.join(sample_dir, 'elements.csv'))
        props_df = pd.read_csv(os.path.join(sample_dir, 'properties.csv'))
        wear_df = pd.read_csv(os.path.join(sample_dir, 'wear.csv'))

        # Node features (coordinates)
        pos = nodes_df[['x', 'y', 'z']].values
        n_nodes = len(pos)

        # Physical properties (broadcast to all nodes for simplicity, or load per node if available)
        # For simplicity, we assume properties are scalar values for the whole mesh
        # If they vary per node, this would be updated
        thickness = float(props_df['thickness'].iloc[0])
        youngs = float(props_df['youngs_modulus'].iloc[0])
        density = float(props_df['density'].iloc[0])

        # We might have roller path distance per node, or just a scalar number of paths
        if 'roller_dist' in props_df.columns:
            roller_dist = props_df['roller_dist'].values
        else:
            roller_dist = np.random.exponential(scale=0.5, size=n_nodes)

        # Combine node features
        x = np.zeros((n_nodes, 7))
        x[:, :3] = pos
        x[:, 3] = thickness
        x[:, 4] = youngs
        x[:, 5] = density
        x[:, 6] = roller_dist

        x_tensor = torch.tensor(x, dtype=torch.float)

        # Create edges from elements (assuming triangles or quads)
        edge_index = []
        edge_attr = []

        for _, row in elements_df.iterrows():
            # Connect all nodes in the element to each other (clique)
            elem_nodes = [int(v) for k, v in row.items() if k.startswith('node_')]
            elem_nodes = [n for n in elem_nodes if not pd.isna(n)]

            for i in range(len(elem_nodes)):
                for j in range(len(elem_nodes)):
                    if i != j:
                        n1, n2 = elem_nodes[i], elem_nodes[j]
                        dist = np.linalg.norm(pos[n1] - pos[n2])
                        edge_index.append([n1, n2])

                        # Edge features: [distance, stiffness proxy]
                        stiffness = youngs * thickness / (dist + 1e-6)
                        edge_attr.append([dist, stiffness])

        # Remove duplicates
        if edge_index:
            edge_index_np = np.array(edge_index)
            edge_attr_np = np.array(edge_attr)

            # Find unique edges
            unique_edges, indices = np.unique(edge_index_np, axis=0, return_index=True)
            edge_index_tensor = torch.tensor(unique_edges, dtype=torch.long).t().contiguous()
            edge_attr_tensor = torch.tensor(edge_attr_np[indices], dtype=torch.float)
        else:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
            edge_attr_tensor = torch.empty((0, 2), dtype=torch.float)

        # Target wear values
        wear = wear_df['wear'].values
        y_tensor = torch.tensor(wear, dtype=torch.float).view(-1, 1)

        data = Data(x=x_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor, y=y_tensor)

        if self.transform is not None:
            data = self.transform(data)

        return data
