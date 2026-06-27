import torch
import numpy as np
from torch_geometric.data import Data
from typing import List, Tuple

def generate_samples(
    num_samples: int = 30,
    n_nodes_range: Tuple[int, int] = (30, 81),
    thickness_range: Tuple[float, float] = (0.5, 2.0),
    youngs_range: Tuple[float, float] = (1e5, 5e5),
    density_range: Tuple[float, float] = (2000.0, 8000.0),
    roller_paths_range: Tuple[int, int] = (1, 11),
    noise_level: float = 0.1,
    seed: int = 42
) -> List[Data]:
    """Generates synthetic mesh graph samples simulating material wear."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    graphs = []

    for _ in range(num_samples):
        n_nodes = np.random.randint(n_nodes_range[0], n_nodes_range[1])

        # 1. Random 3D node positions [0, 1]
        pos = np.random.rand(n_nodes, 3)

        # 2. Material properties
        thickness = np.random.uniform(thickness_range[0], thickness_range[1])
        youngs = np.random.uniform(youngs_range[0], youngs_range[1])
        density = np.random.uniform(density_range[0], density_range[1])
        roller_paths = np.random.randint(roller_paths_range[0], roller_paths_range[1])

        # Simulated distance to roller path for each node
        roller_dist = np.random.exponential(scale=0.5, size=(n_nodes, 1))

        # Combine node features: [x, y, z, thickness, youngs, density, roller_dist]
        x = np.zeros((n_nodes, 7))
        x[:, :3] = pos
        x[:, 3] = thickness
        x[:, 4] = youngs
        x[:, 5] = density
        x[:, 6] = roller_dist.flatten()

        x_tensor = torch.tensor(x, dtype=torch.float)

        # 3. Create edges (k-nearest neighbors proxy for Delaunay triangulation)
        # For simplicity in synthetic data, we connect nodes if distance < threshold
        edge_index = []
        edge_attr = []

        threshold = 0.4
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    dist = np.linalg.norm(pos[i] - pos[j])
                    if dist < threshold:
                        edge_index.append([i, j])
                        # Edge features: [distance, stiffness proxy]
                        stiffness = youngs * thickness / (dist + 1e-6)
                        edge_attr.append([dist, stiffness])

        if not edge_index:
            # Fallback if no edges found: connect to a random node
            for i in range(n_nodes):
                j = (i + 1) % n_nodes
                dist = np.linalg.norm(pos[i] - pos[j])
                edge_index.append([i, j])
                edge_index.append([j, i])
                stiffness = youngs * thickness / (dist + 1e-6)
                edge_attr.extend([[dist, stiffness], [dist, stiffness]])

        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)

        # 4. Generate target wear values (synthetic physics rule)
        # Wear increases with roller_paths, decreases with thickness and youngs modulus
        # Wear increases closer to roller path (smaller roller_dist)
        base_wear = (roller_paths * 0.1) / (thickness * (youngs / 1e5))
        dist_factor = np.exp(-roller_dist.flatten())

        wear_true = base_wear * dist_factor

        # Add spatial non-linearity based on z-coordinate
        wear_true *= (1.0 + 0.5 * np.sin(pos[:, 2] * np.pi))

        # Add noise
        noise = np.random.normal(0, noise_level * np.mean(wear_true), size=n_nodes)
        wear = np.clip(wear_true + noise, a_min=0.0, a_max=None)

        y_tensor = torch.tensor(wear, dtype=torch.float).view(-1, 1)

        # Construct PyTorch Geometric Data object
        data = Data(x=x_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor, y=y_tensor)
        graphs.append(data)

    return graphs
