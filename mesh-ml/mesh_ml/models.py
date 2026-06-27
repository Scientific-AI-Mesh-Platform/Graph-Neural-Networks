import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv
from .layers import CustomWearMessagePassingLayer

class PhysicsInformedGNN(torch.nn.Module):
    """
    Graph Neural Network for predicting wear patterns with physics regularization.
    Combines custom physics-aware message passing with attention.
    """
    def __init__(self, num_node_features, hidden_channels):
        super(PhysicsInformedGNN, self).__init__()

        # First custom physics-informed message passing layer
        self.conv1 = CustomWearMessagePassingLayer(num_node_features, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        # Second custom physics-informed message passing layer
        self.conv2 = CustomWearMessagePassingLayer(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        # Third layer with attention
        self.conv3 = GATConv(hidden_channels, hidden_channels, heads=4, concat=False)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        # MLP for final prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, hidden_channels//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels//2, 1)
        )

        # Physics-informed regularization parameters
        self.stress_weight = nn.Parameter(torch.tensor(1.0))
        self.thickness_weight = nn.Parameter(torch.tensor(1.0))
        self.distance_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, edge_index, edge_attr, batch=None):
        # First layer
        h = self.conv1(x, edge_index, edge_attr)
        h = self.bn1(h)
        h = F.relu(h)

        # Second layer
        h = self.conv2(h, edge_index, edge_attr)
        h = self.bn2(h)
        h = F.relu(h)

        # Third layer
        h = self.conv3(h, edge_index)
        h = self.bn3(h)
        h = F.relu(h)

        # Final MLP
        wear = self.mlp(h).view(-1)

        return wear

    def physics_regularization(self, data, predicted_wear):
        """
        Add physics-based regularization to the loss function
        """
        x = data.x
        thickness = x[:, 3].view(-1)
        youngs_modulus = x[:, 4].view(-1)

        # 1. Wear should be inversely proportional to thickness
        thickness_reg = torch.mean(predicted_wear * thickness)

        # 2. Wear should be inversely proportional to Young's modulus
        modulus_reg = torch.mean(predicted_wear * youngs_modulus / 1e10)

        # 3. Wear should decrease with distance from roller paths
        if x.shape[1] > 6:  # If we have roller path distances
            roller_distances = x[:, 6].view(-1)
            distance_reg = -torch.mean(predicted_wear * torch.exp(-roller_distances))
        else:
            distance_reg = 0.0

        return self.thickness_weight * thickness_reg + \
               self.stress_weight * modulus_reg + \
               self.distance_weight * distance_reg
