"""
Live Demo: Graph Neural Network for Structural Wear Prediction
Run with: streamlit run app.py
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st
from sklearn.preprocessing import StandardScaler

# ── Make GNN.py importable without triggering its __main__ block ──────────────
sys.path.insert(0, os.path.dirname(__file__))

from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import MessagePassing

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GNN Wear Prediction — Live Demo",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Model / helper code (self-contained copy so we avoid GNN.py side-effects) ─

class EdgeFeatureExtractor:
    def compute_edge_features(self, node_positions, elements, thickness, youngs_modulus):
        edge_list = []
        try:
            if isinstance(elements, np.ndarray):
                elements = elements.tolist()
            for elem in elements:
                if not hasattr(elem, "__iter__"):
                    continue
                for i in range(len(elem)):
                    for j in range(i + 1, len(elem)):
                        if elem[i] < node_positions.shape[0] and elem[j] < node_positions.shape[0]:
                            edge_list.append([elem[i], elem[j]])
                            edge_list.append([elem[j], elem[i]])
        except Exception:
            for i in range(node_positions.shape[0] - 1):
                edge_list.append([i, i + 1])
                edge_list.append([i + 1, i])

        edge_list = np.array(edge_list)
        edge_index = torch.tensor(edge_list.T, dtype=torch.long)

        start_pos = node_positions[edge_index[0]]
        end_pos = node_positions[edge_index[1]]
        distances = torch.norm(end_pos - start_pos, dim=1, keepdim=True)

        edge_thickness = torch.ones(edge_index.shape[1], 1) * float(
            thickness.mean() if isinstance(thickness, torch.Tensor) else thickness
        )
        edge_youngs = torch.ones(edge_index.shape[1], 1) * float(
            youngs_modulus.mean() if isinstance(youngs_modulus, torch.Tensor) else youngs_modulus
        )
        edge_attr = torch.cat([distances, edge_thickness, edge_youngs], dim=1)
        return edge_index, edge_attr


class MeshGraphDataset:
    def __init__(self):
        self.scaler = StandardScaler()
        self.edge_extractor = EdgeFeatureExtractor()

    def create_graph(self, nodes, elements, thickness, youngs_modulus, density, roller_paths, wear_values=None):
        nodes_tensor = torch.tensor(nodes, dtype=torch.float)
        if not isinstance(thickness, torch.Tensor):
            thickness = torch.tensor(thickness, dtype=torch.float)
        if not isinstance(youngs_modulus, torch.Tensor):
            youngs_modulus = torch.tensor(youngs_modulus, dtype=torch.float)
        if not isinstance(density, torch.Tensor):
            density = torch.tensor(float(density), dtype=torch.float)

        edge_index, edge_attr = self.edge_extractor.compute_edge_features(
            nodes_tensor, elements, thickness, youngs_modulus
        )

        edge_index_sl, _ = add_self_loops(edge_index)
        node_degree = degree(edge_index_sl[0], num_nodes=len(nodes))

        x = torch.cat([
            nodes_tensor,
            node_degree.view(-1, 1),
            torch.ones(len(nodes), 1) * density,
            torch.ones(len(nodes), 1) * roller_paths,
        ], dim=1)

        x_scaled = self.scaler.fit_transform(x.numpy())
        x = torch.tensor(x_scaled, dtype=torch.float)

        if wear_values is not None:
            y = torch.tensor(wear_values, dtype=torch.float).view(-1, 1)
            graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        else:
            graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        graph_data.num_nodes = len(nodes)
        return graph_data


class EdgeAttentionLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels, out_channels)
        self.att = nn.Linear(2 * out_channels + 3, 1)

    def forward(self, x, edge_index, edge_attr):
        x = self.lin(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        alpha = self.att(torch.cat([x_i, x_j, edge_attr], dim=-1))
        return x_j * torch.sigmoid(F.leaky_relu(alpha))

    def update(self, aggr_out):
        return aggr_out


class WearPredictionGNN(nn.Module):
    def __init__(self, node_features, edge_features, hidden_channels, num_conv_layers=3, dropout=0.3):
        super().__init__()
        self.num_conv_layers = num_conv_layers
        self.node_encoder = nn.Linear(node_features, hidden_channels)
        self.edge_encoder = nn.Linear(edge_features, hidden_channels)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.convs.append(EdgeAttentionLayer(hidden_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        for i in range(1, num_conv_layers):
            if i % 3 == 0:
                self.convs.append(GATv2Conv(hidden_channels, hidden_channels, heads=2, concat=False, edge_dim=edge_features))
            elif i % 3 == 1:
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            else:
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout
        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        x = self.node_encoder(x)
        for i in range(self.num_conv_layers):
            identity = x
            if i == 0:
                x = self.convs[i](x, edge_index, edge_attr)
            elif isinstance(self.convs[i], GATv2Conv):
                x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            else:
                x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if x.shape == identity.shape:
                x = x + identity
        return self.regressor(x)


def generate_samples(num_samples, n_nodes_range, thickness_range, youngs_range,
                     density_range, roller_paths_range, noise_level):
    np.random.seed(42)
    graphs = []
    dataset = MeshGraphDataset()

    for _ in range(num_samples):
        n_nodes = np.random.randint(*n_nodes_range)
        nodes = np.random.rand(n_nodes, 3) * 10
        elements = [[j, j + 1, j + 2] for j in range(0, n_nodes - 2, 3) if j + 2 < n_nodes]

        thickness = np.random.uniform(*thickness_range, len(elements)) if elements else np.array([thickness_range[0]])
        youngs_modulus = np.random.uniform(*youngs_range, len(elements)) if elements else np.array([youngs_range[0]])
        density = np.random.uniform(*density_range)
        roller_paths = np.random.randint(*roller_paths_range)

        base_wear = np.sum(np.abs(nodes - nodes.mean(axis=0)), axis=1) / 10
        base_wear *= 1 + 0.2 * roller_paths
        for j, elem in enumerate(elements):
            wear_factor = 1 / (thickness[j] * youngs_modulus[j] * 1e-6)
            for nidx in elem:
                base_wear[nidx] += wear_factor * 0.1
        base_wear *= density / 5000
        wear = np.maximum(base_wear + np.random.normal(0, noise_level * base_wear.mean(), n_nodes), 0.01)

        graph = dataset.create_graph(
            nodes=nodes, elements=elements,
            thickness=torch.tensor(thickness, dtype=torch.float),
            youngs_modulus=torch.tensor(youngs_modulus, dtype=torch.float),
            density=density, roller_paths=roller_paths, wear_values=wear,
        )
        graphs.append(graph)
    return graphs


def train_model(graphs, hidden_channels, num_conv_layers, dropout, num_epochs, lr, batch_size,
                progress_bar, status_text):
    device = torch.device("cpu")

    # Split
    n = len(graphs)
    idx = list(range(n))
    np.random.shuffle(idx)
    t_end = int(0.7 * n)
    v_end = int(0.85 * n)
    train_loader = DataLoader([graphs[i] for i in idx[:t_end]], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader([graphs[i] for i in idx[t_end:v_end]], batch_size=batch_size)
    test_loader = DataLoader([graphs[i] for i in idx[v_end:]], batch_size=batch_size)

    # Ensure there's at least one sample per split
    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0 or len(test_loader.dataset) == 0:
        # Fall back: use all data for all splits when very few samples
        train_loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(graphs, batch_size=batch_size)
        test_loader = DataLoader(graphs, batch_size=batch_size)

    sample = next(iter(train_loader))
    node_features = sample.x.size(1)
    edge_features = sample.edge_attr.size(1)

    model = WearPredictionGNN(node_features, edge_features, hidden_channels, num_conv_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        t_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * data.num_graphs
        t_loss /= len(train_loader.dataset)

        # Validate
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                v_loss += criterion(out, data.y).item() * data.num_graphs
        v_loss /= len(val_loader.dataset)
        scheduler.step(v_loss)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        progress_bar.progress(epoch / num_epochs)
        status_text.text(f"Epoch {epoch}/{num_epochs} — Train Loss: {t_loss:.5f}  Val Loss: {v_loss:.5f}")

    # Evaluate on test set
    model.eval()
    preds_all, targets_all = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            preds_all.append(out.cpu())
            targets_all.append(data.y.cpu())

    preds = torch.cat(preds_all)
    targets = torch.cat(targets_all)
    mse = F.mse_loss(preds, targets).item()
    mae = F.l1_loss(preds, targets).item()
    rmse = float(torch.sqrt(torch.tensor(mse)))

    return model, history, {"mse": mse, "mae": mae, "rmse": rmse}, test_loader, device


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_training_history(history):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(history["val_loss"], label="Val Loss", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_mesh_sample(graph, title="Sample Mesh (coloured by wear)"):
    """2-D scatter of node positions coloured by wear values."""
    node_pos = graph.x[:, :2].numpy()
    wear = graph.y.numpy().flatten() if graph.y is not None else np.zeros(graph.num_nodes)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(node_pos[:, 0], node_pos[:, 1], c=wear, cmap="viridis", s=40, alpha=0.85)
    plt.colorbar(sc, ax=ax, label="Wear")
    ax.set_title(title)
    ax.set_xlabel("X (normalised)")
    ax.set_ylabel("Y (normalised)")
    plt.tight_layout()
    return fig


def plot_predictions(model, loader, device, max_nodes=500):
    """Actual vs Predicted scatter + error map on the first batch."""
    model.eval()
    batch = next(iter(loader)).to(device)
    with torch.no_grad():
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

    actual = batch.y.cpu().numpy().flatten()
    predicted = pred.cpu().numpy().flatten()
    node_pos = batch.x[:, :2].cpu().numpy()

    # Trim for readability
    if len(actual) > max_nodes:
        idx = np.random.choice(len(actual), max_nodes, replace=False)
        actual, predicted, node_pos = actual[idx], predicted[idx], node_pos[idx]

    error = np.abs(actual - predicted)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sc0 = axes[0].scatter(node_pos[:, 0], node_pos[:, 1], c=actual, cmap="viridis", s=35)
    plt.colorbar(sc0, ax=axes[0], label="Wear")
    axes[0].set_title("Actual Wear")

    sc1 = axes[1].scatter(node_pos[:, 0], node_pos[:, 1], c=predicted, cmap="viridis", s=35)
    plt.colorbar(sc1, ax=axes[1], label="Wear")
    axes[1].set_title("Predicted Wear")

    sc2 = axes[2].scatter(node_pos[:, 0], node_pos[:, 1], c=error, cmap="Reds", s=35)
    plt.colorbar(sc2, ax=axes[2], label="Error")
    axes[2].set_title("Absolute Error")

    for ax in axes:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.tight_layout()
    return fig


def plot_actual_vs_predicted(model, loader, device):
    """Scatter plot of actual vs predicted values with identity line."""
    model.eval()
    all_actual, all_pred = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            all_actual.append(data.y.cpu().numpy().flatten())
            all_pred.append(out.cpu().numpy().flatten())

    actual = np.concatenate(all_actual)
    predicted = np.concatenate(all_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(actual, predicted, alpha=0.4, s=20, label="Predictions")
    lim = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    ax.plot(lim, lim, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Wear")
    ax.set_ylabel("Predicted Wear")
    ax.set_title("Actual vs Predicted Wear")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.title("⚙️ GNN Wear Prediction — Live Demo")
st.markdown(
    """
    This demo uses a **Physics-Informed Graph Neural Network** to predict material wear  
    at every node in a structural mesh. Adjust the parameters in the sidebar and click  
    **▶ Run Demo** to train the model and visualise results instantly.
    """
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Configuration")

    st.subheader("Data")
    num_samples = st.slider("Number of mesh samples", 10, 80, 30, step=5)
    n_nodes_min = st.slider("Min nodes per sample", 20, 60, 30, step=5)
    n_nodes_max = st.slider("Max nodes per sample", 60, 150, 80, step=10)
    noise_level = st.slider("Noise level", 0.0, 0.5, 0.1, step=0.05)

    st.subheader("Material Properties")
    thickness_min, thickness_max = st.slider("Thickness range", 0.1, 5.0, (0.5, 2.0), step=0.1)
    youngs_min, youngs_max = st.slider("Young's modulus (×10⁵ Pa)", 1, 10, (1, 5), step=1)
    density_min, density_max = st.slider("Density range (kg/m³)", 1000, 9000, (2000, 8000), step=500)
    roller_min, roller_max = st.slider("Roller paths range", 1, 15, (1, 10), step=1)

    st.subheader("Model")
    hidden_channels = st.selectbox("Hidden channels", [16, 32, 64], index=1)
    num_layers = st.selectbox("GNN layers", [2, 3, 4], index=1)
    dropout = st.slider("Dropout", 0.0, 0.5, 0.2, step=0.05)

    st.subheader("Training")
    num_epochs = st.slider("Epochs", 5, 100, 30, step=5)
    lr = st.select_slider("Learning rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
    batch_size = st.selectbox("Batch size", [4, 8, 16, 32], index=1)

    run_button = st.button("▶ Run Demo", type="primary", use_container_width=True)

# ── Main area placeholders ────────────────────────────────────────────────────
metrics_placeholder = st.empty()
progress_bar = st.progress(0)
status_text = st.empty()

tab_data, tab_train, tab_predict = st.tabs(["📊 Sample Data", "📈 Training", "🔍 Predictions"])

if run_button:
    # 1. Generate data
    status_text.text("Generating synthetic mesh data…")
    graphs = generate_samples(
        num_samples=num_samples,
        n_nodes_range=(n_nodes_min, n_nodes_max + 1),
        thickness_range=(thickness_min, thickness_max),
        youngs_range=(youngs_min * 1e5, youngs_max * 1e5),
        density_range=(density_min, density_max),
        roller_paths_range=(roller_min, roller_max + 1),
        noise_level=noise_level,
    )
    progress_bar.progress(0.05)

    # Show a sample mesh before training
    with tab_data:
        st.subheader("Sample Mesh Visualisations")
        cols = st.columns(min(3, len(graphs)))
        for col_idx, col in enumerate(cols):
            with col:
                fig = plot_mesh_sample(graphs[col_idx], title=f"Sample {col_idx + 1}")
                st.pyplot(fig)
                plt.close(fig)

        st.markdown(
            f"**{len(graphs)} samples** generated — "
            f"average {np.mean([g.num_nodes for g in graphs]):.0f} nodes/sample, "
            f"average {np.mean([g.edge_index.shape[1] for g in graphs]):.0f} edges/sample."
        )

    # 2. Train model
    status_text.text("Training model…")
    model, history, test_metrics, test_loader, device = train_model(
        graphs=graphs,
        hidden_channels=hidden_channels,
        num_conv_layers=num_layers,
        dropout=dropout,
        num_epochs=num_epochs,
        lr=lr,
        batch_size=batch_size,
        progress_bar=progress_bar,
        status_text=status_text,
    )

    # 3. Show metrics
    progress_bar.progress(1.0)
    status_text.success("✅ Training complete!")

    with metrics_placeholder.container():
        st.subheader("📋 Test-Set Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("MSE", f"{test_metrics['mse']:.6f}")
        c2.metric("MAE", f"{test_metrics['mae']:.6f}")
        c3.metric("RMSE", f"{test_metrics['rmse']:.6f}")

    # 4. Training curves
    with tab_train:
        st.subheader("Loss Curves")
        fig_hist = plot_training_history(history)
        st.pyplot(fig_hist)
        plt.close(fig_hist)

        best_epoch = int(np.argmin(history["val_loss"])) + 1
        best_val = min(history["val_loss"])
        st.info(f"Best validation loss **{best_val:.6f}** reached at epoch **{best_epoch}**.")

    # 5. Predictions
    with tab_predict:
        st.subheader("Wear Predictions on Test Data")
        fig_pred = plot_predictions(model, test_loader, device)
        st.pyplot(fig_pred)
        plt.close(fig_pred)

        st.subheader("Actual vs Predicted (all test nodes)")
        fig_scatter = plot_actual_vs_predicted(model, test_loader, device)
        st.pyplot(fig_scatter)
        plt.close(fig_scatter)

    # Store model in session state for re-use
    st.session_state["model"] = model
    st.session_state["device"] = device

else:
    with tab_data:
        st.info("👈  Adjust the parameters in the sidebar and click **▶ Run Demo** to begin.")
    with tab_train:
        st.info("Training results will appear here after clicking **▶ Run Demo**.")
    with tab_predict:
        st.info("Prediction visualisations will appear here after clicking **▶ Run Demo**.")
