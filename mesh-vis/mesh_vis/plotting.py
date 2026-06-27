import matplotlib.pyplot as plt
import numpy as np
import torch

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
