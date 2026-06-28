import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from torch_geometric.loader import DataLoader

# Add root directory to path to allow imports from other modular packages

from mesh_sim.generation import generate_samples
from mesh_ml.models import PhysicsInformedGNN
from mesh_ml.training import train_model, evaluate
from mesh_vis.plotting import (
    plot_training_history,
    plot_mesh_sample,
    plot_predictions,
    plot_actual_vs_predicted,
)

st.title("⚙️ GNN Wear Prediction — Live Demo (Modular Architecture)")
st.markdown("""
    This demo uses a **Physics-Informed Graph Neural Network** to predict material wear
    at every node in a structural mesh. This version runs on the newly re-architected modular ecosystem.
    """)

with st.sidebar:
    st.header("🔧 Configuration")

    st.subheader("Data")
    num_samples = st.slider("Number of mesh samples", 10, 80, 30, step=5)
    noise_level = st.slider("Noise level", 0.0, 0.5, 0.1, step=0.05)

    st.subheader("Model")
    hidden_channels = st.selectbox("Hidden channels", [16, 32, 64], index=1)

    st.subheader("Training")
    num_epochs = st.slider("Epochs", 5, 100, 30, step=5)
    lr = st.select_slider(
        "Learning rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001
    )
    batch_size = st.selectbox("Batch size", [4, 8, 16, 32], index=1)

    run_button = st.button("▶ Run Demo", type="primary", use_container_width=True)

metrics_placeholder = st.empty()
progress_bar = st.progress(0)
status_text = st.empty()

tab_data, tab_train, tab_predict = st.tabs(
    ["📊 Sample Data", "📈 Training", "🔍 Predictions"]
)

if run_button:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    status_text.text("Generating synthetic mesh data…")
    graphs = generate_samples(num_samples=num_samples, noise_level=noise_level)
    progress_bar.progress(0.1)

    with tab_data:
        st.subheader("Sample Mesh Visualisations")
        cols = st.columns(min(3, len(graphs)))
        for col_idx, col in enumerate(cols):
            with col:
                fig = plot_mesh_sample(graphs[col_idx], title=f"Sample {col_idx + 1}")
                st.pyplot(fig)
                plt.close(fig)

    status_text.text("Preparing datasets…")
    if len(graphs) < 3:
        st.error("Not enough samples.")
        st.stop()

    train_size = int(0.7 * len(graphs))
    val_size = max(1, int(0.15 * len(graphs)))
    test_size = len(graphs) - train_size - val_size

    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size : train_size + val_size]
    test_graphs = graphs[train_size + val_size :]

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    progress_bar.progress(0.2)

    status_text.text("Training model…")
    num_node_features = train_graphs[0].x.size(1)
    model = PhysicsInformedGNN(num_node_features, hidden_channels).to(device)

    history = train_model(
        model, train_loader, val_loader, device, num_epochs=num_epochs, lr=lr
    )
    progress_bar.progress(0.8)

    status_text.text("Evaluating…")
    mse, mae, rmse = evaluate(model, test_loader, device)

    progress_bar.progress(1.0)
    status_text.success("✅ Training complete!")

    with metrics_placeholder.container():
        st.subheader("📋 Test-Set Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("MSE", f"{mse:.6f}")
        c2.metric("MAE", f"{mae:.6f}")
        c3.metric("RMSE", f"{rmse:.6f}")

    with tab_train:
        st.subheader("Loss Curves")
        fig_hist = plot_training_history(history)
        st.pyplot(fig_hist)
        plt.close(fig_hist)

    with tab_predict:
        st.subheader("Wear Predictions on Test Data")
        if len(test_graphs) > 0:
            fig_pred = plot_predictions(model, test_loader, device)
            st.pyplot(fig_pred)
            plt.close(fig_pred)

            st.subheader("Actual vs Predicted")
            fig_scatter = plot_actual_vs_predicted(model, test_loader, device)
            st.pyplot(fig_scatter)
            plt.close(fig_scatter)
        else:
            st.info("Not enough test data to plot predictions.")
