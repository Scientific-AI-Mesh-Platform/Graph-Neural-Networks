# Scientific Mesh Intelligence Ecosystem - Architectural Audit & Redesign

## 1. Architectural Audit of Existing Monolith

### Architecture Diagram
The current architecture is a highly coupled monolithic structure:
- **`app.py`**: Contains web UI (Streamlit), data generation logic, model training code, and matplotlib visualization all intertwined.
- **`GNN.py`**: A massive script containing dataset classes, custom PyTorch Geographic layers (`WearPredictionGNN`, `CustomWearMessagePassingLayer`), training loops, and evaluation/visualization tools.
- **`load_and_train.py`**: Includes `WearMeshDataset`, a slightly different GNN model `PhysicsInformedGNN`, and duplicate training loops/visualization.
- **Outputs**: Generates `.pt` files, logs, and PNG images directly in the root directory.

### Dependency Graph
- PyTorch / PyTorch Geometric
- Streamlit (tightly coupled with data/model logic in `app.py`)
- NumPy (version specific requirement < 2.0)
- Matplotlib (used for visualization across multiple files)
- Pandas, Scikit-learn

### Repository Relationships
Currently, everything resides in a single root repository, meaning there are no relationships or clear boundaries between components (Data generation, ML, Visualization, Web Interface).

### Duplicated Logic
- **Training loops**: Present in both `GNN.py` and `app.py` (via `train_model` function) and `load_and_train.py`.
- **GNN Architecture**: Multiple implementations of similar physics-informed layers in `GNN.py` and `load_and_train.py`.
- **Visualization**: `plot_predictions`, `plot_training_history`, etc., are redefined or slightly modified across `app.py`, `GNN.py`, and `load_and_train.py`.
- **Data Loading/Generation**: Synthetic data generation and custom dataset parsing logic are repeated and tightly coupled in multiple files.

### Missing Abstractions
- No independent library for managing Mesh Graph Data Structures (nodes, elements, physics parameters).
- No modular ML framework for registering and reusing Physics-Informed layers, loss functions, and models.
- No separation between frontend inference (Serving) and backend training/simulation.
- No unified visualization API (relying on scattered matplotlib calls).

### Repositories Status
- **Unnecessary repositories**: N/A (single repo).
- **Repositories that should be split**: The entire monolith must be split into specialized packages.
- **Repositories that should be merged**: N/A.
- **Common libraries**: We need a shared `mesh-core` for data structures and `mesh-ml` for GNN primitives.
- **Shared SDK opportunities**: A Python API client (`mesh-hub`) to download models/datasets.

---

## 2. Redesign: Modular Ecosystem

Following the `ECOSYSTEM_ARCHITECTURE.md`, we will reconstruct the organization from scratch into a modular multi-repository (simulated as multi-package) ecosystem.

### Core Ecosystem Components:
1. **`mesh-core`**: High-performance, standalone mesh data structures library.
2. **`mesh-ml`**: Extensible scientific GNN framework (Models, Layers, Losses).
3. **`mesh-sim`**: Synthetic data generation and simulation engine.
4. **`mesh-hub`**: Centralized model and dataset hub for sharing artifacts.
5. **`mesh-vis`**: Standalone 3D scientific visualization library.
6. **`mesh-serve`**: Scalable inference and web serving architecture.

*(See subsequent implementation steps for detailed restructuring.)*
