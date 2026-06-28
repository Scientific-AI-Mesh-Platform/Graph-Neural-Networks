import pytest
from mesh_vis.plotting import plot_training_history


def test_plot_training_history() -> None:
    history = {"train_loss": [0.1, 0.05], "val_loss": [0.2, 0.1]}
    fig = plot_training_history(history)
    assert fig is not None
