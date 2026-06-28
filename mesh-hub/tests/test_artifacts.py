import pytest
import torch
import os
from mesh_hub.artifacts import ModelHub


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)


def test_model_hub(tmp_path) -> None:
    model = DummyModel()
    model_path = os.path.join(tmp_path, "model.pt")
    ModelHub.save_model(model, model_path)
    assert os.path.exists(model_path)
    loaded_model = ModelHub.load_model(model_path, DummyModel, torch.device("cpu"))
    assert isinstance(loaded_model, DummyModel)
