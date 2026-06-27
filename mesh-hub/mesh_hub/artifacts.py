import torch
import os

class ModelHub:
    """Basic hub for loading and saving models."""

    @staticmethod
    def load_model(path: str, model_class, device, **model_kwargs):
        """Loads a model from a local file path."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        model = model_class(**model_kwargs)
        # Using weights_only=True to address security warnings for torch.load
        state_dict = torch.load(path, map_location=device, weights_only=True)

        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)

        model.to(device)
        model.eval()
        return model

    @staticmethod
    def save_model(model, path: str):
        """Saves a model to a local file path."""
        torch.save(model.state_dict(), path)
