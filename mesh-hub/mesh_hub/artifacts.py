import torch
import os

import logging
from typing import Any, Type, Dict

logger = logging.getLogger(__name__)


class ModelHub:
    """Basic hub for loading and saving models."""

    @staticmethod
    def load_model(
        path: str,
        model_class: Type[torch.nn.Module],
        device: torch.device,
        **model_kwargs: Any,
    ) -> torch.nn.Module:
        """Loads a model from a local file path."""
        logger.info(f"Loading model from {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        model = model_class(**model_kwargs)
        # Using weights_only=True to address security warnings for torch.load
        state_dict = torch.load(path, map_location=device, weights_only=True)

        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)

        model.to(device)
        model.eval()
        return model

    @staticmethod
    def save_model(model: torch.nn.Module, path: str) -> None:
        """Saves a model to a local file path."""
        logger.info(f"Saving model to {path}")
        torch.save(model.state_dict(), path)
