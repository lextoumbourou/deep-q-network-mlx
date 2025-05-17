from pathlib import Path
import mlx.nn as nn # For type hinting model argument, can be removed if not strict

def save_model(model: nn.Module, path: str):
    """Save model weights to a file."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(path_obj))
    print(f"Model saved to {path_obj}")


def load_model(model: nn.Module, path: str):
    """Load model weights from a file."""
    path_obj = Path(path)
    model.load_weights(str(path_obj))
    print(f"Model loaded from {path_obj}")
    return model 