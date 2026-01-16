"""
Model persistence utilities for sales prediction. This module handles: Saving trained models and
transformers to disk; Loading models and transformers from disk; Managing model metadata.
"""

from pathlib import Path
from typing import Any

import joblib
from sklearn.preprocessing import PolynomialFeatures


class ModelLoader:
  """
  Handles loading and saving of trained models and transformers. This class provides a centralized
  way to persist and retrieve ML artifacts (models, transformers, metadata).

  Attributes:
    models_dir: Path to the directory containing saved models.
  """
  def __init__(self, models_dir: str | Path | None = None) -> None:
    """
    Initialize the model loader.

    Args:
      models_dir: Path to models directory. Uses default if None.
    """
    if models_dir:
      self.models_dir = Path(models_dir)
    else:
      self.models_dir = self._find_models_dir()

  def _find_models_dir(self) -> Path:
    """
    Find the saved_models directory. Searches in common locations relative to the current working
    directory and the module location.

    Returns:
      Path to the saved_models directory.

    Raises:
      FileNotFoundError: If saved_models directory cannot be found.
    """
    # Possible locations to search
    possible_paths = [
      Path('saved_models'),  # Current working directory
      Path('/app/saved_models'),  # Docker container
      Path(__file__).parent.parent / 'saved_models',  # Relative to this file
    ]

    for path in possible_paths:
      if path.exists():
        return path

    # If not found, default to current directory (for saving new models)
    return Path('saved_models')

  def save_model(self, model: Any, filename: str) -> Path:
    """
    Save a model to disk.

    Args:
      model: The model object to save.
      filename: Name of the file (with or without .joblib extension).

    Returns:
      Path to the saved file.
    """
    self.models_dir.mkdir(exist_ok=True)

    if not filename.endswith('.joblib'):
      filename = f'{filename}.joblib'

    filepath = self.models_dir / filename
    joblib.dump(model, filepath)

    return filepath

  def load_model(self, filename: str) -> Any:
    """
    Load a model from disk.

    Args:
      filename: Name of the file to load.

    Returns:
      The loaded model object.

    Raises:
      FileNotFoundError: If the model file doesn't exist.
    """
    if not filename.endswith('.joblib'):
      filename = f'{filename}.joblib'

    filepath = self.models_dir / filename

    if not filepath.exists():
      raise FileNotFoundError(f'Model not found: {filepath}')

    return joblib.load(filepath)

  def load_transformer(self) -> PolynomialFeatures:
    """
    Load the polynomial transformer.

    Returns:
      Fitted PolynomialFeatures transformer.
    """
    return self.load_model('polynomial_transformer')

  def load_predictor(self) -> Any:
    """
    Load the trained prediction model.

    Returns:
      Trained model object.
    """
    return self.load_model('revenue_model')

  def load_metadata(self) -> dict:
    """
    Load model metadata.

    Returns:
      Dictionary containing model metadata.
    """
    return self.load_model('model_metadata')

  def load_all(self) -> tuple[PolynomialFeatures, Any, dict]:
    """
    Load all model artifacts at once.

    Returns:
      Tuple of (transformer, model, metadata).
    """
    return (self.load_transformer(), self.load_predictor(), self.load_metadata())
