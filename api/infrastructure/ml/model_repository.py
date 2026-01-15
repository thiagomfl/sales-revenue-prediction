"""
Model repository implementation. This module implements the ModelRepositoryProtocol and handles
loading the trained ML model and making predictions.
"""

from pathlib import Path

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from core.persistence import ModelLoader
from api.infrastructure.ml.protocols import RegressorModelProtocol


class ModelRepository:
  """
  Repository for accessing the trained ML model. This class implements the ModelRepositoryProtocol
  defined in the application layer.

  Attributes:
    model_loader: Loader for ML artifacts.
    transformer: Fitted polynomial transformer.
    model: Trained prediction model.
    metadata: Model metadata.
  """
  def __init__(self, models_dir: Path | str | None = None) -> None:
    """
    Initialize the repository and load the model.

    Args:
      models_dir: Path to the models directory.
    """
    self.model_loader = ModelLoader(models_dir)
    self.transformer: PolynomialFeatures | None = None
    self.model: RegressorModelProtocol | None = None
    self.metadata: dict | None = None
    self._load_model()

  def _load_model(self) -> None:
    """
    Load all model artifacts from disk.
    """
    self.transformer, self.model, self.metadata = self.model_loader.load_all()

  def predict(self, experience_months: int, number_of_sales: int, seasonal_factor: int) -> float:
    """
    Make a revenue prediction.

    Args:
      experience_months: Seller's experience in months.
      number_of_sales: Number of sales made.
      seasonal_factor: Seasonal factor (1-10).

    Returns:
      Predicted revenue in BRL.
    """
    # Prepare input array
    input_array = np.array([[experience_months, number_of_sales, seasonal_factor]])

    # Transform features
    input_transformed = self.transformer.transform(input_array)

    # Make prediction
    prediction = self.model.predict(input_transformed)[0]

    return float(prediction)

  def get_model_info(self) -> dict:
    """
    Get information about the loaded model.

    Returns:
      Dictionary with model metadata.
    """
    return self.metadata