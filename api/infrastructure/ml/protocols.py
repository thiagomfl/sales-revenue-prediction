"""
Protocols for ML infrastructure. Defines contracts for ML models to enable flexibility and easier
testing.
"""

from typing import Protocol

import numpy as np


class RegressorModelProtocol(Protocol):
  """
  Protocol for regression models. Any model that implements a predict method with this signature can
  be used as a regressor in the application.
  """

  def predict(self, X: np.ndarray) -> np.ndarray:
    """
    Make predictions for input samples.

    Args:
      X: Input samples with shape (n_samples, n_features).

    Returns:
      Predicted values with shape (n_samples,).
    """
    pass