"""
Data preprocessing utilities for sales prediction. This module handles: Loading and validating input
data; Polynomial feature transformation; Model and transformer persistence.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


class DataPreprocessor:
  """
  Handles data preprocessing for the sales prediction model. This class manages polynomial feature
  transformation for converting raw input features into the format expected by the trained model.

  Attributes:
    degree: Polynomial degree for feature transformation.
    transformer: Fitted PolynomialFeatures instance.
    feature_names: List of original feature names. 
  """

  EXPECTED_FEATURES = ['years_of_experience', 'number_of_sales', 'seasonal_factor']

  def __init__(self, degree: int = 2) -> None:
    """
    Initialize the preprocessor with the specified polynomial degree.

    Args:
      degree: Polynomial degree for feature transformation.
    """
    self.degree = degree
    self.transformer: PolynomialFeatures | None = None
    self.feature_names = self.EXPECTED_FEATURES.copy()

  def fit(self, X: pd.DataFrame | np.ndarray) -> 'DataPreprocessor':
    """
    Fit the polynomial transformer on training data.

    Args:
      X: Training features with shape (n_samples, n_features).

    Returns:
      Fitted DataPreprocessor instance.
    """
    self.transformer = PolynomialFeatures(degree=self.degree, include_bias=False)
    self.transformer.fit(X)
    return self

  def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """
    Transform features using the fitted polynomial transformer.

    Args:
      X: Features to transform with shape (n_samples, n_features).

    Returns:
      Transformed features with shape (n_samples, n_transformed_features).

    Raises:
      ValueError: If transformer has not been fitted.
    """
    if self.transformer is None:
      raise ValueError('Transformer not fitted. Call fit() first.')

    return self.transformer.transform(X)

  def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """
    Fit and transform in a single step.

    Args:
      X: Features to transform with shape (n_samples, n_features).

    Returns:
      Transformed features as numpy array.
    """
    return self.fit(X).transform(X)

  def get_feature_names_out(self) -> list[str]:
    """
    Gets the names of transformed features.

    Returns:
      List of transformed feature names.

    Raises:
      ValueError: If transformer has not been fitted.
    """
    if self.transformer is None:
      raise ValueError('Transformer not fitted. Call fit() first.')

    return self.transformer.get_feature_names_out().tolist()

  def validate_input(self, data: dict) -> np.ndarray:
    """
    Validate and convert input dictionary to numpy array.

    Args:
      data: Dictionary with feature names as keys.

    Returns:
      Numpy array with shape (1, n_features).
    
    Raises:
      ValueError: If required features are missing.
    """
    missing = set(self.EXPECTED_FEATURES) - set(data.keys())
    if missing:
      raise ValueError(f'Missing required features: {missing}')

    return np.array([[
      data['years_of_experience'],
      data['number_of_sales'],
      data['seasonal_factor']
    ]])
