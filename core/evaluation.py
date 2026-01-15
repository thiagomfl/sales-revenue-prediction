"""
Model evaluation utilities for sales prediction. This module handles: Calculating regression metrics
(MSE, RMSE, MAE, R²); Formatting evaluation results.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
  mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error)


@dataclass
class EvaluationMetrics:
  """
  Container for model evaluation metrics.

  Attributes:
    mse: Mean Squared Error.
    rmse: Root Mean Squared Error.
    mae: Mean Absolute Error.
    r2: R-squared (coefficient of determination).
  """
  mse: float
  rmse: float
  mae: float
  r2: float

  def to_dict(self) -> dict:
    """
    Convert metrics to dictionary format.

    Returns:
      Dictionary with metric names as keys.
    """
    return {'mse': self.mse, 'rmse': self.rmse, 'mae': self.mae, 'r2': self.r2}

  def __str__(self) -> str:
    """
    Return a formatted string representation of metrics.

    Returns:
      Formatted string with all metrics.
    """
    return (
      f'MSE:  {self.mse:,.2f}\n'
      f'RMSE: {self.rmse:,.2f}\n'
      f'MAE:  {self.mae:,.2f}\n'
      f'R²:   {self.r2:.4f} ({self.r2 * 100:.2f}%)'
    )


class ModelEvaluator:
  """
  Evaluates regression model performance. This class provides methods to calculate and format
  standard regression metrics.
  """

  @staticmethod
  def evaluate(y_true: np.ndarray | list, y_pred: np.ndarray | list) -> EvaluationMetrics:
    """
    Calculate all evaluation metrics.

    Args:
      y_true: Ground truth (actual) values.
      y_pred: Predicted values.

    Returns:
      EvaluationMetrics dataclass with all metrics.
    """
    return EvaluationMetrics(
      mse=mean_squared_error(y_true, y_pred),
      rmse=root_mean_squared_error(y_true, y_pred),
      mae=mean_absolute_error(y_true, y_pred),
      r2=r2_score(y_true, y_pred)
    )
  
  @staticmethod
  def interpret_r2(r2: float) -> str:
    """
    Provide interpretation of R² score.

    Args:
      r2: R-squared value.

    Returns:
      String interpretation of the R² value.
    """
    if r2 < 0:
      return 'Very poor - model is worse than predicting the mean'
    elif r2 < 0.3:
      return 'Poor - model has weak predictive power'
    elif r2 < 0.7:
      return 'Moderate - model captures some patterns'
    elif r2 < 0.9:
      return 'Good - model captures most patterns'
    else:
      return 'Excellent - model has strong predictive power'
    