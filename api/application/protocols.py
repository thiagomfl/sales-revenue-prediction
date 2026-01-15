"""
Protocols (interfaces) for the application layer. Protocols define contracts that infrastructure
must implement. This allows the application layer to depend on abstractions, not concrete
implementations (Dependency Inversion Principle).
"""

from typing import Protocol


class ModelRepositoryProtocol(Protocol):
  """
  Protocol for model repository. Any class that implements this protocol can be used as a model
  repository in the use cases.
  """

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
    pass