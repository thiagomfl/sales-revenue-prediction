"""
Domain entities for sales prediction. This entity represents the core business concept or a revenue
prediction.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SalesPrediction:
  """
  Represents a sales revenue prediction. This is a domain entity that encapsulates the result of a
  revenue prediction operation.

  Attributes:
    experience_months: Seller's experience in months.
    number_of_sales: Number of sales made.
    seasonal_factor: Seasonal factor (1-10).
    predicted_revenue: Predicted revenue in BRL.
  """
  experience_months: int
  number_of_sales: int
  seasonal_factor: int
  predicted_revenue: float

  def __post_init__(self) -> None:
    """
    Validate entity attributes after initialization.

    Raises:
      ValueError: If any attribute is invalid.
    """
    if self.experience_months < 0:
      raise ValueError('Experience months cannot be negative')

    if self.number_of_sales < 0:
      raise ValueError('Number of sales cannot be negative')

    if not 1 <= self.seasonal_factor <= 10:
      raise ValueError('Seasonal factor must be between 1 and 10')

    if self.predicted_revenue < 0:
      raise ValueError('Predicted revenue cannot be negative')
