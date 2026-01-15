"""
Data Transfer Objects for the sales prediction API. DTOs define the structure of data that flows in
and out of the API. They use Pydantic for automatic validation and serialization.
"""

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
  """
  Input data for revenue prediction.

  Attributes:
    experience_months: Seller's experience in months (>= 0).
    number_of_sales: Number of sales made (>= 0).
    seasonal_factor: Seasonal factor from 1 to 10.
  """
  experience_months: int = Field(
    ..., ge=0, description='Seller\'s experience in months', examples=[36])

  number_of_sales: int = Field(
    ..., ge=0, description='Number of sales made by the seller', examples=[50])

  seasonal_factor: int = Field(
    ..., ge=1, le=10, description='Seasonal factor (1 = low season, 10 = peak season)', examples=[7])


class PredictionOutput(BaseModel):
  """
  Output data from revenue prediction.

  Attributes:
    predicted_revenue: Predicted revenue in BRL.
    experience_months: Input experience months (echo).
    number_of_sales: Input number of sales (echo).
    seasonal_factor: Input seasonal factor (echo).
    model_info: Information about the model used.
  """
  predicted_revenue: float = Field(
    ..., description='Predicted revenue in BRL (R$)', examples=[5644.24])

  experience_months: int = Field(..., description='Seller\'s experience in months')

  number_of_sales: int = Field(..., description='Number of sales made')

  seasonal_factor: int = Field(..., description='Seasonal factor used')

  model_info: str = Field(
    default='Polynomial Regression (degree=2)', description='Model used for prediction')
