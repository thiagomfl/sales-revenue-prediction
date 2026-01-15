"""
Use cases for the sales prediction API. Use cases contain the application business logic and
orchestrate the interaction between domain entities and infrastructure.
"""

from api.application.dtos import PredictionInput, PredictionOutput
from api.application.protocols import ModelRepositoryProtocol
from api.domain.sales_prediction import SalesPrediction


class PredictRevenueUseCase:
  """
  Use case for predicting sales revenue. This use case orchestrates the prediction flow: Receives
  validated input; Uses the ML model to predict revenue; Returns the prediction as a domain entity.

  Attributes:
    model_repository: Repository for accessing the ML model.
  """
  def __init__(self, model_repository: ModelRepositoryProtocol) -> None:
    """
    Initialize the use case with dependencies.

    Args:
      model_repository: Repository for model access.
    """
    self.model_repository = model_repository

  def execute(self, input_data: PredictionInput) -> PredictionOutput:
    """
    Execute the revenue prediction.

    Args:
      input_data: Validated input data from the API.

    Returns:
      PredictionOutput with the predicted revenue.
    """
    # Get prediction from model repository
    predicted_revenue = self.model_repository.predict(
      experience_months=input_data.experience_months,
      number_of_sales=input_data.number_of_sales,
      seasonal_factor=input_data.seasonal_factor)

    # Create domain entity (validates business rules)
    prediction = SalesPrediction(
      experience_months=input_data.experience_months,
      number_of_sales=input_data.number_of_sales,
      seasonal_factor=input_data.seasonal_factor,
      predicted_revenue=predicted_revenue)

    # Return DTO for API response
    return PredictionOutput(
      predicted_revenue=prediction.predicted_revenue,
      experience_months=prediction.experience_months,
      number_of_sales=prediction.number_of_sales,
      seasonal_factor=prediction.seasonal_factor)