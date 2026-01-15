"""
API routes for sales prediction. This module defines the HTTP endpoints for the prediction API.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from api.application.dtos import PredictionInput, PredictionOutput
from api.application.use_cases import PredictRevenueUseCase
from api.infrastructure.ml.model_repository import ModelRepository

router = APIRouter(prefix='/api/v1', tags=['predictions'])

# -----------------------------------------------------------------------------
# Dependency Injection
# -----------------------------------------------------------------------------
# We use a simple singleton pattern for the model repository
# In a larger app, you might use a proper DI container

_model_repository: ModelRepository | None = None


def get_model_repository() -> ModelRepository:
  """
  Get or create the model repository singleton.

  Returns:
    ModelRepository instance.
  """
  global _model_repository
  if _model_repository is None:
    _model_repository = ModelRepository()

  return _model_repository


def get_predict_use_case(
  model_repository: ModelRepository = Depends(get_model_repository)) -> PredictRevenueUseCase:
  """
  Get the prediction use case with dependencies injected.

  Args:
    model_repository: Injected model repository.

  Returns:
    PredictRevenueUseCase instance.
  """
  return PredictRevenueUseCase(model_repository)


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@router.post(
  '/predict',
  response_model=PredictionOutput,
  status_code=status.HTTP_200_OK,
  summary='Predict sales revenue',
  description='Predict revenue based on seller experience, sales count, and seasonal factor.',
)
def predict_revenue(
  input_data: PredictionInput,
  use_case: PredictRevenueUseCase = Depends(get_predict_use_case)) -> PredictionOutput:
  """
  Predict sales revenue for a seller.

  Args:
    input_data: Input features for prediction.
    use_case: Injected prediction use case.

  Returns:
    Prediction output with estimated revenue.

  Raises:
    HTTPException: If prediction fails.
  """
  try:
    return use_case.execute(input_data)
  except ValueError as e:
    raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
  except Exception as e:
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=f'Prediction failed: {str(e)}')


@router.get(
  '/model/info',
  status_code=status.HTTP_200_OK,
  summary='Get model information',
  description='Returns metadata about the currently loaded model.')
def get_model_info(model_repository: ModelRepository = Depends(get_model_repository)) -> dict:
  """
  Get information about the loaded model.

  Args:
    model_repository: Injected model repository.

  Returns:
    Dictionary with model metadata.
  """
  return model_repository.get_model_info()
