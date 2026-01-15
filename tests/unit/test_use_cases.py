"""
Unit tests for application use cases.
"""

import unittest
from unittest.mock import Mock

from api.application.dtos import PredictionInput, PredictionOutput
from api.application.use_cases import PredictRevenueUseCase


class TestPredictRevenueUseCase(unittest.TestCase):
  """
  Test cases for PredictRevenueUseCase.
  """
  def setUp(self) -> None:
    """
    Set up test fixtures.
    """
    # Create a mock model repository
    self.mock_repository = Mock()
    self.mock_repository.predict.return_value = 5644.24

    self.use_case = PredictRevenueUseCase(self.mock_repository)

  def test_execute_returns_prediction_output(self) -> None:
    """
    Test that execute returns a PredictionOutput.
    """
    input_data = PredictionInput(experience_months=36, number_of_sales=50, seasonal_factor=7)

    result = self.use_case.execute(input_data)

    self.assertIsInstance(result, PredictionOutput)
    self.assertEqual(result.predicted_revenue, 5644.24)
    self.assertEqual(result.experience_months, 36)
    self.assertEqual(result.number_of_sales, 50)
    self.assertEqual(result.seasonal_factor, 7)

  def test_execute_calls_repository_predict(self) -> None:
    """
    Test that execute calls repository.predict with correct arguments.
    """
    input_data = PredictionInput(experience_months=36, number_of_sales=50, seasonal_factor=7)

    self.use_case.execute(input_data)

    self.mock_repository.predict.assert_called_once_with(
      experience_months=36, number_of_sales=50, seasonal_factor=7)

  def test_execute_with_different_inputs(self) -> None:
    """
    Test execute with different input values.
    """
    self.mock_repository.predict.return_value = 3000.00

    input_data = PredictionInput(experience_months=12, number_of_sales=20, seasonal_factor=3)

    result = self.use_case.execute(input_data)

    self.assertEqual(result.predicted_revenue, 3000.00)
    self.assertEqual(result.experience_months, 12)
    self.assertEqual(result.number_of_sales, 20)
    self.assertEqual(result.seasonal_factor, 3)


if __name__ == '__main__':
  unittest.main()
