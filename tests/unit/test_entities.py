"""
Unit tests for domain entities.
"""

import unittest

from api.domain.sales_prediction import SalesPrediction


class TestSalesPrediction(unittest.TestCase):
  """
  Unit tests for SalesPrediction entity.
  """

  def test_create_valid_prediction(self) -> None:
    """
    Test creating a valid SalesPrediction entity.
    """
    prediction = SalesPrediction(
      experience_months=36, number_of_sales=50, seasonal_factor=7, predicted_revenue=5644.24)

    self.assertEqual(prediction.experience_months, 36)
    self.assertEqual(prediction.number_of_sales, 50)
    self.assertEqual(prediction.seasonal_factor, 7)
    self.assertEqual(prediction.predicted_revenue, 5644.24)

  def test_negative_experience_raises_error(self) -> None:
    """
    Test that negative experience months raises ValueError.
    """
    with self.assertRaises(ValueError) as context:
      SalesPrediction(
        experience_months=-1, number_of_sales=50, seasonal_factor=7, predicted_revenue=5644.24)

    self.assertIn('Experience months cannot be negative', str(context.exception))

  def test_negative_sales_raises_error(self) -> None:
    """
    Test that negative number of sales raises ValueError.
    """
    with self.assertRaises(ValueError) as context:
      SalesPrediction(
        experience_months=36, number_of_sales=-1, seasonal_factor=7, predicted_revenue=5644.24)

    self.assertIn('Number of sales cannot be negative', str(context.exception))

  def test_seasonal_factor_below_range_raises_error(self) -> None:
    """
    Test that seasonal factor below 1 raises ValueError.
    """
    with self.assertRaises(ValueError) as context:
      SalesPrediction(
        experience_months=36, number_of_sales=50, seasonal_factor=0, predicted_revenue=5644.24)

    self.assertIn('Seasonal factor must be between 1 and 10', str(context.exception))

  def test_seasonal_factor_above_range_raises_error(self) -> None:
    """
    Test that seasonal factor above 10 raises ValueError.
    """
    with self.assertRaises(ValueError) as context:
      SalesPrediction(
        experience_months=36, number_of_sales=50, seasonal_factor=11, predicted_revenue=5644.24)

    self.assertIn('Seasonal factor must be between 1 and 10', str(context.exception))

  def test_negative_revenue_raises_error(self) -> None:
    """
    Test that negative predicted revenue raises ValueError.
    """
    with self.assertRaises(ValueError) as context:
      SalesPrediction(
        experience_months=36, number_of_sales=50, seasonal_factor=7, predicted_revenue=-100.0)

    self.assertIn('Predicted revenue cannot be negative', str(context.exception))

  def test_entity_is_immutable(self) -> None:
    """
    Test that SalesPrediction is immutable (frozen dataclass).
    """
    prediction = SalesPrediction(
      experience_months=36, number_of_sales=50, seasonal_factor=7, predicted_revenue=5644.24)

    with self.assertRaises(AttributeError):
      prediction.experience_months = 100  # type: ignore


if __name__ == '__main__':
  unittest.main()
