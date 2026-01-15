"""
Unit tests for preprocessing module.
"""

import unittest

import numpy as np

from core.preprocessing import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
  """
  Test cases for DataPreprocessor class.
  """
  def setUp(self) -> None:
    """
    Set up test fixtures.
    """
    self.preprocessor = DataPreprocessor(degree=2)
    self.sample_data = np.array([[36, 50, 7], [24, 30, 5], [48, 70, 9]])

  def test_fit_creates_transformer(self) -> None:
    """
    Test that fit creates a transformer.
    """
    self.assertIsNone(self.preprocessor.transformer)
    self.preprocessor.fit(self.sample_data)
    self.assertIsNotNone(self.preprocessor.transformer)

  def test_transform_without_fit_raises_error(self) -> None:
    """
    Test that transform without fit raises ValueError.
    """
    with self.assertRaises(ValueError) as context:
      self.preprocessor.transform(self.sample_data)

    self.assertIn('Transformer not fitted', str(context.exception))

  def test_fit_transform_returns_correct_shape(self) -> None:
    """
    Test that fit_transform returns correct number of features.
    """
    # Degree 2 with 3 features should create 9 polynomial features
    transformed = self.preprocessor.fit_transform(self.sample_data)

    self.assertEqual(transformed.shape[0], 3)  # Same number of samples
    self.assertEqual(transformed.shape[1], 9)  # 9 polynomial features

  def test_get_feature_names_out(self) -> None:
    """
    Test that feature names are returned after fitting.
    """
    self.preprocessor.fit(self.sample_data)

    feature_names = self.preprocessor.get_feature_names_out()

    self.assertEqual(len(feature_names), 9)
    self.assertIsInstance(feature_names, list)

  def test_get_feature_names_without_fit_raises_error(self) -> None:
    """
    Test that get_feature_names_out without fit raises ValueError.
    """
    with self.assertRaises(ValueError) as context:
      self.preprocessor.get_feature_names_out()

    self.assertIn('Transformer not fitted', str(context.exception))

  def test_validate_input_valid_data(self) -> None:
    """
    Test validate_input with valid data.
    """
    data = {'years_of_experience': 36, 'number_of_sales': 50, 'seasonal_factor': 7}
    result = self.preprocessor.validate_input(data)

    self.assertEqual(result.shape, (1, 3))
    np.testing.assert_array_equal(result, [[36, 50, 7]])

  def test_validate_input_missing_feature_raises_error(self) -> None:
    """
    Test validate_input with missing feature raises ValueError.
    """
    data = {"years_of_experience": 36, "number_of_sales": 50} # Missing seasonal_factor

    with self.assertRaises(ValueError) as context:
      self.preprocessor.validate_input(data)

    self.assertIn('Missing required features', str(context.exception))


if __name__ == '__main__':
  unittest.main()
