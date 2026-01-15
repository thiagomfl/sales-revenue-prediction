"""
Integration tests for the FastAPI application.
"""

import unittest

from fastapi.testclient import TestClient

from api.infrastructure.api.main import create_app


class TestHealthEndpoint(unittest.TestCase):
  """
  Test cases for health check endpoint.
  """

  @classmethod
  def setUpClass(cls) -> None:
    """
    Set up test client once for all tests.
    """
    cls.app = create_app()
    cls.client = TestClient(cls.app)

  def test_health_check_returns_200(self) -> None:
    """
    Test that health check returns 200 OK.
    """
    response = self.client.get('/health')
    self.assertEqual(response.status_code, 200)

  def test_health_check_returns_healthy_status(self) -> None:
    """
    Test that health check returns healthy status.
    """
    response = self.client.get('/health')
    data = response.json()
    self.assertEqual(data['status'], 'healthy')


class TestPredictEndpoint(unittest.TestCase):
  """
  Test cases for prediction endpoint.
  """

  @classmethod
  def setUpClass(cls) -> None:
    """
    Set up test client once for all tests.
    """
    cls.app = create_app()
    cls.client = TestClient(cls.app)

  def test_predict_returns_200_with_valid_input(self) -> None:
    """
    Test that predict returns 200 with valid input.
    """
    payload = {'experience_months': 36, 'number_of_sales': 50, 'seasonal_factor': 7}
    response = self.client.post('/api/v1/predict', json=payload)
    self.assertEqual(response.status_code, 200)

  def test_predict_returns_predicted_revenue(self) -> None:
    """
    Test that predict returns a predicted revenue.
    """
    payload = {'experience_months': 36, 'number_of_sales': 50, 'seasonal_factor': 7}

    response = self.client.post('/api/v1/predict', json=payload)
    data = response.json()

    self.assertIn('predicted_revenue', data)
    self.assertIsInstance(data['predicted_revenue'], float)

  def test_predict_echoes_input_values(self) -> None:
    """
    Test that predict echoes the input values in response.
    """
    payload = {'experience_months': 36, 'number_of_sales': 50, 'seasonal_factor': 7}

    response = self.client.post('/api/v1/predict', json=payload)
    data = response.json()

    self.assertEqual(data['experience_months'], 36)
    self.assertEqual(data['number_of_sales'], 50)
    self.assertEqual(data['seasonal_factor'], 7)

  def test_predict_returns_422_with_missing_field(self) -> None:
    """
    Test that predict returns 422 when a required field is missing.
    """
    payload = {'experience_months': 36, 'number_of_sales': 50} # Missing seasonal_factor

    response = self.client.post('/api/v1/predict', json=payload)

    self.assertEqual(response.status_code, 422)

  def test_predict_returns_422_with_invalid_seasonal_factor(self) -> None:
    """
    Test that predict returns 422 when seasonal factor is out of range.
    """
    # Invalid seasonal_factor: must be 1-10
    payload = {'experience_months': 36, 'number_of_sales': 50, 'seasonal_factor': 15} 

    response = self.client.post('/api/v1/predict', json=payload)

    self.assertEqual(response.status_code, 422)

  def test_predict_returns_422_with_negative_experience(self) -> None:
    """
    Test that predict returns 422 when experience is negative.
    """
    payload = {'experience_months': -5, 'number_of_sales': 50, 'seasonal_factor': 7}

    response = self.client.post('/api/v1/predict', json=payload)

    self.assertEqual(response.status_code, 422)


class TestModelInfoEndpoint(unittest.TestCase):
  """
  Test cases for model info endpoint.
  """
  @classmethod
  def setUpClass(cls) -> None:
    """
    Set up test client once for all tests.
    """
    cls.app = create_app()
    cls.client = TestClient(cls.app)

  def test_model_info_returns_200(self) -> None:
    """
    Test that model info returns 200 OK.
    """
    response = self.client.get('/api/v1/model/info')
    self.assertEqual(response.status_code, 200)

  def test_model_info_returns_metadata(self) -> None:
    """
    Test that model info returns model metadata.
    """
    response = self.client.get('/api/v1/model/info')
    data = response.json()

    self.assertIn('model_type', data)
    self.assertIn('polynomial_degree', data)
    self.assertIn('features', data)


if __name__ == '__main__':
  unittest.main()
