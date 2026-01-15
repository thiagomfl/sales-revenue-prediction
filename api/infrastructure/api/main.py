"""
FastAPI application factory and configuration. This module creates and configures the FastAPI
application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.infrastructure.api.routes import router


def create_app() -> FastAPI:
  """
  Create and configure the FastAPI application.

  Returns:
    Configured FastAPI application instance.
  """
  app = FastAPI(
    title='Sales Revenue Prediction API',
    description=(
      'A machine learning API that predicts sales revenue based on '
      'seller experience, number of sales, and seasonal factors.'
    ),
    version='1.0.0',
    docs_url='/docs',
    redoc_url='/redoc',
  )

  # Configure CORS
  app.add_middleware(
    CORSMiddleware, allow_origins=['*'],
    allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

  # Include routers
  app.include_router(router)

  # Health check endpoint
  @app.get(
    '/health',
    tags=['health'],
    summary='Health check',
    description='Check if the API is running.')
  def health_check() -> dict:
    """
    Health check endpoint.

    Returns:
      Dictionary with status.
    """
    return {'status': 'healthy'}

  return app


# Create app instance for uvicorn
app = create_app()


def start() -> None:
  """
  Start the API server. This function is used by the poetry script command.
  """
  import uvicorn
  uvicorn.run('api.infrastructure.api.main:app', host='0.0.0.0', port=8000, reload=True)
