"""
API infrastructure - FastAPI routes and configuration.
"""

from api.infrastructure.api.main import create_app
from api.infrastructure.api.routes import router


__all__ = ['create_app', 'router']
