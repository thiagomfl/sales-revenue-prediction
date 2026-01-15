"""
ML infrastructure - Model loading and prediction.
"""

from api.infrastructure.ml.model_repository import ModelRepository
from api.infrastructure.ml.protocols import RegressorModelProtocol


__all__ = ['ModelRepository', 'RegressorModelProtocol']
