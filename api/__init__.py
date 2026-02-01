"""FastAPI application and routes"""
from .main import app
from . import models

__all__ = ['app', 'models']
