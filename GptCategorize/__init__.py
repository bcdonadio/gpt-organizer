"""
GptCategorize package for categorizing ChatGPT chats by title.

This package provides functionality to analyze ChatGPT conversation exports,
cluster similar chats by title using embeddings and machine learning,
and generate provisional move plans for organizing chats into project folders.
"""

from .categorize import categorize_chats, QDRANT_COLLECTION

__version__ = "1.0.0"
__all__ = ["categorize_chats", "QDRANT_COLLECTION"]
