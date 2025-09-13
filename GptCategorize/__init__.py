"""
GptCategorize package for categorizing ChatGPT chats by title and initial prompt.

This package provides functionality to analyze ChatGPT conversation exports,
cluster similar chats by title and the first user prompt using embeddings and machine learning,
and generate provisional move plans for organizing chats into project folders.
"""

from .categorize import categorize_chats, QDRANT_COLLECTION

__version__ = "1.0.0"
__all__ = ["categorize_chats", "QDRANT_COLLECTION"]
