"""
Chevir: Bidirectional AI-Powered Sign Language Accessibility Layer
"""

__version__ = "1.0.0"
__author__ = "Chevir App Development Team"

# Expose global configuration at the package root level
from .config import (
    BASE_DIR, 
    DATA_DIR, 
    SEQUENCE_LENGTH, 
    INPUT_SIZE, 
    NUM_CLASSES
)

__all__ = [
    "__version__",
    "__author__",
    "BASE_DIR",
    "DATA_DIR",
    "SEQUENCE_LENGTH",
    "INPUT_SIZE",
    "NUM_CLASSES"
]
