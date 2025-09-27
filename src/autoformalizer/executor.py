"""
DEPRECATED: Legacy executor module for backward compatibility.

This module has been refactored into a package structure in autoformalizer.executor.
Please import from the new package directly:

- autoformalizer.executor.lean for compilation functions
- autoformalizer.executor for the main interfaces

This file will be removed in a future version.
"""

import warnings

# Import everything from the new executor package for backward compatibility
from .executor import *  # noqa: F403

# Issue deprecation warning
warnings.warn(
    "Importing from autoformalizer.executor is deprecated. "
    "Use autoformalizer.executor.lean for compilation functions "
    "or autoformalizer.executor for main interfaces.",
    DeprecationWarning,
    stacklevel=2,
)
