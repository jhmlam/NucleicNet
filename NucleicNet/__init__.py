"""Root package info."""

import logging
import os

from NucleicNet.__about__ import *  # noqa: F401, F403

_root_logger = logging.getLogger()
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

# if root logger has handlers, propagate messages up and let root logger process them
if not _root_logger.hasHandlers():
    _logger.addHandler(logging.StreamHandler())
    _logger.propagate = False

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

#__all__ = ["Load", "Fuel", "Burn", "util"]

# for compatibility with namespace packages
__import__("pkg_resources").declare_namespace(__name__)

