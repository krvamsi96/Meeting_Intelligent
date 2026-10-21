"""Dynamic API router discovery."""

import importlib
import logging
import pkgutil
from typing import List

from fastapi import APIRouter

logger = logging.getLogger(__name__)

_PACKAGE_NAME = "backend.api"


def discover_routers() -> List[APIRouter]:
    """Return all APIRouter instances defined in modules of this package."""
    routers: List[APIRouter] = []
    package = importlib.import_module(_PACKAGE_NAME)
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name in ("__init__",):
            continue
        module = importlib.import_module(f"{_PACKAGE_NAME}.{module_info.name}")
        router = getattr(module, "router", None)
        if isinstance(router, APIRouter):
            routers.append(router)
            logger.info(f"Discovered API router from backend.api.{module_info.name}")
    return routers
