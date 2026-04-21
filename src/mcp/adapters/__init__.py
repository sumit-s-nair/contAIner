"""Adapter registry — re-exports every adapter class for convenient import."""

from .pip_adapter import PipAdapter
from .npm_adapter import NpmAdapter
from .apt_adapter import AptAdapter
from .brew_adapter import BrewAdapter
from .conda_adapter import CondaAdapter
from .docker_adapter import DockerAdapter
from .cargo_adapter import CargoAdapter
from .go_adapter import GoAdapter
from .maven_adapter import MavenAdapter
from .base import BaseAdapter

__all__ = [
    "BaseAdapter",
    "PipAdapter",
    "NpmAdapter",
    "AptAdapter",
    "BrewAdapter",
    "CondaAdapter",
    "DockerAdapter",
    "CargoAdapter",
    "GoAdapter",
    "MavenAdapter",
]
