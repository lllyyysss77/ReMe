"""Module providing a registry class for managing class-to-name mappings via decorators."""

import inspect
from typing import Callable, TypeVar

from .base_context import BaseContext
from ..utils import singleton

T = TypeVar("T")


class Registry(BaseContext):
    """A registry container that uses decorators to map and store class references."""

    def register(self, name: str | type = "") -> Callable[[type[T]], type[T]] | type[T]:
        """Return a decorator that registers a class under a specific name in the registry."""
        if inspect.isclass(name):
            self[name.__name__] = name
            return name

        else:

            def decorator(cls):
                key: str = name if isinstance(name, str) and name else cls.__name__
                self[key] = cls
                return cls

            return decorator


@singleton
class RegistryFactory:
    """A factory class for creating registries."""

    def __init__(self):
        self.llm = Registry()
        self.embedding_model = Registry()
        self.vector_store = Registry()
        self.op = Registry()
        self.flow = Registry()
        self.service = Registry()
        self.token_counter = Registry()


R = RegistryFactory()
