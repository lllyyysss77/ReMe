"""Module providing a registry class for managing class-to-name mappings via decorators."""

import inspect
from typing import Callable, TypeVar

from .base_context import BaseContext

T = TypeVar('T')


class Registry(BaseContext):
    """A registry container that uses decorators to map and store class references."""

    def register(self, name: str | type = "", add_cls: bool = True) -> Callable[[type[T]], type[T]] | type[T]:
        """Return a decorator that registers a class under a specific name in the registry.

        Can be used in three ways:
        - @C.register_op()  # with empty parentheses, uses class name
        - @C.register_op    # without parentheses, uses class name
        - @C.register_op("custom_name")  # with custom name

        Args:
            name: Either a string name for the class, or the class itself when used without parentheses
            add_cls: Whether to actually add the class to the registry

        Returns:
            Either a decorator function or the registered class itself
        """

        def decorator(cls):
            if add_cls:
                # Use provided name or default to the class name as the key
                key = name if isinstance(name, str) and name else cls.__name__
                self[key] = cls
            return cls

        # If used without parentheses: @C.register_op
        if inspect.isclass(name):
            cls = name
            # Register with class name as key
            if add_cls:
                self[cls.__name__] = cls
            return cls

        # If used with parentheses: @C.register_op() or @C.register_op("name")
        return decorator
