"""Module providing a registry class for managing class-to-name mappings via decorators."""

import inspect
from typing import Callable, TypeVar

from .base_context import BaseContext
from ..enumeration import RegistryEnum
from ...core_old.utils import singleton

T = TypeVar("T")


@singleton
class Registry(BaseContext):
    """A singleton registry manager that maintains separate registries for different component types.

    This class serves as the central registry hub for the entire ReMe application, providing:
    - Component registration for different types (LLMs, embeddings, vector stores, etc.)
    - Convenient access methods for retrieving registered classes
    - Decorator-based registration API

    The singleton pattern ensures only one instance exists throughout the application lifecycle,
    accessible via the global `R` variable exported at the bottom of this module.
    """

    def __init__(self, **kwargs):
        """Initialize the registry manager with separate registries for each component type."""
        super().__init__(**kwargs)

        # Registry system: stores class definitions for different component types
        self.registry_dict: dict[RegistryEnum, dict] = {
            v: {} for v in RegistryEnum.__members__.values()
        }

    def register(self, name: str | type = "", register_type: RegistryEnum = None) -> Callable[[type[T]], type[T]] | type[T]:
        """Return a decorator to register a component within a specific registry category.

        Can be used in multiple ways:
        - @R.register_op()  # with empty parentheses, uses class name
        - @R.register_op    # without parentheses, uses class name
        - @R.register_op("custom_name")  # with custom name

        Args:
            name: Either a string name for the class, or the class itself when used without parentheses
            register_type: The type of registry (LLM, EMBEDDING_MODEL, VECTOR_STORE, etc.)

        Returns:
            Either a decorator function or the registered class itself

        Example:
            @R.register("my_llm", RegistryEnum.LLM)
            class MyLLM(BaseLLM):
                pass
        """
        if inspect.isclass(name):
            # Used without parentheses: @R.register_op
            self.registry_dict[register_type][name.__name__] = name
            return name
        else:
            # Used with parentheses: @R.register_op() or @R.register_op("name")
            def decorator(cls):
                key = name if isinstance(name, str) and name else cls.__name__
                self.registry_dict[register_type][key] = cls
                return cls

            return decorator

    def register_llm(self, name: str = ""):
        """Register a Large Language Model class."""
        return self.register(name=name, register_type=RegistryEnum.LLM)

    def register_embedding_model(self, name: str = ""):
        """Register an embedding model class."""
        return self.register(name=name, register_type=RegistryEnum.EMBEDDING_MODEL)

    def register_vector_store(self, name: str = ""):
        """Register a vector store implementation class."""
        return self.register(name=name, register_type=RegistryEnum.VECTOR_STORE)

    def register_op(self, name: str = ""):
        """Register an operation (Op) class."""
        return self.register(name=name, register_type=RegistryEnum.OP)

    def register_flow(self, name: str = ""):
        """Register a workflow or logic flow class."""
        return self.register(name=name, register_type=RegistryEnum.FLOW)

    def register_service(self, name: str = ""):
        """Register a backend service class."""
        return self.register(name=name, register_type=RegistryEnum.SERVICE)

    def register_token_counter(self, name: str = ""):
        """Register a token counting utility class."""
        return self.register(name=name, register_type=RegistryEnum.TOKEN_COUNTER)

    def get_model_class(self, name: str, register_type: RegistryEnum):
        """Retrieve a registered class by name from a specific registry category.

        Args:
            name: The registration name of the class
            register_type: The type of registry to search in

        Returns:
            The registered class (not an instance, but the class itself)

        Raises:
            AssertionError: If the class is not found in the registry
        """
        assert name in self.registry_dict[register_type], f"{name} not in registry_dict[{register_type}]"
        return self.registry_dict[register_type][name]

    def get_llm_class(self, name: str):
        """Get the LLM class registered under the given name."""
        return self.get_model_class(name, RegistryEnum.LLM)

    def get_embedding_model_class(self, name: str):
        """Get the embedding model class registered under the given name."""
        return self.get_model_class(name, RegistryEnum.EMBEDDING_MODEL)

    def get_vector_store_class(self, name: str):
        """Get the vector store class registered under the given name."""
        return self.get_model_class(name, RegistryEnum.VECTOR_STORE)

    def get_op_class(self, name: str):
        """Get the operation class registered under the given name."""
        return self.get_model_class(name, RegistryEnum.OP)

    def get_flow_class(self, name: str):
        """Get the flow class registered under the given name."""
        return self.get_model_class(name, RegistryEnum.FLOW)

    def get_service_class(self, name: str):
        """Get the service class registered under the given name."""
        return self.get_model_class(name, RegistryEnum.SERVICE)

    def get_token_counter_class(self, name: str):
        """Get the token counter class registered under the given name."""
        return self.get_model_class(name, RegistryEnum.TOKEN_COUNTER)


# Export a global singleton instance for easy access across the application
# This is the primary way to access the registry throughout the codebase
R = Registry()
