"""ReMe application classes for simplified configuration and execution."""

import sys

from .application import Application
from .config import ReMeConfigParser
from .context import C


class ReMe(Application):
    """Simplified ReMe application that auto-initializes the service context."""

    def __init__(
        self,
        *args,
        llm_api_key: str | None = None,
        llm_api_base: str | None = None,
        embedding_api_key: str | None = None,
        embedding_api_base: str | None = None,
        enable_logo: bool = True,
        llm: dict | None = None,
        embedding_model: dict | None = None,
        vector_store: dict | None = None,
        token_counter: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
            embedding_api_key=embedding_api_key,
            embedding_api_base=embedding_api_base,
            service_config=None,
            parser=ReMeConfigParser,
            config_path=None,
            enable_logo=enable_logo,
            llm=llm,
            embedding_model=embedding_model,
            vector_store=vector_store,
            token_counter=token_counter,
            **kwargs,
        )

        C.initialize_service_context()

    async def summary(self):
        """Execute summary operations."""

    async def retrieve(self):
        """Execute retrieve operations."""


class ReMeApp(Application):
    """ReMe application with config file support and flow execution methods."""

    def __init__(
        self,
        *args,
        llm_api_key: str | None = None,
        llm_api_base: str | None = None,
        embedding_api_key: str | None = None,
        embedding_api_base: str | None = None,
        config_path: str | None = None,
        enable_logo: bool = True,
        **kwargs,
    ):
        super().__init__(
            *args,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
            embedding_api_key=embedding_api_key,
            embedding_api_base=embedding_api_base,
            service_config=None,
            parser=ReMeConfigParser,
            config_path=config_path,
            enable_logo=enable_logo,
            **kwargs,
        )

    async def async_execute(self, name: str, **kwargs) -> dict:
        """Execute a flow asynchronously and return the result as a dictionary."""
        return (await self.execute_flow(name=name, **kwargs)).model_dump()


def main():
    """Main entry point for running ReMe application from command line."""
    with ReMeApp(*sys.argv[1:]) as app:
        app.run_service()


if __name__ == "__main__":
    main()
