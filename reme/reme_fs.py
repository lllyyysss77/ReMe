"""ReMe File System"""

from .config import ReMeConfigParser
from .core import Application


class ReMeFs(Application):
    """ReMe File System"""

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
        working_dir: str = "./agent",
        **kwargs,
    ):
        """Initialize ReMe with config."""
        super().__init__(
            *args,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
            embedding_api_key=embedding_api_key,
            embedding_api_base=embedding_api_base,
            enable_logo=enable_logo,
            parser=ReMeConfigParser,
            llm=llm,
            embedding_model=embedding_model,
            vector_store=vector_store,
            token_counter=token_counter,
            **kwargs,
        )

        self.working_dir: str = working_dir
