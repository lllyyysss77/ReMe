"""Service context."""

import os
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from .base_context import BaseContext
from .registry_factory import R
from ..schema import ServiceConfig
from ..utils import MCPClient, print_logo, PydanticConfigParser, init_logger, load_env, run_coro_safely


class ServiceContext(BaseContext):
    """Service context."""

    def __init__(
        self,
        *args,
        llm_api_key: str | None = None,
        llm_api_base: str | None = None,
        embedding_api_key: str | None = None,
        embedding_api_base: str | None = None,
        service_config: ServiceConfig | None = None,
        parser: type[PydanticConfigParser] | None = None,
        config_path: str | None = None,
        enable_logo: bool = True,
        llm: dict | None = None,
        embedding_model: dict | None = None,
        vector_store: dict | None = None,
        token_counter: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        # Set environment variables
        load_env()
        self._update_env("REME_LLM_API_KEY", llm_api_key)
        self._update_env("REME_LLM_BASE_URL", llm_api_base)
        self._update_env("REME_EMBEDDING_API_KEY", embedding_api_key)
        self._update_env("REME_EMBEDDING_BASE_URL", embedding_api_base)

        # Use default parser if not provided
        parser_class = parser if parser is not None else PydanticConfigParser
        self.parser = parser_class(ServiceConfig)

        # Service configuration
        if service_config is None:
            input_args = []
            if config_path:
                input_args.append(f"config={config_path}")
            if args:
                input_args.extend(args)
            if kwargs:
                input_args.extend([f"{k}={v}" for k, v in kwargs.items()])
            service_config = self.parser.parse_args(*input_args)
        self.service_config: ServiceConfig = service_config

        # Initialize logger
        if self.service_config.init_logger:
            init_logger()

        # Update service config with provided arguments
        if llm:
            self.update_section_config("llm", **llm)
        if embedding_model:
            self.update_section_config("embedding_model", **embedding_model)
        if token_counter:
            self.update_section_config("token_counter", **token_counter)
        if vector_store:
            self.update_section_config("vector_store", **vector_store)

        # Print the ReMe logo if enabled in configuration.
        self.service_config.enable_logo = enable_logo
        if self.service_config.enable_logo:
            print_logo(service_config=self.service_config)

        # Service configuration and runtime settings
        self.language: str = self.service_config.language
        self.thread_pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=service_config.thread_pool_max_workers)

        # Initialize Ray for distributed computing if configured
        if self.service_config.ray_max_workers > 1:
            import ray

            ray.init(num_cpus=self.service_config.ray_max_workers)

        from ..llm import BaseLLM
        from ..embedding import BaseEmbeddingModel
        from ..vector_store import BaseVectorStore
        from ..token_counter import BaseTokenCounter
        from ..flow import BaseFlow, ExpressionFlow
        from ..service import BaseService

        # Initialize LLM instances
        self.llms: dict[str, BaseLLM] = {}
        for name, config in self.service_config.llm.items():
            self.llms[name] = R.llm[config.backend](model_name=config.model_name, **config.model_extra)

        # Initialize Embedding model instances
        self.embedding_models: dict[str, BaseEmbeddingModel] = {}
        for name, config in self.service_config.embedding_model.items():
            self.embedding_models[name] = R.embedding_model[config.backend](
                model_name=config.model_name,
                **config.model_extra,
            )

        # Initialize Token counter instances
        self.token_counters: dict[str, BaseTokenCounter] = {}
        for name, config in self.service_config.token_counter.items():
            self.token_counters[name] = R.token_counter[config.backend](
                model_name=config.model_name,
                **config.model_extra,
            )

        # Initialize Vector store instances
        self.vector_stores: dict[str, BaseVectorStore] = {}
        for name, config in self.service_config.vector_store.items():
            self.vector_stores[name] = R.vector_store[config.backend](
                collection_name=config.collection_name,
                embedding_model=self.embedding_models[config.embedding_model],
                thread_pool=self.thread_pool,
                **config.model_extra,
            )

        # Initialize flow instances
        self.flows: dict[str, BaseFlow] = {}
        for name, flow_cls in R.flow.items():
            if not self._filter_flows(name):
                continue
            flow: "BaseFlow" = flow_cls(name=name, service_context=self)
            self.flows[flow.name] = flow

        # Initialize flow instances from service config
        for name, flow_config in self.service_config.flow.items():
            if not self._filter_flows(name):
                continue
            flow_config.name = name
            flow: BaseFlow = ExpressionFlow(flow_config=flow_config, service_context=self)
            self.flows[flow.name] = flow

        # Initialize service instance
        self.service: BaseService = R.service[self.service_config.backend](service_context=self)

        # MCP server mapping: maps server_name -> {tool_name: ToolCall}
        if self.service_config.mcp_servers:
            self.mcp_server_mapping: dict[str, dict] = run_coro_safely(self.prepare_mcp_servers())
        else:
            self.mcp_server_mapping: dict[str, dict] = {}

    @staticmethod
    def _update_env(key: str, value: str | None):
        """Update environment variable if value is provided."""
        if value:
            os.environ[key] = value

    def update_section_config(self, section_name: str, **kwargs):
        """Update a specific section of the service config with new values."""
        section_dict: dict = getattr(self.service_config, section_name)
        if "default" not in section_dict:
            raise KeyError(f"Default `{section_name}` config not found")

        current_config = section_dict["default"]
        section_dict["default"] = current_config.model_copy(update=kwargs, deep=True)

    def _filter_flows(self, name: str) -> bool:
        """Filter flows based on enabled_flows and disabled_flows configuration."""
        if self.service_config.enabled_flows:
            return name in self.service_config.enabled_flows
        elif self.service_config.disabled_flows:
            return name not in self.service_config.disabled_flows
        else:
            return True

    async def prepare_mcp_servers(self):
        """Prepare and initialize MCP server connections."""
        mcp_client = MCPClient(config={"mcpServers": self.service_config.mcp_servers})
        for server_name in self.service_config.mcp_servers.keys():
            try:
                # Retrieve all available tool calls from this MCP server
                tool_calls = await mcp_client.list_tool_calls(server_name=server_name, return_dict=False)

                # Build mapping: tool_name -> ToolCall for quick lookup
                self.mcp_server_mapping[server_name] = {tool_call.name: tool_call for tool_call in tool_calls}

                # Log discovered tools for debugging
                for tool_call in tool_calls:
                    logger.info(f"list_tool_calls: {server_name}@{tool_call.name} {tool_call.simple_input_dump()}")

            except Exception as e:
                logger.exception(f"list_tool_calls: {server_name} error: {e}")

    async def close(self):
        """Close all service components asynchronously."""
        for _, vector_store in self.vector_stores.items():
            await vector_store.close()

        for _, llm in self.llms.items():
            await llm.close()

        for _, embedding_model in self.embedding_models.items():
            await embedding_model.close()

        self.shutdown_thread_pool()
        self.shutdown_ray()

    def close_sync(self):
        """Close all service components synchronously."""
        for _, vector_store in self.vector_stores.items():
            run_coro_safely(vector_store.close())

        for _, llm in self.llms.items():
            llm.close_sync()

        for _, embedding_model in self.embedding_models.items():
            embedding_model.close_sync()

        self.shutdown_thread_pool()
        self.shutdown_ray()

    def shutdown_thread_pool(self, wait: bool = True):
        """Shutdown the thread pool executor."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=wait)

    def shutdown_ray(self, wait: bool = True):
        """Shutdown Ray cluster if it was initialized."""
        if self.service_config and self.service_config.ray_max_workers > 1:
            import ray

            ray.shutdown(_exiting_interpreter=not wait)
