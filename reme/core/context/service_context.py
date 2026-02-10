"""Service context."""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from loguru import logger

from .base_context import BaseContext
from .registry_factory import R
from ..schema import ServiceConfig
from ..utils import load_env, MCPClient, print_logo, PydanticConfigParser, init_logger

if TYPE_CHECKING:
    from ..llm import BaseLLM
    from ..embedding import BaseEmbeddingModel
    from ..vector_store import BaseVectorStore
    from ..memory_store import BaseMemoryStore
    from ..token_counter import BaseTokenCounter
    from ..flow import BaseFlow
    from ..service import BaseService
    from ..file_watcher import BaseFileWatcher


class ServiceContext(BaseContext):
    """Service context."""

    def __init__(
        self,
        *args,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
        service_config: ServiceConfig | None = None,
        parser: type[PydanticConfigParser] | None = None,
        config_path: str | None = None,
        enable_logo: bool = True,
        log_to_console: bool = True,
        default_llm_config: dict | None = None,
        default_embedding_model_config: dict | None = None,
        default_vector_store_config: dict | None = None,
        default_memory_store_config: dict | None = None,
        default_token_counter_config: dict | None = None,
        default_file_watcher_config: dict | None = None,
        **kwargs,
    ):
        super().__init__()

        load_env()
        self._update_env("REME_LLM_API_KEY", llm_api_key)
        self._update_env("REME_LLM_BASE_URL", llm_base_url)
        self._update_env("REME_EMBEDDING_API_KEY", embedding_api_key)
        self._update_env("REME_EMBEDDING_BASE_URL", embedding_base_url)

        if service_config is None:
            parser_class = parser if parser is not None else PydanticConfigParser
            parser = parser_class(ServiceConfig)
            input_args = []
            if config_path:
                input_args.append(f"config={config_path}")
            if args:
                input_args.extend(args)

            if default_llm_config:
                self._update_section_config(kwargs, "llms", **default_llm_config)
            if default_embedding_model_config:
                self._update_section_config(kwargs, "embedding_models", **default_embedding_model_config)
            if default_token_counter_config:
                self._update_section_config(kwargs, "token_counters", **default_token_counter_config)
            if default_vector_store_config:
                self._update_section_config(kwargs, "vector_stores", **default_vector_store_config)
            if default_memory_store_config:
                self._update_section_config(kwargs, "memory_stores", **default_memory_store_config)
            if default_file_watcher_config:
                self._update_section_config(kwargs, "file_watchers", **default_file_watcher_config)
            kwargs["enable_logo"] = enable_logo
            kwargs["log_to_console"] = log_to_console
            logger.info(f"update with args: {input_args} kwargs: {kwargs}")
            service_config = parser.parse_args(*input_args, **kwargs)

        self.service_config: ServiceConfig = service_config
        init_logger(log_to_console=self.service_config.log_to_console)

        if self.service_config.enable_logo:
            print_logo(service_config=self.service_config)

        self.language: str = self.service_config.language
        self.thread_pool: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=self.service_config.thread_pool_max_workers,
        )
        if self.service_config.ray_max_workers > 1:
            import ray

            ray.init(num_cpus=self.service_config.ray_max_workers)

        self.llms: dict[str, "BaseLLM"] = {}
        self.embedding_models: dict[str, "BaseEmbeddingModel"] = {}
        self.token_counters: dict[str, "BaseTokenCounter"] = {}
        self.vector_stores: dict[str, "BaseVectorStore"] = {}
        self.memory_stores: dict[str, "BaseMemoryStore"] = {}
        self.file_watchers: dict[str, "BaseFileWatcher"] = {}

        self.flows: dict[str, "BaseFlow"] = {}
        self.mcp_server_mapping: dict[str, dict] = {}
        self.service: "BaseService" = R.services[self.service_config.backend](service_context=self)

        self._build_flows()

    @staticmethod
    def _update_env(key: str, value: str | None):
        """Update environment variable if value is provided."""
        if value:
            os.environ[key] = value

    @staticmethod
    def _update_section_config(config: dict, section_name: str, **kwargs):
        """Update a specific section of the service config with new values."""
        if section_name not in config:
            config[section_name] = {}
        if "default" not in config[section_name]:
            config[section_name]["default"] = {}
        config[section_name]["default"].update(kwargs)

    def _build_flows(self):
        expression_flow_cls = None
        for name, flow_cls in R.flows.items():
            if not self._filter_flows(name):
                continue

            if name == "ExpressionFlow":
                expression_flow_cls = flow_cls
            else:
                flow: "BaseFlow" = flow_cls(name=name, service_context=self)
                self.flows[flow.name] = flow

        if expression_flow_cls is not None:
            for name, flow_config in self.service_config.flows.items():
                if not self._filter_flows(name):
                    continue
                flow_config.name = name
                flow: BaseFlow = expression_flow_cls(flow_config=flow_config, service_context=self)  # noqa
                self.flows[flow.name] = flow
        else:
            logger.info("No expression flow found, please check your configuration.")

    async def start(self):
        """Start the service context by initializing all configured components."""
        for name, config in self.service_config.llms.items():
            self.llms[name] = R.llms[config.backend](model_name=config.model_name, **config.model_extra)

        for name, config in self.service_config.embedding_models.items():
            self.embedding_models[name] = R.embedding_models[config.backend](
                model_name=config.model_name,
                **config.model_extra,
            )

        for name, config in self.service_config.token_counters.items():
            self.token_counters[name] = R.token_counters[config.backend](
                model_name=config.model_name,
                **config.model_extra,
            )

        for name, config in self.service_config.vector_stores.items():
            # Extract config dict and replace special fields with actual instances
            config_dict = config.model_dump(exclude={"backend", "embedding_model"})
            config_dict["embedding_model"] = self.embedding_models[config.embedding_model]
            config_dict["thread_pool"] = self.thread_pool
            self.vector_stores[name] = R.vector_stores[config.backend](**config_dict)
            await self.vector_stores[name].create_collection(config.collection_name)

        for name, config in self.service_config.memory_stores.items():
            # Extract config dict and replace embedding_model string with actual instance
            config_dict = config.model_dump(exclude={"backend", "embedding_model"})
            config_dict["embedding_model"] = self.embedding_models[config.embedding_model]
            self.memory_stores[name] = R.memory_stores[config.backend](**config_dict)
            await self.memory_stores[name].start()

        for name, config in self.service_config.file_watchers.items():
            # Extract config dict and replace memory_store string with actual instance
            config_dict = config.model_dump(exclude={"backend", "memory_store"})
            config_dict["memory_store"] = self.memory_stores[config.memory_store]
            self.file_watchers[name] = R.file_watchers[config.backend](**config_dict)
            await self.file_watchers[name].start()

        if self.service_config.mcp_servers:
            await self.prepare_mcp_servers()

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
                tool_calls = await mcp_client.list_tool_calls(server_name=server_name, return_dict=False)
                self.mcp_server_mapping[server_name] = {tool_call.name: tool_call for tool_call in tool_calls}
                for tool_call in tool_calls:
                    logger.info(f"list_tool_calls: {server_name}@{tool_call.name} {tool_call.simple_input_dump()}")
            except Exception as e:
                logger.exception(f"list_tool_calls: {server_name} error: {e}")

    async def reset_default_collection(self, collection_name: str):
        """Reset the default vector store."""
        await self.vector_stores["default"].reset_collection(collection_name)

    async def close(self):
        """Close all service components asynchronously."""
        for _, vector_store in self.vector_stores.items():
            await vector_store.close()

        for _, memory_store in self.memory_stores.items():
            await memory_store.close()

        for _, file_watcher in self.file_watchers.items():
            await file_watcher.close()

        for _, llm in self.llms.items():
            await llm.close()

        for _, embedding_model in self.embedding_models.items():
            await embedding_model.close()

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
