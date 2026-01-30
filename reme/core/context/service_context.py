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
    from ..token_counter import BaseTokenCounter
    from ..flow import BaseFlow
    from ..service import BaseService


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
        self.service_config: ServiceConfig = self._build_service_config(
            *args,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
            embedding_api_key=embedding_api_key,
            embedding_api_base=embedding_api_base,
            service_config=service_config,
            parser=parser,
            config_path=config_path,
            enable_logo=enable_logo,
            llm=llm,
            embedding_model=embedding_model,
            vector_store=vector_store,
            token_counter=token_counter,
            **kwargs,
        )

        if self.service_config.init_logger:
            init_logger()

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
        self.flows: dict[str, "BaseFlow"] = {}
        self.mcp_server_mapping: dict[str, dict] = {}
        self.service: "BaseService" = R.service[self.service_config.backend](service_context=self)

        self._build_flows()

    def _build_service_config(
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
    ) -> ServiceConfig:

        load_env()
        self._update_env("REME_LLM_API_KEY", llm_api_key)
        self._update_env("REME_LLM_BASE_URL", llm_api_base)
        self._update_env("REME_EMBEDDING_API_KEY", embedding_api_key)
        self._update_env("REME_EMBEDDING_BASE_URL", embedding_api_base)

        if service_config is None:
            parser_class = parser if parser is not None else PydanticConfigParser
            parser = parser_class(ServiceConfig)
            input_args = []
            if config_path:
                input_args.append(f"config={config_path}")
            if args:
                input_args.extend(args)
            if kwargs:
                input_args.extend([f"{k}={v}" for k, v in kwargs.items()])
            service_config = parser.parse_args(*input_args)

        service_config.enable_logo = enable_logo
        if llm:
            self._update_section_config(service_config, "llm", **llm)
        if embedding_model:
            self._update_section_config(service_config, "embedding_model", **embedding_model)
        if token_counter:
            self._update_section_config(service_config, "token_counter", **token_counter)
        if vector_store:
            self._update_section_config(service_config, "vector_store", **vector_store)
        return service_config

    @staticmethod
    def _update_env(key: str, value: str | None):
        """Update environment variable if value is provided."""
        if value:
            os.environ[key] = value

    @staticmethod
    def _update_section_config(service_config: ServiceConfig, section_name: str, **kwargs):
        """Update a specific section of the service config with new values."""
        section_dict: dict = getattr(service_config, section_name)
        if "default" not in section_dict:
            raise KeyError(f"Default `{section_name}` config not found")
        current_config = section_dict["default"]
        section_dict["default"] = current_config.model_copy(update=kwargs, deep=True)

    def _build_flows(self):
        expression_flow_cls = None
        for name, flow_cls in R.flow.items():
            if not self._filter_flows(name):
                continue

            if name == "ExpressionFlow":
                expression_flow_cls = flow_cls
            else:
                flow: "BaseFlow" = flow_cls(name=name, service_context=self)
                self.flows[flow.name] = flow

        if expression_flow_cls is not None:
            for name, flow_config in self.service_config.flow.items():
                if not self._filter_flows(name):
                    continue
                flow_config.name = name
                flow: BaseFlow = expression_flow_cls(flow_config=flow_config, service_context=self)  # noqa
                self.flows[flow.name] = flow
        else:
            logger.info("No expression flow found, please check your configuration.")

    async def start(self):
        """Start the service context by initializing all configured components."""
        for name, config in self.service_config.llm.items():
            self.llms[name] = R.llm[config.backend](model_name=config.model_name, **config.model_extra)

        for name, config in self.service_config.embedding_model.items():
            self.embedding_models[name] = R.embedding_model[config.backend](
                model_name=config.model_name,
                **config.model_extra,
            )

        for name, config in self.service_config.token_counter.items():
            self.token_counters[name] = R.token_counter[config.backend](
                model_name=config.model_name,
                **config.model_extra,
            )

        for name, config in self.service_config.vector_store.items():
            self.vector_stores[name] = R.vector_store[config.backend](
                collection_name=config.collection_name,
                embedding_model=self.embedding_models[config.embedding_model],
                thread_pool=self.thread_pool,
                **config.model_extra,
            )
            await self.vector_stores[name].create_collection(config.collection_name)

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
