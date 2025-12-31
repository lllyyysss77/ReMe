"""Base operator class for LLM workflow execution and composition."""

import asyncio
import copy
import inspect
from pathlib import Path
from typing import Callable, Any, Union

from loguru import logger
from tqdm import tqdm

from ..context import RuntimeContext, PromptHandler, C, BaseContext
from ..embedding import BaseEmbeddingModel
from ..llm import BaseLLM
from ..schema import ToolCall, ToolAttr
from ..token_counter import BaseTokenCounter
from ..utils import camel_to_snake, CacheHandler, timer
from ..vector_store import BaseVectorStore


class BaseOp:
    """Base operator class for LLM workflow execution and composition."""

    def __new__(cls, *args, **kwargs):
        """Capture initialization arguments for object cloning."""
        instance = super().__new__(cls)
        instance._init_args = copy.copy(args)
        instance._init_kwargs = copy.copy(kwargs)
        return instance

    def __init__(
        self,
        name: str = "",
        async_mode: bool = True,
        language: str = "",
        prompt_name: str = "",
        llm: str | BaseLLM = "default",
        embedding_model: str | BaseEmbeddingModel = "default",
        vector_store: str | BaseVectorStore = "default",
        token_counter: str | BaseTokenCounter = "default",
        enable_cache: bool = False,
        cache_path: str = "cache/op",
        sub_ops: Union[list["BaseOp"], dict[str, "BaseOp"], "BaseOp", None] = None,
        input_mapping: dict[str, str] | None = None,
        output_mapping: dict[str, str] | None = None,
        enable_tool_response: bool = False,
        enable_sync_thread_pool: bool = True,
        max_retries: int = 1,
        raise_exception: bool = False,
        **kwargs,
    ):
        """Initialize operator configurations and internal state."""
        self.name = name or camel_to_snake(self.__class__.__name__)
        self.async_mode = async_mode
        self.language = language or C.language
        self.prompt = self._get_prompt_handler(prompt_name)

        self._llm = llm
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._token_counter = token_counter

        self.enable_cache = enable_cache
        self.cache_path = cache_path
        self.sub_ops = BaseContext[str, BaseOp]()
        self.add_sub_ops(sub_ops)

        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
        self.enable_tool_response = enable_tool_response
        self.enable_sync_thread_pool = enable_sync_thread_pool
        self.max_retries = max(1, max_retries)
        self.raise_exception = raise_exception
        self.op_params = kwargs

        self._pending_tasks: list = []
        self.context: RuntimeContext | None = None
        self._cache: CacheHandler | None = None
        self._tool_call: ToolCall | None = None

    def _get_prompt_handler(self, prompt_name: str) -> PromptHandler:
        """Load prompt configuration from the associated YAML file."""
        path = Path(inspect.getfile(self.__class__))
        path = path.with_stem(prompt_name) if prompt_name else path
        return PromptHandler(language=self.language).load_prompt_by_file(path.with_suffix(".yaml"))

    def _build_tool_call(self) -> ToolCall | None:
        """Build and return the tool call schema; override in subclasses."""

    def _validate_inputs(self):
        """Ensure all required tool inputs are present in context."""
        if self.tool_call:
            parameters = self.tool_call.parameters
            if parameters.type == "object" and parameters.properties:
                required_list = parameters.required or []
                required_keys = {k: (k in required_list) for k in parameters.properties.keys()}
                self.context.validate_required_keys(required_keys, self.name)

    def _handle_failure(self, e: Exception, attempt: int):
        """Log failures and handle final retry logic."""
        logger.exception(f"{self.name} failed (attempt {attempt + 1}): {e}")
        if attempt == self.max_retries - 1:
            if self.raise_exception:
                raise e

            if self.tool_call:
                self.output = f"{self.name} failed: {e}"

    @property
    def tool_call(self) -> ToolCall:
        """Lazily construct and return the tool call metadata."""
        if self._tool_call is None:
            self._tool_call = self._build_tool_call()
            assert self._tool_call, "tool_call is not defined!"
            self._tool_call.name = self._tool_call.name or self.name
            if not self._tool_call.output.properties:
                self._tool_call.output = ToolAttr(
                    type="object",
                    properties={
                        f"{self.name}_result": ToolAttr(type="string", description=f"Execution result of {self.name}"),
                    },
                )
        return self._tool_call

    @property
    def input_dict(self) -> dict:
        """Extract required and optional inputs from context based on schema."""
        parameters = self.tool_call.parameters
        if parameters.type != "object" or not parameters.properties:
            return {}
        required_keys = set(parameters.required or [])
        return {k: self.context[k] for k in parameters.properties.keys() if (k in required_keys or k in self.context)}

    @property
    def output(self) -> Any:
        """Get the single output value from context."""
        output_properties = self.tool_call.output.properties
        if not output_properties:
            return None
        keys = list(output_properties.keys())
        return self.context[keys[0]]

    @output.setter
    def output(self, value: Any):
        """Set the single output value into context."""
        output_properties = self.tool_call.output.properties
        if not output_properties:
            return
        keys = list(output_properties.keys())
        self.context[keys[0]] = value

    @property
    def cache(self) -> CacheHandler:
        """Access the operator-specific cache handler."""
        assert self.enable_cache, "Cache is disabled!"
        if not self._cache:
            self._cache = CacheHandler(f"{self.cache_path}/{self.name}")
        return self._cache

    @property
    def llm(self) -> BaseLLM:
        """Lazily initialize and return the LLM instance."""
        if isinstance(self._llm, str):
            cfg = C.service_config.llm[self._llm]
            self._llm = C.get_llm_class(cfg.backend)(model_name=cfg.model_name, **cfg.model_extra)
        return self._llm

    @property
    def embedding_model(self) -> BaseEmbeddingModel:
        """Lazily initialize and return the embedding model instance."""
        if isinstance(self._embedding_model, str):
            cfg = C.service_config.embedding_model[self._embedding_model]
            self._embedding_model = C.get_embedding_model_class(cfg.backend)(
                model_name=cfg.model_name,
                **cfg.model_extra,
            )
        return self._embedding_model

    @property
    def vector_store(self) -> BaseVectorStore:
        """Lazily initialize and return the vector store instance."""
        if isinstance(self._vector_store, str):
            self._vector_store = C.get_vector_store(self._vector_store)
        return self._vector_store

    @property
    def token_counter(self) -> BaseTokenCounter:
        """Lazily initialize and return the token counter instance."""
        if isinstance(self._token_counter, str):
            cfg = C.service_config.token_counter[self._token_counter]
            self._token_counter = C.get_token_counter_class(cfg.backend)(
                model_name=cfg.model_name,
                **cfg.model_extra,
            )
        return self._token_counter

    async def before_execute(self):
        """Prepare context and validate before async execution."""
        self.context.apply_mapping(self.input_mapping)
        self._validate_inputs()

    async def execute(self):
        """Define core async logic in subclasses."""

    async def after_execute(self):
        """Finalize context and mappings after async execution."""
        self.context.apply_mapping(self.output_mapping)
        if self.tool_call and self.enable_tool_response:
            self.context.response.answer = self.output

        if not isinstance(self._llm, str) and hasattr(self._llm, "close"):
            await self._llm.close()
        if not isinstance(self._embedding_model, str) and hasattr(self._embedding_model, "close"):
            await self._embedding_model.close()

    def before_execute_sync(self):
        """Prepare context and validate before sync execution."""
        self.context.apply_mapping(self.input_mapping)
        self._validate_inputs()

    def execute_sync(self):
        """Define core sync logic in subclasses."""

    def after_execute_sync(self):
        """Finalize context and mappings after sync execution."""
        self.context.apply_mapping(self.output_mapping)
        if self.tool_call and self.enable_tool_response:
            self.context.response.answer = self.output

        if not isinstance(self._llm, str) and hasattr(self._llm, "close_sync"):
            self._llm.close_sync()
        if not isinstance(self._embedding_model, str) and hasattr(self._embedding_model, "close_sync"):
            self._embedding_model.close_sync()

    @timer
    def call_sync(self, context: RuntimeContext = None, **kwargs):
        """Execute the operator synchronously with retry logic."""
        self.context = RuntimeContext.from_context(context, **kwargs)
        for i in range(self.max_retries):
            try:
                self.before_execute_sync()
                self.execute_sync()
                self.after_execute_sync()
                break
            except Exception as e:
                self._handle_failure(e, i)
        return self.output if self.tool_call else None

    async def call(self, context: RuntimeContext = None, **kwargs):
        """Execute the operator asynchronously with retry logic."""
        self.context = RuntimeContext.from_context(context, **kwargs)
        for i in range(self.max_retries):
            try:
                await self.before_execute()
                await self.execute()
                await self.after_execute()
                break
            except Exception as e:
                self._handle_failure(e, i)
        return self.output if self.tool_call else None

    def submit_sync_task(self, fn: Callable, *args, **kwargs) -> "BaseOp":
        """Submit a task to the thread pool or local queue."""
        task = C.thread_pool.submit(fn, *args, **kwargs) if self.enable_sync_thread_pool else (fn, args, kwargs)
        self._pending_tasks.append(task)
        return self

    def submit_async_task(self, coro_fn: Callable, *args, **kwargs) -> "BaseOp":
        """Submit an async task to the pending tasks queue."""
        task = coro_fn(*args, **kwargs)
        self._pending_tasks.append(task)
        return self

    def join_sync_tasks(self, task_desc: str = None) -> list:
        """Wait for all pending sync tasks and return flattened results."""
        results = []
        for task in tqdm(self._pending_tasks, desc=task_desc or self.name):
            res = task.result() if self.enable_sync_thread_pool else task[0](*task[1], **task[2])
            if res:
                results.extend(res if isinstance(res, list) else [res])
        self._pending_tasks.clear()
        return results

    async def join_async_tasks(self, return_exceptions: bool = True) -> list:
        """Wait for all pending async tasks and aggregate results."""
        try:
            raw_results = await asyncio.gather(*self._pending_tasks, return_exceptions=return_exceptions)
            results = []
            for res in raw_results:
                if isinstance(res, Exception):
                    logger.error(f"Async task failed: {res}")
                    continue
                if res:
                    results.extend(res if isinstance(res, list) else [res])
            return results
        finally:
            self._pending_tasks.clear()

    def add_sub_ops(self, sub_ops: Union[list["BaseOp"], dict[str, "BaseOp"], "BaseOp", None]):
        """Add child operators to this operator's sub_ops context."""
        if not sub_ops:
            return

        if isinstance(sub_ops, dict):
            ops_dict = sub_ops
        else:
            ops_dict = {op.name: op for op in (sub_ops if isinstance(sub_ops, list) else [sub_ops])}

        for name, op in ops_dict.items():
            assert self.async_mode == op.async_mode, "Async mode mismatch!"
            self.sub_ops[name] = op

    def add_sub_op(self, sub_op: "BaseOp"):
        """Add a single child operator to this operator's sub_ops context."""
        self.add_sub_ops(sub_op)

    def __lshift__(self, ops):
        """Operator overload for adding sub-operators."""
        self.add_sub_ops(ops)
        return self

    def __rshift__(self, op: "BaseOp"):
        """Operator overload for sequential execution composition."""
        from .sequential_op import SequentialOp

        seq = SequentialOp(sub_ops=[self], async_mode=self.async_mode)
        seq.add_sub_ops(op.sub_ops if isinstance(op, SequentialOp) else op)
        return seq

    def __or__(self, op: "BaseOp"):
        """Operator overload for parallel execution composition."""
        from .parallel_op import ParallelOp

        par = ParallelOp(sub_ops=[self], async_mode=self.async_mode)
        par.add_sub_ops(op.sub_ops if isinstance(op, ParallelOp) else op)
        return par
