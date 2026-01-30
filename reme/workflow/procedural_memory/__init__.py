# summary retriever vector store 的 op

# 1. BaseAsyncOp -> reme.core.op.BaseOp
# 2. @C.register_op() -> R.op.register()(MergeMemoryOp)
# for name in __all__:
#     tool_class = globals()[name]
#     R.op.register()(tool_class)
# 3. class name 不一定叫 op
# 4. async def async_execute(self): —》async def execute(self):
# 5. self.llm.chat
# 6.     file_path: str = __file__ 不需要
# 7. self.op_params.get("enable_llm_rerank", True) 都改成 self.context.get("enable_llm_rerank", True)
#
#
# # 跑起来 reme = "reme_ai.main:main" -> reme = "reme.reme_app:main"
#
# app = ReMeApp()
#
#
# def test_search():
#     """Test search tool operations.
#
#     Tests DashscopeSearch, MockSearch, and TavilySearch operations
#     with a sample query to verify they work correctly.
#     """
#     from reme.tool.search import DashscopeSearch, MockSearch, TavilySearch
#
#     query = "今天杭州的天气如何？"
#
#     for op in [
#         DashscopeSearch(),
#         MockSearch(),
#         TavilySearch(),
#     ]:
#         print("\n" + "=" * 60)
#         print(f"Testing {op.__class__.__name__}")
#         print("=" * 60)
#         print(f"Query: {query}")
#         output = asyncio.run(op.call(query=query, service_context=app.service_context))
#
# self.context.query
# app.service_context 保证了 self.llm emb vectorstore

# examples
# bench 里的llm ，辛苦改成 app = ReMeApp() app.default_llm
# clear && pre-commit run --all-files
