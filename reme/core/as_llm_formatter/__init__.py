from agentscope.formatter import DashScopeChatFormatter
from agentscope.formatter import OpenAIChatFormatter

from ..registry_factory import R

R.as_llm_formatters.register(OpenAIChatFormatter, "openai")
R.as_llm_formatters.register(DashScopeChatFormatter, "dashscope")
