from agentscope.model import DashScopeChatModel
from agentscope.model import OpenAIChatModel

from ..registry_factory import R

R.as_llms.register(OpenAIChatModel, "openai")
R.as_llms.register(DashScopeChatModel, "dashscope")
