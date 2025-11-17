from flowllm.core.context import C
from flowllm.core.op import BaseAsyncOp
from flowllm.core.schema import ToolCall


@C.register_op()
class ContextCompactOp(BaseAsyncOp):

    def __init__(self, ratio: 0.3, **kwargs):
        super().__init__(**kwargs)
        self.ratio: float = ratio

    def async_execute(self):
        messages = self.context.messages

        self.llm.achat(messages=messages, tools=None)
        self.token_count()

        self.context["messages"] = messages
        self.context["offloaded_data"] = {"f": "ddd"}
