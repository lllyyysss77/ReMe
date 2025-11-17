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


"""
1. token counter 计数，最长上下文X20%=20K
  1. openai  tiktoken / hagggingface / modelscope / rule-based
          self.token_counter.count()
2. tool:
  1. dump_tool(write_file)
  2. grep / rip_grep / read_file
3. compact:
  if > threashold(20K):
    tool_call -> write_file
  写一个引用：/xx/xxx/xxx.txt。是否保留前面的token（）
4. compress:
  1. prompt 
  2. context -> write_file


1. case study:
2. messages
  1. agentscope: reme
  2. http远程 reme

"""