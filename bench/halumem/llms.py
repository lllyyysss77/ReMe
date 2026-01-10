import asyncio
import json
import logging
import re

from tenacity import retry, stop_after_attempt, wait_random_exponential, before_sleep_log

from reme_ai.core.llm import OpenAILLM
from reme_ai.core.schema import Message
from reme_ai.core.utils import load_env

logger = logging.getLogger(__name__)

load_env()

WAIT_TIME_LOWER = 1
WAIT_TIME_UPPER = 60
RETRY_TIMES = 5


@retry(
    wait=wait_random_exponential(min=WAIT_TIME_LOWER, max=WAIT_TIME_UPPER),
    stop=stop_after_attempt(3),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def llm_request(prompt, **kwargs) -> str:
    llm = OpenAILLM(model_name="qwen3-max")
    assistant_message = await llm.chat(
        messages=[
            Message(
                **{
                    "role": "user",
                    "content": prompt,
                },
            ),
        ],
        **kwargs,
    )
    return assistant_message.content


@retry(
    wait=wait_random_exponential(min=WAIT_TIME_LOWER, max=WAIT_TIME_UPPER),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def llm_request_for_json(prompt, **kwargs):
    content = await llm_request(prompt, **kwargs)

    match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON block found in model output: {content}")

    json_str = match.group(1).strip()
    return json.loads(json_str)


if __name__ == "__main__":
    r = asyncio.run(llm_request_for_json('hello? answer in ```json\n{"answer": "..."}```'))
    print(r)
