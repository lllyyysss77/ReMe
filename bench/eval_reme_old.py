"""ReMe evaluation script for HaluMem-like benchmarks."""

import asyncio
import copy
import json
import os
import re
import time
from datetime import datetime, timezone

from tqdm import tqdm

from reme_ai.core_old.enumeration import Role
from reme_ai.core_old.schema import Message, MemoryNode
from reme_ai.reme import ReMe

TEMPLATE_REME = """Memories for user {user_id}:

    {memories}
"""

RETRY_TIMES = 3
WAIT_TIME = 2

# Default prompt for answering questions with memory context
PROMPT_REME = """You are a helpful AI assistant with access to the user's memories.
Use the following context to answer the user's question accurately.

Context:
{context}

Question: {question}

Please provide a detailed and accurate answer based on the available context.
If the context doesn't contain enough information to answer the question, say so clearly."""


async def add_memory_async(
    reme: ReMe,
    user_id: str,
    messages: list[dict],
    description: str = "",
):
    """Add memory to ReMe system asynchronously."""
    start = time.time()

    result = await reme.summary(
        messages=messages,
        user_id=user_id,
        description=description,
        memory_mode="personal",
    )

    duration_ms = (time.time() - start) * 1000
    return result, duration_ms


async def search_memory_async(
    reme: ReMe,
    query: str,
    user_id: str,
    top_k: int = 20,
):
    """Search memory from ReMe system asynchronously."""
    start = time.time()

    result = await reme.retrieve(
        query=query,
        user_id=user_id,
        memory_mode="personal",
        top_k=top_k,
    )

    # Format the context
    context = TEMPLATE_REME.format(
        user_id=user_id,
        memories=result if isinstance(result, str) else json.dumps(result, indent=4, ensure_ascii=False),
    )

    duration_ms = (time.time() - start) * 1000

    return context, result, duration_ms


async def llm_request_async(reme: ReMe, prompt: str):
    """Make LLM request using ReMe's llm."""
    messages = [
        Message(role=Role.SYSTEM, content="You are a helpful assistant."),
        Message(role=Role.USER, content=prompt),
    ]

    response = await reme.llm.chat(messages=messages)
    return response.content


def extract_user_name(persona_info: str):
    """Extract user name from persona info."""
    match = re.search(r"Name:\s*(.*?); Gender:", persona_info)

    if match:
        username = match.group(1).strip()
        return username
    else:
        raise ValueError("No name found.")


async def _process_session_questions(
    session: dict,
    new_session: dict,
    reme: ReMe,
    user_name: str,
    top_k_value: int,
) -> None:
    """Process questions for a session."""
    if "questions" not in session:
        return

    new_session["questions"] = []

    for qa in session["questions"]:
        context, _, duration_ms = await search_memory_async(
            reme=reme,
            query=qa["question"],
            user_id=user_name,
            top_k=top_k_value,
        )

        new_qa = copy.deepcopy(qa)
        new_qa["context"] = context
        new_qa["search_duration_ms"] = duration_ms

        prompt = PROMPT_REME.format(
            context=context,
            question=qa["question"],
        )

        start_time = time.time()
        response = await llm_request_async(reme, prompt)
        new_qa["system_response"] = response
        new_qa["response_duration_ms"] = (time.time() - start_time) * 1000

        new_session["questions"].append(new_qa)


async def process_user_async(
    user_data: dict,
    top_k_value: int,
    save_path: str,
    reme: ReMe,
):
    """Process a single user's data asynchronously."""
    user_name = extract_user_name(user_data["persona_info"])
    sessions = user_data["sessions"]

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    tmp_file = os.path.join(tmp_dir, f"{user_data['uuid']}.json")

    # Clear existing memories for this user
    collection_name = f"reme_eval_{user_name}".replace(" ", "_").lower()
    await reme.vector_store.delete_collection(collection_name)

    # Update collection name for this user
    reme.vector_store.set_collection_name(collection_name)

    new_user_data = {
        "uuid": user_data["uuid"],
        "user_name": user_name,
        "sessions": [],
    }

    for session in tqdm(sessions, total=len(sessions), desc=f"Processing user {user_name}"):
        new_session = {
            "memory_points": session["memory_points"],
            "dialogue": session["dialogue"],
        }

        # Add messages to ReMe
        dialogue = session["dialogue"]
        formatted_dialogue = [
            {
                "role": turn["role"],
                "content": turn["content"],
                "time_created": datetime.strptime(turn["timestamp"], "%b %d, %Y, %H:%M:%S")
                .replace(tzinfo=timezone.utc)
                .strftime("%Y-%m-%d %H:%M:%S"),
            }
            for turn in dialogue
        ]

        # Add memory - process every 2 messages
        result = []
        total_duration_ms = 0
        batch_size = 4

        for i in range(0, len(formatted_dialogue), batch_size):
            batch = formatted_dialogue[i : i + batch_size]
            batch_result, duration_ms = await add_memory_async(
                reme=reme,
                user_id=user_name,
                messages=batch,
            )
            result.extend(batch_result)
            total_duration_ms += duration_ms

        duration_ms = total_duration_ms

        memories = []
        for memory_mode in result:
            if not isinstance(memory_mode, MemoryNode):
                continue

            memories.append(memory_mode.content)

        print(memories)

        if session.get("is_generated_qa_session", False):
            new_session["add_dialogue_duration_ms"] = duration_ms
            new_session["is_generated_qa_session"] = True
            del new_session["dialogue"]
            del new_session["memory_points"]
            new_user_data["sessions"].append(new_session)
            continue

        # Store the result from summary
        new_session["extracted_memories"] = memories
        new_session["add_dialogue_duration_ms"] = duration_ms

        # Search updated memories for memory points
        # for memory in new_session["memory_points"]:
        #     if memory["is_update"] == "False" or not memory["original_memories"]:
        #         continue
        #
        #     _, memories_from_system, duration_ms = await search_memory_async(
        #         reme=reme,
        #         query=memory["memory_content"],
        #         user_id=user_name,
        #         top_k=10,
        #     )
        #
        #     memory["memories_from_system"] = str(memories_from_system)

        # Process questions
        await _process_session_questions(session, new_session, reme, user_name, top_k_value)

        new_user_data["sessions"].append(new_session)
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(new_user_data, f, ensure_ascii=False, indent=2)
        # raise NotImplementedError

    # Save results
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(new_user_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved user {user_name} to {tmp_file}")
    return {"uuid": user_data["uuid"], "status": "ok", "path": tmp_file}


def iter_jsonl(file_path: str):
    """Iterate over lines in a JSONL file."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


async def main_async(
    data_path_arg: str,
    version_arg: str = "default",
    top_k_arg: int = 20,
):
    """Main evaluation function."""
    frame = "reme"
    save_path = f"bench_results/{frame}-{version_arg}/"
    os.makedirs(save_path, exist_ok=True)

    output_file = os.path.join(save_path, f"{frame}_eval_results.jsonl")
    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    start_time = time.time()

    # Initialize ReMe instance (will reuse for all users)
    reme = ReMe()

    # Load all user data
    user_data_list = list(iter_jsonl(data_path_arg))
    total_users = len(user_data_list)

    print(f"Processing {total_users} users sequentially...")

    # Sequential processing
    for idx, user_data in enumerate(user_data_list, 1):
        result = await process_user_async(user_data, top_k_arg, save_path, reme)
        print(f"[{idx}/{total_users}] ✅ Finished {user_data['uuid']} ({result['status']})")

    # Combine all results into final output
    with open(output_file, "w", encoding="utf-8") as f_out:
        for file in os.listdir(tmp_dir):
            if file.endswith(".json"):
                file_path = os.path.join(tmp_dir, file)
                with open(file_path, "r", encoding="utf-8") as f_in:
                    data = json.load(f_in)
                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    print(f"✅ All done in {elapsed:.2f}s")
    print(f"✅ Final results saved to: {output_file}")


def main(
    data_path_arg: str,
    version_arg: str = "default",
    top_k_arg: int = 20,
):
    """Synchronous entry point for main evaluation."""
    asyncio.run(main_async(data_path_arg, version_arg, top_k_arg))


if __name__ == "__main__":
    # Example usage - update these paths as needed
    # Note: Don't use HaluMem-long.jsonl directly as each line is too large
    # Instead, create a smaller test dataset or use a different data file

    DEFAULT_DATA_PATH = "/Users/yuli/workspace/HaluMem/data/HaluMem-Long.jsonl"
    DEFAULT_VERSION = "test"
    DEFAULT_TOP_K = 20

    main(
        data_path_arg=DEFAULT_DATA_PATH,
        version_arg=DEFAULT_VERSION,
        top_k_arg=DEFAULT_TOP_K,
    )
