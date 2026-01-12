"""
Simplified evaluation script for ReMe on HaluMem benchmark - Question Answering only.

This script performs a simplified evaluation pipeline:
1. Load HaluMem data
2. Process each user's sessions with ReMe (summary + retrieve)
3. Evaluate question answering only
4. Generate metrics and statistics

Usage:
    python bench/halumem/eval_reme_simple.py --data_path /Users/yuli/workspace/HaluMem/data/HaluMem-Medium.jsonl \
        --top_k 20 --user_num 100 --max_concurrency 20
"""

import asyncio
import copy
import json
import os
import re
import time
from datetime import datetime, timezone

from loguru import logger

from eval_tools import (
    _PROMPTS,
    evaluation_for_question,
    evaluation_for_question2,
)
from llms import llm_request
from reme_ai.core.enumeration import MemoryType
from reme_ai.core.schema import MemoryNode
from reme_ai.reme import ReMe

# Initialize ReMe
reme: ReMe = ReMe()


def extract_user_name(persona_info: str):
    """Extract user name from persona info."""
    match = re.search(r"Name:\s*(.*?); Gender:", persona_info)
    if match:
        username = match.group(1).strip()
        return username
    else:
        raise ValueError("No name found.")


def iter_jsonl(file_path: str):
    """Iterate over lines in a JSONL file."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ==================== Main Processing ====================


async def add_memory_async(user_id: str, messages: list[dict]) -> tuple[list[MemoryNode], float]:
    """Add memory to ReMe system asynchronously."""
    start = time.time()
    result = await reme.summary_v2(messages=messages, user_id=user_id)
    duration_ms = (time.time() - start) * 1000
    return result, duration_ms


async def search_memory_async(query: str, user_id: str, top_k: int = 20):
    """Search memory and get LLM response directly."""
    start = time.time()
    memories = await reme.retrieve_v2(query=query, user_id=user_id, top_k=top_k)
    
    # Format the context
    context = f"User: {user_id}\nMemories:\n{memories}"
    
    # Get LLM response directly
    prompt = f"Based on the following context, answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    response = await llm_request(prompt)
    
    duration_ms = (time.time() - start) * 1000
    return response, duration_ms


async def process_user_stage1(
    user_data: dict,
    top_k_value: int,
    save_path: str,
):
    """Process user data through ReMe (summary + retrieve + QA evaluation only)."""
    user_name = extract_user_name(user_data["persona_info"])
    sessions = user_data["sessions"]

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, f"{user_data['uuid']}.json")

    new_user_data = {
        "uuid": user_data["uuid"],
        "user_name": user_name,
        "sessions": [],
    }

    for idx, session in enumerate(sessions):
        logger.info(f"Processing user {user_name}: session {idx}/{len(sessions)}")
        new_session = {
            "memory_points": session["memory_points"],
            "dialogue": session["dialogue"],
        }

        # Format dialogue
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

        # Process in batches
        result = []
        total_duration_ms = 0
        batch_size = 20

        for i in range(0, len(formatted_dialogue), batch_size):
            batch = formatted_dialogue[i : i + batch_size]
            batch_result, duration_ms = await add_memory_async(
                user_id=user_name,
                messages=batch,
            )
            if batch_result:
                result.extend(batch_result)
            total_duration_ms += duration_ms

        duration_ms = total_duration_ms

        # Extract memory content
        memories = []
        for memory_node in result:
            if isinstance(memory_node, MemoryNode) and memory_node.memory_type is not MemoryType.HISTORY:
                memories.append(memory_node.content)

        if session.get("is_generated_qa_session", False):
            new_session["add_dialogue_duration_ms"] = duration_ms
            new_session["is_generated_qa_session"] = True
            del new_session["dialogue"]
            del new_session["memory_points"]
            new_user_data["sessions"].append(new_session)
            continue

        # Store extracted memories
        new_session["extracted_memories"] = memories
        new_session["add_dialogue_duration_ms"] = duration_ms

        # Process questions
        if "questions" not in session:
            new_user_data["sessions"].append(new_session)
            continue

        new_session["questions"] = []

        for qa in session["questions"]:
            response, duration_ms = await search_memory_async(
                query=qa["question"],
                user_id=user_name,
                top_k=top_k_value,
            )

            new_qa = copy.deepcopy(qa)
            new_qa["system_response"] = response
            new_qa["search_duration_ms"] = duration_ms

            new_session["questions"].append(new_qa)

        # ==================== Evaluation for this session ====================
        session_eval_results = {
            "question_answering_records": [],
        }

        uuid = user_data["uuid"]

        # Evaluate Question Answering
        if "questions" in new_session:
            logger.info(f"Evaluating Question Answering for session {idx}...")
            
            # Format dialogue for evaluation
            # Format: Each turn contains role, content, and time_created
            dialogue_for_eval = []
            for turn in dialogue:
                dialogue_for_eval.append(
                    f"Role: {turn['role']}\n"
                    f"Content: {turn['content']}\n"
                    f"Time: {datetime.strptime(turn['timestamp'], '%b %d, %Y, %H:%M:%S').replace(tzinfo=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
                )
            formatted_dialogue_str = "\n\n".join(dialogue_for_eval)
            
            for qa in new_session["questions"]:
                new_qa = copy.deepcopy(qa)
                new_qa["uuid"] = uuid
                new_qa["session_id"] = idx

                result = await evaluation_for_question2(
                    qa["question"],
                    qa["answer"],
                    "\n".join([i["memory_content"] for i in qa["evidence"]]),
                    qa["system_response"],
                    formatted_dialogue_str,
                )
                result_type = result.get("evaluation_result")
                reasoning = result.get("reasoning", "")
                new_qa["result_type"] = result_type
                new_qa["question_answering_reasoning"] = reasoning
                session_eval_results["question_answering_records"].append(new_qa)

        # Store evaluation results in session
        new_session["evaluation_results"] = session_eval_results

        new_user_data["sessions"].append(new_session)

        # Save results
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(new_user_data, f, ensure_ascii=False, indent=2)
        session_size = len(new_user_data["sessions"])
        logger.info(f"✅ Saved user {user_name} to {tmp_file} session_size={session_size}")

    logger.info(f"✅ Saved user {user_name} to {tmp_file} all!")
    return {"uuid": user_data["uuid"], "status": "ok", "path": tmp_file}


# ==================== Evaluation Aggregation ====================


def aggregate_eval_results(eval_results):
    """Aggregate evaluation results and compute metrics (QA only)."""
    
    # Question-Answering Evaluation
    correct_qa_num = 0
    hallucination_qa_num = 0
    omission_qa_num = 0
    qa_num = 0
    qa_valid_num = 0

    for item in eval_results["question_answering_records"]:
        item["is_valid"] = True
        qa_num += 1

        if item["result_type"] not in ["Correct", "Hallucination", "Omission"]:
            item["is_valid"] = False
            continue

        if item["result_type"] == "Correct":
            correct_qa_num += 1
        elif item["result_type"] == "Hallucination":
            hallucination_qa_num += 1
        elif item["result_type"] == "Omission":
            omission_qa_num += 1

        qa_valid_num += 1

    if qa_num > 0:
        eval_results["overall_score"]["question_answering"]["correct_qa_ratio(all)"] = correct_qa_num / qa_num
        eval_results["overall_score"]["question_answering"]["hallucination_qa_ratio(all)"] = (
            hallucination_qa_num / qa_num
        )
        eval_results["overall_score"]["question_answering"]["omission_qa_ratio(all)"] = omission_qa_num / qa_num
    else:
        eval_results["overall_score"]["question_answering"]["correct_qa_ratio(all)"] = 0
        eval_results["overall_score"]["question_answering"]["hallucination_qa_ratio(all)"] = 0
        eval_results["overall_score"]["question_answering"]["omission_qa_ratio(all)"] = 0

    if qa_valid_num > 0:
        eval_results["overall_score"]["question_answering"]["correct_qa_ratio(valid)"] = correct_qa_num / qa_valid_num
        eval_results["overall_score"]["question_answering"]["hallucination_qa_ratio(valid)"] = (
            hallucination_qa_num / qa_valid_num
        )
        eval_results["overall_score"]["question_answering"]["omission_qa_ratio(valid)"] = omission_qa_num / qa_valid_num
    else:
        eval_results["overall_score"]["question_answering"]["correct_qa_ratio(valid)"] = 0
        eval_results["overall_score"]["question_answering"]["hallucination_qa_ratio(valid)"] = 0
        eval_results["overall_score"]["question_answering"]["omission_qa_ratio(valid)"] = 0

    eval_results["overall_score"]["question_answering"]["qa_valid_num"] = qa_valid_num
    eval_results["overall_score"]["question_answering"]["qa_num"] = qa_num

    return eval_results


# ==================== Main Pipeline ====================


async def main_async(
    data_path: str,
    top_k: int = 20,
    user_num: int = 1,
    max_concurrency: int = 2,
):
    """Main evaluation pipeline - simplified for QA only."""
    frame = "reme_simple"
    save_path = f"bench_results/{frame}/"
    os.makedirs(save_path, exist_ok=True)

    output_file_stage1 = os.path.join(save_path, f"{frame}_eval_results.jsonl")
    output_file_final = os.path.join(save_path, f"{frame}_eval_stat_result.json")

    start_time = time.time()
    await reme.vector_store.delete_all()

    # ==================== Stage 1: Data Processing ====================
    print("\n" + "=" * 80)
    print("PROCESSING DATA WITH ReMe (Simplified - QA Only)")
    print(f"Max Concurrency: {max_concurrency}")
    print("=" * 80)

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Load all user data
    user_data_list = list(iter_jsonl(data_path))
    total_users = min(len(user_data_list), user_num)
    user_data_list = user_data_list[:total_users]

    print(f"Processing {total_users} users with max concurrency {max_concurrency}...")
    
    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_single_user(idx: int, user_data: dict):
        """Process a single user with semaphore control."""
        async with semaphore:
            uuid = user_data['uuid']
            tmp_file = os.path.join(tmp_dir, f"{uuid}.json")
            
            if os.path.exists(tmp_file):
                print(f"⚡ Skipping user {uuid} ({idx}/{total_users}) — cached result found.")
                return {"uuid": uuid, "status": "cached", "path": tmp_file}
            
            print(f"[{idx}/{total_users}] Processing user {uuid}...")
            result = await process_user_stage1(user_data, top_k, save_path)
            print(f"[{idx}/{total_users}] ✅ Finished {uuid} ({result['status']})")
            return result

    # Process users in parallel with controlled concurrency
    tasks = [process_single_user(idx, user_data) for idx, user_data in enumerate(user_data_list, 1)]
    await asyncio.gather(*tasks)

    # Combine all results into final output
    with open(output_file_stage1, "w", encoding="utf-8") as f_out:
        for file in os.listdir(tmp_dir):
            if file.endswith(".json"):
                file_path = os.path.join(tmp_dir, file)
                with open(file_path, "r", encoding="utf-8") as f_in:
                    data = json.load(f_in)
                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    elapsed_stage1 = time.time() - start_time
    print(f"\n✅ Processing completed in {elapsed_stage1:.2f}s")
    print(f"✅ Results saved to: {output_file_stage1}")

    # ==================== Aggregate Results ====================
    print("\n" + "=" * 80)
    print("AGGREGATING EVALUATION RESULTS")
    print("=" * 80)

    # Calculate time consuming
    add_dialogue_duration_time = 0
    search_memory_duration_time = 0

    for user_data in iter_jsonl(output_file_stage1):
        sessions = user_data["sessions"]

        for session in sessions:
            if "add_dialogue_duration_ms" in session:
                add_dialogue_duration_time += session["add_dialogue_duration_ms"]

            if "questions" in session:
                for question in session["questions"]:
                    if "search_duration_ms" in question:
                        search_memory_duration_time += question["search_duration_ms"]

    add_dialogue_duration_time = add_dialogue_duration_time / 1000 / 60
    search_memory_duration_time = search_memory_duration_time / 1000 / 60

    print("\n🔄 Aggregating all user results...")

    eval_results = {
        "overall_score": {
            "question_answering": {},
            "time_consuming": {
                "add_dialogue_duration_time": add_dialogue_duration_time,
                "search_memory_duration_time": search_memory_duration_time,
                "total_duration_time": add_dialogue_duration_time + search_memory_duration_time,
            },
        },
        "question_answering_records": [],
    }

    # Extract QA records from all users
    for user_data in iter_jsonl(output_file_stage1):
        for session in user_data["sessions"]:
            if session.get("is_generated_qa_session", False):
                continue
            
            if "evaluation_results" in session:
                eval_results["question_answering_records"].extend(
                    session["evaluation_results"].get("question_answering_records", [])
                )

    eval_results = aggregate_eval_results(eval_results)

    with open(output_file_final, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=4)

    elapsed_total = time.time() - start_time
    print(f"\n✅ All done in {elapsed_total:.2f}s. Results saved to {output_file_final}")

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY (Question Answering Only)")
    print("=" * 80)

    print(f"\n📊 Question Answering:")
    print(
        f"  - Correct (all): {eval_results['overall_score']['question_answering'].get('correct_qa_ratio(all)', 0):.4f}",
    )
    print(
        f"  - Hallucination (all): {eval_results['overall_score']['question_answering'].get('hallucination_qa_ratio(all)', 0):.4f}",
    )
    print(
        f"  - Omission (all): {eval_results['overall_score']['question_answering'].get('omission_qa_ratio(all)', 0):.4f}",
    )
    print(
        f"  - Correct (valid): {eval_results['overall_score']['question_answering'].get('correct_qa_ratio(valid)', 0):.4f}",
    )
    print(
        f"  - Hallucination (valid): {eval_results['overall_score']['question_answering'].get('hallucination_qa_ratio(valid)', 0):.4f}",
    )
    print(
        f"  - Omission (valid): {eval_results['overall_score']['question_answering'].get('omission_qa_ratio(valid)', 0):.4f}",
    )
    print(
        f"  - Valid QA: {eval_results['overall_score']['question_answering'].get('qa_valid_num', 0)}/{eval_results['overall_score']['question_answering'].get('qa_num', 0)}",
    )

    print(f"\n⏱️  Time Consuming:")
    print(f"  - Add Dialogue: {add_dialogue_duration_time:.2f} min")
    print(f"  - Search Memory: {search_memory_duration_time:.2f} min")
    print(f"  - Total: {add_dialogue_duration_time + search_memory_duration_time:.2f} min")
    print("=" * 80)


def main(
    data_path: str,
    top_k: int = 20,
    user_num: int = 1,
    max_concurrency: int = 2,
):
    """Synchronous entry point."""
    asyncio.run(main_async(data_path, top_k, user_num, max_concurrency))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simplified evaluation for ReMe on HaluMem benchmark (QA only)")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to HaluMem data file (e.g., HaluMem-medium.jsonl)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of top memories to retrieve (default: 20)",
    )
    parser.add_argument(
        "--user_num",
        type=int,
        default=1,
        help="Number of users to evaluate (default: 1)",
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=2,
        help="Maximum concurrency for processing (default: 2)",
    )
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        top_k=args.top_k,
        user_num=args.user_num,
        max_concurrency=args.max_concurrency,
    )
