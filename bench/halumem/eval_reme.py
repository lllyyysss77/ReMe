"""
Complete evaluation script for ReMe on HaluMem benchmark.

This script performs the full evaluation pipeline:
1. Load HaluMem data
2. Process each user's sessions with ReMe (summary + retrieve) - Stage 1 (Parallel)
3. Evaluate memory integrity, accuracy, updates, and question answering - Stage 2 (Sequential)
4. Generate metrics and statistics

Usage:
    python bench/halumem/eval_reme.py --data_path /Users/yuli/workspace/HaluMem/data/HaluMem-Long.jsonl \
        --top_k 20 --user_num 100 --max_concurrency 20
    python bench/halumem/eval_reme.py --data_path ./HaluMem-Long.jsonl \
        --top_k 20 --user_num 100 --max_concurrency 20

    python bench/halumem/eval_reme.py --data_path /Users/yuli/workspace/HaluMem/data/tmp_14.jsonl \
        --top_k 20 --user_num 1 --max_concurrency 1
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
    evaluation_for_memory_accuracy,
    evaluation_for_memory_integrity,
    evaluation_for_question,
    evaluation_for_update_memory,
)
from llms import llm_request
from reme_ai.core_old.enumeration import MemoryType
from reme_ai.core_old.schema import MemoryNode
from reme_ai.reme import ReMe

# Template for formatting memories (from shared YAML config)
TEMPLATE_MEMOS = _PROMPTS["TEMPLATE_MEMOS"]

# Prompt for question answering (using optimized PROMPT_MEMOS)
PROMPT_MEMOS = _PROMPTS["PROMPT_MEMOS"]

# Initialize ReMe with rate limiting configuration
# The default LLM can be overridden at call time using model_name parameter
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


def compute_f1(precision: float, recall: float) -> float:
    """Compute F1-score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


# ==================== Stage 1: Data Processing ====================


async def add_memory_async(user_id: str, messages: list[dict]) -> tuple[list[MemoryNode], float]:
    """Add memory to ReMe system asynchronously."""
    start = time.time()
    result = await reme.summary_v2(messages=messages, user_id=user_id)
    duration_ms = (time.time() - start) * 1000
    return result, duration_ms


async def search_memory_async(query: str, user_id: str, top_k: int = 20):
    """Search memory from ReMe system asynchronously."""
    start = time.time()
    memories = await reme.retrieve_v2(query=query, user_id=user_id, top_k=top_k)

    # Format the context
    context = TEMPLATE_MEMOS.format(user_id=user_id, memories=memories)
    duration_ms = (time.time() - start) * 1000
    return context, memories, duration_ms


async def process_user_stage1(
    user_data: dict,
    top_k_value: int,
    save_path: str,
):
    """Stage 1: Process user data through ReMe (summary + retrieve)."""
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

        # Search updated memories for memory points
        for memory in new_session["memory_points"]:
            if memory["is_update"] == "False" or not memory.get("original_memories"):
                continue

            _, memories_from_system, duration_ms = await search_memory_async(
                query=memory["memory_content"],
                user_id=user_name,
                top_k=10,
            )

            memory["memories_from_system"] = memories_from_system

        # Process questions
        if "questions" not in session:
            new_user_data["sessions"].append(new_session)
            continue

        new_session["questions"] = []

        for qa in session["questions"]:
            context, _, duration_ms = await search_memory_async(
                query=qa["question"],
                user_id=user_name,
                top_k=top_k_value,
            )

            new_qa = copy.deepcopy(qa)
            new_qa["context"] = context
            new_qa["search_duration_ms"] = duration_ms

            prompt = PROMPT_MEMOS.format(
                context=context,
                question=qa["question"],
            )

            start_time = time.time()
            response = await llm_request(prompt)
            new_qa["system_response"] = response
            new_qa["response_duration_ms"] = (time.time() - start_time) * 1000

            new_session["questions"].append(new_qa)

        # ==================== Evaluation for this session ====================
        session_eval_results = {
            "memory_integrity_records": [],
            "memory_accuracy_records": [],
            "memory_update_records": [],
            "question_answering_records": [],
        }

        uuid = user_data["uuid"]
        golden_memories = session["memory_points"]
        extract_memories = new_session["extracted_memories"]
        extract_memories_str = "\n".join(extract_memories)

        # Evaluate Memory Integrity
        logger.info(f"Evaluating Memory Integrity for session {idx}...")
        for memory in golden_memories:
            if memory["is_update"] == "True" and memory.get("memories_from_system", []):
                # Skip update memories for integrity check
                continue

            new_memory = copy.deepcopy(memory)
            new_memory["uuid"] = uuid
            new_memory["session_id"] = idx

            if extract_memories_str.strip() == "":
                new_memory["memory_integrity_score"] = 0
                new_memory["memory_integrity_reasoning"] = "No memories extracted"
                session_eval_results["memory_integrity_records"].append(new_memory)
                continue

            result = await evaluation_for_memory_integrity(extract_memories_str, memory["memory_content"])
            score = int(result.get("score"))
            reasoning = result.get("reasoning", "")
            new_memory["memory_integrity_score"] = score
            new_memory["memory_integrity_reasoning"] = reasoning
            session_eval_results["memory_integrity_records"].append(new_memory)

        # Evaluate Memory Accuracy
        logger.info(f"Evaluating Memory Accuracy for session {idx}...")
        dialogue = session["dialogue"]
        dialogue_str = []
        for turn in dialogue:
            dialogue_str.append(f'[{turn["timestamp"]}]{turn["role"]}: {turn["content"]}')
            if turn["role"] == "assistant":
                dialogue_str.append("")
        dialogue_str = "\n".join(dialogue_str)

        golden_memories_str = "\n".join(
            [m["memory_content"] for m in golden_memories if m["memory_source"] != "interference"],
        )

        for memory in extract_memories:
            new_memory = {
                "uuid": uuid,
                "session_id": idx,
                "memory_content": memory,
            }
            result = await evaluation_for_memory_accuracy(dialogue_str, golden_memories_str, memory)
            score = int(result.get("accuracy_score"))
            is_included_in_golden_memories = result.get("is_included_in_golden_memories", "false")
            reason = result.get("reason", "")
            new_memory["memory_accuracy_score"] = score
            new_memory["is_included_in_golden_memories"] = is_included_in_golden_memories
            new_memory["memory_accuracy_reason"] = reason
            session_eval_results["memory_accuracy_records"].append(new_memory)

        # Evaluate Memory Update
        logger.info(f"Evaluating Memory Update for session {idx}...")
        for memory in golden_memories:
            if memory["is_update"] == "False" or not memory.get("original_memories"):
                continue

            if not memory.get("memories_from_system", []):
                continue

            update_memory = copy.deepcopy(memory)
            update_memory["uuid"] = uuid
            update_memory["session_id"] = idx

            result = await evaluation_for_update_memory(
                "\n".join(update_memory["memories_from_system"]),
                update_memory["memory_content"],
                "\n".join(update_memory["original_memories"]),
            )
            update_type = result.get("evaluation_result")
            reason = result.get("reason", "")
            update_memory["memory_update_type"] = update_type
            update_memory["memory_update_reason"] = reason
            session_eval_results["memory_update_records"].append(update_memory)

        # Evaluate Question Answering
        if "questions" in new_session:
            logger.info(f"Evaluating Question Answering for session {idx}...")
            for qa in new_session["questions"]:
                new_qa = copy.deepcopy(qa)
                new_qa["uuid"] = uuid
                new_qa["session_id"] = idx

                result = await evaluation_for_question(
                    qa["question"],
                    qa["answer"],
                    "\n".join([i["memory_content"] for i in qa["evidence"]]),
                    qa["system_response"],
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


# ==================== Stage 2: Evaluation ====================


async def process_user_stage2(idx: int, user_data: dict):
    """Stage 2: Extract evaluation results from user's sessions (already computed in Stage 1)."""
    user_name = user_data["user_name"]

    eval_results = {
        "memory_integrity_records": [],
        "memory_accuracy_records": [],
        "memory_update_records": [],
        "question_answering_records": [],
    }

    logger.info(f"[{idx}]{user_name}: Extracting evaluation results from sessions...")

    # Extract evaluation results from each session
    for session in user_data["sessions"]:
        if session.get("is_generated_qa_session", False):
            continue

        if "evaluation_results" not in session:
            logger.warning(f"[{idx}]{user_name}: Session missing evaluation_results, skipping...")
            continue

        session_eval = session["evaluation_results"]
        eval_results["memory_integrity_records"].extend(session_eval.get("memory_integrity_records", []))
        eval_results["memory_accuracy_records"].extend(session_eval.get("memory_accuracy_records", []))
        eval_results["memory_update_records"].extend(session_eval.get("memory_update_records", []))
        eval_results["question_answering_records"].extend(session_eval.get("question_answering_records", []))

    logger.info(
        f"[{idx}]{user_name}: Extracted {len(eval_results['memory_integrity_records'])} integrity, "
        f"{len(eval_results['memory_accuracy_records'])} accuracy, "
        f"{len(eval_results['memory_update_records'])} update, "
        f"{len(eval_results['question_answering_records'])} QA records",
    )

    return eval_results


def aggregate_eval_results(eval_results):
    """Aggregate evaluation results and compute metrics."""

    # Memory Integrity Evaluation
    memory_integrity_scores = 0
    memory_integrity_weighted_scores = 0
    memory_integrity_valid_num = 0
    memory_integrity_num = 0
    memory_integrity_weighted_valid_num = 0
    memory_integrity_weighted_num = 0
    interference_memory_scores = 0
    interference_memory_valid_num = 0
    interference_memory_num = 0

    for item in eval_results["memory_integrity_records"]:
        item["is_valid"] = True

        if item["memory_source"] != "interference":
            memory_integrity_num += 1
            memory_integrity_weighted_num += item["importance"]
        else:
            interference_memory_num += 1

        if item["memory_integrity_score"] is None:
            item["is_valid"] = False
            continue

        if item["memory_source"] != "interference":
            if item["memory_integrity_score"] == 2:
                memory_integrity_scores += 1
            memory_integrity_weighted_scores += 0.5 * item["memory_integrity_score"] * item["importance"]
            memory_integrity_valid_num += 1
            memory_integrity_weighted_valid_num += item["importance"]
        else:
            if item["memory_integrity_score"] == 0:
                interference_memory_scores += 1
            interference_memory_valid_num += 1

    eval_results["overall_score"]["memory_integrity"]["recall(all)"] = (
        memory_integrity_scores / memory_integrity_num if memory_integrity_num > 0 else 0
    )
    eval_results["overall_score"]["memory_integrity"]["recall(valid)"] = (
        memory_integrity_scores / memory_integrity_valid_num if memory_integrity_valid_num > 0 else 0
    )
    eval_results["overall_score"]["memory_integrity"]["weighted_recall(all)"] = (
        memory_integrity_weighted_scores / memory_integrity_weighted_num if memory_integrity_weighted_num > 0 else 0
    )
    eval_results["overall_score"]["memory_integrity"]["weighted_recall(valid)"] = (
        memory_integrity_weighted_scores / memory_integrity_weighted_valid_num
        if memory_integrity_weighted_valid_num > 0
        else 0
    )
    eval_results["overall_score"]["memory_integrity"][
        "memory_valid_importance_sum"
    ] = memory_integrity_weighted_valid_num
    eval_results["overall_score"]["memory_integrity"]["memory_importance_sum"] = memory_integrity_weighted_num
    eval_results["overall_score"]["memory_integrity"]["memory_valid_num"] = memory_integrity_valid_num
    eval_results["overall_score"]["memory_integrity"]["memory_num"] = memory_integrity_num
    eval_results["overall_score"]["memory_accuracy"]["interference_accuracy(all)"] = (
        interference_memory_scores / interference_memory_num if interference_memory_num > 0 else 0
    )
    eval_results["overall_score"]["memory_accuracy"]["interference_accuracy(valid)"] = (
        interference_memory_scores / interference_memory_valid_num if interference_memory_valid_num > 0 else 0
    )
    eval_results["overall_score"]["memory_accuracy"]["interference_memory_valid_num"] = interference_memory_valid_num
    eval_results["overall_score"]["memory_accuracy"]["interference_memory_num"] = interference_memory_num

    # Memory Accuracy Evaluation
    target_memory_accuracy_scores = 0
    memory_accuracy_weighted_scores = 0
    target_memory_accuracy_valid_num = 0
    target_memory_accuracy_num = 0
    memory_accuracy_valid_num = 0
    memory_accuracy_num = 0

    for item in eval_results["memory_accuracy_records"]:
        item["is_valid"] = True
        memory_accuracy_num += 1

        if item["is_included_in_golden_memories"] in ["true", "True"]:
            target_memory_accuracy_num += 1

        if item["memory_accuracy_score"] is None:
            item["is_valid"] = False
            continue

        if item["is_included_in_golden_memories"] in ["true", "True"]:
            target_memory_accuracy_scores += 0.5 * item["memory_accuracy_score"]
            target_memory_accuracy_valid_num += 1

        memory_accuracy_weighted_scores += 0.5 * item["memory_accuracy_score"]
        memory_accuracy_valid_num += 1

    eval_results["overall_score"]["memory_accuracy"]["target_accuracy(all)"] = (
        target_memory_accuracy_scores / target_memory_accuracy_num if target_memory_accuracy_num > 0 else 0
    )
    eval_results["overall_score"]["memory_accuracy"]["target_accuracy(valid)"] = (
        target_memory_accuracy_scores / target_memory_accuracy_valid_num if target_memory_accuracy_valid_num > 0 else 0
    )
    eval_results["overall_score"]["memory_accuracy"]["target_memory_valid_num"] = target_memory_accuracy_valid_num
    eval_results["overall_score"]["memory_accuracy"]["target_memory_num"] = target_memory_accuracy_num
    eval_results["overall_score"]["memory_accuracy"]["weighted_accuracy(all)"] = (
        memory_accuracy_weighted_scores / memory_accuracy_num if memory_accuracy_num > 0 else 0
    )
    eval_results["overall_score"]["memory_accuracy"]["weighted_accuracy(valid)"] = (
        memory_accuracy_weighted_scores / memory_accuracy_valid_num if memory_accuracy_valid_num > 0 else 0
    )
    eval_results["overall_score"]["memory_accuracy"]["memory_valid_num"] = memory_accuracy_valid_num
    eval_results["overall_score"]["memory_accuracy"]["memory_num"] = memory_accuracy_num

    # Memory Extraction F1-score
    eval_results["overall_score"]["memory_extraction_f1"] = compute_f1(
        precision=eval_results["overall_score"]["memory_accuracy"]["target_accuracy(all)"],
        recall=eval_results["overall_score"]["memory_integrity"]["recall(all)"],
    )

    # Memory Update Evaluation
    correct_update_memory_num = 0
    hallucination_update_memory_num = 0
    omission_update_memory_num = 0
    other_update_memory_num = 0
    update_memory_num = 0
    update_memory_valid_num = 0

    for item in eval_results["memory_update_records"]:
        item["is_valid"] = True
        update_memory_num += 1

        if item["memory_update_type"] not in ["Correct", "Hallucination", "Omission", "Other"]:
            item["is_valid"] = False
            continue

        if item["memory_update_type"] == "Correct":
            correct_update_memory_num += 1
        elif item["memory_update_type"] == "Hallucination":
            hallucination_update_memory_num += 1
        elif item["memory_update_type"] == "Omission":
            omission_update_memory_num += 1
        elif item["memory_update_type"] == "Other":
            other_update_memory_num += 1

        update_memory_valid_num += 1

    if update_memory_num > 0:
        eval_results["overall_score"]["memory_update"]["correct_update_memory_ratio(all)"] = (
            correct_update_memory_num / update_memory_num
        )
        eval_results["overall_score"]["memory_update"]["hallucination_update_memory_ratio(all)"] = (
            hallucination_update_memory_num / update_memory_num
        )
        eval_results["overall_score"]["memory_update"]["omission_update_memory_ratio(all)"] = (
            omission_update_memory_num / update_memory_num
        )
        eval_results["overall_score"]["memory_update"]["other_update_memory_ratio(all)"] = (
            other_update_memory_num / update_memory_num
        )
    else:
        eval_results["overall_score"]["memory_update"]["correct_update_memory_ratio(all)"] = 0
        eval_results["overall_score"]["memory_update"]["hallucination_update_memory_ratio(all)"] = 0
        eval_results["overall_score"]["memory_update"]["omission_update_memory_ratio(all)"] = 0
        eval_results["overall_score"]["memory_update"]["other_update_memory_ratio(all)"] = 0

    if update_memory_valid_num > 0:
        eval_results["overall_score"]["memory_update"]["correct_update_memory_ratio(valid)"] = (
            correct_update_memory_num / update_memory_valid_num
        )
        eval_results["overall_score"]["memory_update"]["hallucination_update_memory_ratio(valid)"] = (
            hallucination_update_memory_num / update_memory_valid_num
        )
        eval_results["overall_score"]["memory_update"]["omission_update_memory_ratio(valid)"] = (
            omission_update_memory_num / update_memory_valid_num
        )
        eval_results["overall_score"]["memory_update"]["other_update_memory_ratio(valid)"] = (
            other_update_memory_num / update_memory_valid_num
        )
    else:
        eval_results["overall_score"]["memory_update"]["correct_update_memory_ratio(valid)"] = 0
        eval_results["overall_score"]["memory_update"]["hallucination_update_memory_ratio(valid)"] = 0
        eval_results["overall_score"]["memory_update"]["omission_update_memory_ratio(valid)"] = 0
        eval_results["overall_score"]["memory_update"]["other_update_memory_ratio(valid)"] = 0

    eval_results["overall_score"]["memory_update"]["update_memory_valid_num"] = update_memory_valid_num
    eval_results["overall_score"]["memory_update"]["update_memory_num"] = update_memory_num

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

    # Memory Type Accuracy
    for item in eval_results["memory_integrity_records"]:
        if "memory_integrity_score" not in item or "importance" not in item:
            continue
        score = 1 if item["memory_integrity_score"] == 2 else 0
        eval_results["overall_score"]["memory_type_accuracy"][item["memory_type"]]["memory_integrity_acc"] += score
        eval_results["overall_score"]["memory_type_accuracy"][item["memory_type"]]["total_num"] += 1

    for item in eval_results["memory_update_records"]:
        if "memory_update_type" not in item or "importance" not in item:
            continue
        score = 1 if item["memory_update_type"] == "Correct" else 0
        eval_results["overall_score"]["memory_type_accuracy"][item["memory_type"]]["memory_update_acc"] += score
        eval_results["overall_score"]["memory_type_accuracy"][item["memory_type"]]["total_num"] += 1

    for key in eval_results["overall_score"]["memory_type_accuracy"]:
        if eval_results["overall_score"]["memory_type_accuracy"][key]["total_num"] > 0:
            total = eval_results["overall_score"]["memory_type_accuracy"][key]["total_num"]
            eval_results["overall_score"]["memory_type_accuracy"][key]["memory_integrity_acc"] = (
                eval_results["overall_score"]["memory_type_accuracy"][key]["memory_integrity_acc"] / total
            )
            eval_results["overall_score"]["memory_type_accuracy"][key]["memory_update_acc"] = (
                eval_results["overall_score"]["memory_type_accuracy"][key]["memory_update_acc"] / total
            )
            eval_results["overall_score"]["memory_type_accuracy"][key]["memory_acc"] = (
                eval_results["overall_score"]["memory_type_accuracy"][key]["memory_integrity_acc"]
                + eval_results["overall_score"]["memory_type_accuracy"][key]["memory_update_acc"]
            )
        else:
            eval_results["overall_score"]["memory_type_accuracy"][key]["memory_integrity_acc"] = 0
            eval_results["overall_score"]["memory_type_accuracy"][key]["memory_update_acc"] = 0
            eval_results["overall_score"]["memory_type_accuracy"][key]["memory_acc"] = 0

    return eval_results


# ==================== Main Pipeline ====================


async def main_async(
    data_path: str,
    top_k: int = 20,
    user_num: int = 1,
    max_concurrency: int = 2,
):
    """Main evaluation pipeline."""
    frame = "reme"
    save_path = f"bench_results/{frame}/"
    os.makedirs(save_path, exist_ok=True)

    output_file_stage1 = os.path.join(save_path, f"{frame}_eval_results.jsonl")
    output_file_stage2 = os.path.join(save_path, f"{frame}_eval_stat_result.json")

    start_time = time.time()
    await reme.vector_store.delete_all()

    # ==================== Stage 1: Data Processing ====================
    print("\n" + "=" * 80)
    print("STAGE 1: PROCESSING DATA WITH ReMe")
    print(f"Max Concurrency: {max_concurrency}")
    print("=" * 80)

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Load all user data
    user_data_list = list(iter_jsonl(data_path))
    total_users = min(len(user_data_list), user_num)
    user_data_list = user_data_list[:total_users]

    print(f"Processing {total_users} users with max concurrency {max_concurrency}...")

    # Create semaphore to limit concurrency for Stage 1
    semaphore_stage1 = asyncio.Semaphore(max_concurrency)

    async def process_single_user_stage1(idx: int, user_data: dict):
        """Process a single user in Stage 1 with semaphore control."""
        async with semaphore_stage1:
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
    tasks = [process_single_user_stage1(idx, user_data) for idx, user_data in enumerate(user_data_list, 1)]
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
    print(f"\n✅ Stage 1 completed in {elapsed_stage1:.2f}s")
    print(f"✅ Results saved to: {output_file_stage1}")

    # ==================== Stage 2: Evaluation ====================
    print("\n" + "=" * 80)
    print("STAGE 2: EVALUATING MEMORY PERFORMANCE (Sequential)")
    print("=" * 80)

    tmp_dir2 = os.path.join(save_path, "tmp2")
    os.makedirs(tmp_dir2, exist_ok=True)

    start_stage2 = time.time()

    # Load all users and process sequentially
    user_data_list = list(enumerate(iter_jsonl(output_file_stage1), 1))

    for idx, user_data in user_data_list:
        uuid = user_data["uuid"]
        tmp_file = os.path.join(tmp_dir2, f"{uuid}.json")

        if os.path.exists(tmp_file):
            print(f"⚡ Skipping user {uuid} ({idx}/{len(user_data_list)}) — cached result found.")
            continue

        print(f"[{idx}/{len(user_data_list)}] Processing user {uuid}...")
        t_user_result = await process_user_stage2(idx, user_data)

        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(t_user_result, f, ensure_ascii=False, indent=4)

        elapsed = time.time() - start_stage2
        print(f"[{idx}/{len(user_data_list)}] ✅ Finished user {uuid}, elapsed {elapsed:.2f}s.")

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
            "memory_integrity": {},
            "memory_accuracy": {},
            "memory_extraction_f1": 0,
            "memory_update": {},
            "question_answering": {},
            "memory_type_accuracy": {
                "Event Memory": {
                    "memory_integrity_acc": 0,
                    "memory_update_acc": 0,
                    "total_num": 0,
                },
                "Persona Memory": {
                    "memory_integrity_acc": 0,
                    "memory_update_acc": 0,
                    "total_num": 0,
                },
                "Relationship Memory": {
                    "memory_integrity_acc": 0,
                    "memory_update_acc": 0,
                    "total_num": 0,
                },
            },
            "time_consuming": {
                "add_dialogue_duration_time": add_dialogue_duration_time,
                "search_memory_duration_time": search_memory_duration_time,
                "total_duration_time": add_dialogue_duration_time + search_memory_duration_time,
            },
        },
        "memory_integrity_records": [],
        "memory_accuracy_records": [],
        "memory_update_records": [],
        "question_answering_records": [],
    }

    for file_name in os.listdir(tmp_dir2):
        if not file_name.endswith(".json"):
            continue
        user_file = os.path.join(tmp_dir2, file_name)
        with open(user_file, "r", encoding="utf-8") as f:
            user_result = json.load(f)

        eval_results["memory_accuracy_records"].extend(user_result.get("memory_accuracy_records", []))
        eval_results["memory_integrity_records"].extend(user_result.get("memory_integrity_records", []))
        eval_results["memory_update_records"].extend(user_result.get("memory_update_records", []))
        eval_results["question_answering_records"].extend(user_result.get("question_answering_records", []))

    eval_results = aggregate_eval_results(eval_results)

    with open(output_file_stage2, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=4)

    elapsed_total = time.time() - start_time
    print(f"\n✅ All done in {elapsed_total:.2f}s. Results saved to {output_file_stage2}")

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print("\n📊 Memory Integrity:")
    print(f"  - Recall (all): {eval_results['overall_score']['memory_integrity'].get('recall(all)', 0):.4f}")
    print(f"  - Recall (valid): {eval_results['overall_score']['memory_integrity'].get('recall(valid)', 0):.4f}")
    print(f"  - Weighted Recall (all): "
          f"{eval_results['overall_score']['memory_integrity'].get('weighted_recall(all)', 0):.4f}")

    print(f"\n📊 Memory Accuracy:")
    print(f"  - Target Accuracy (all): {eval_results['overall_score']['memory_accuracy'].get('target_accuracy(all)', 0):.4f}")
    print(f"  - Target Accuracy (valid): {eval_results['overall_score']['memory_accuracy'].get('target_accuracy(valid)', 0):.4f}",
    )
    print(
        f"  - Weighted Accuracy (all): {eval_results['overall_score']['memory_accuracy'].get('weighted_accuracy(all)', 0):.4f}",
    )

    print(f"\n📊 Memory Extraction F1: {eval_results['overall_score']['memory_extraction_f1']:.4f}")

    print(f"\n📊 Memory Update:")
    print(
        f"  - Correct (all): {eval_results['overall_score']['memory_update'].get('correct_update_memory_ratio(all)', 0):.4f}",
    )
    print(
        f"  - Hallucination (all): {eval_results['overall_score']['memory_update'].get('hallucination_update_memory_ratio(all)', 0):.4f}",
    )
    print(
        f"  - Omission (all): {eval_results['overall_score']['memory_update'].get('omission_update_memory_ratio(all)', 0):.4f}",
    )

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

    parser = argparse.ArgumentParser(description="Complete evaluation for ReMe on HaluMem benchmark")
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
        help="Maximum concurrency for stage 1 processing (default: 2)",
    )
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        top_k=args.top_k,
        user_num=args.user_num,
        max_concurrency=args.max_concurrency,
    )
