"""Evaluation tools for ReMe HaluMem benchmark."""

from pathlib import Path

import yaml

from llms import llm_request_for_json

# Load prompts from YAML file
_YAML_PATH = Path(__file__).parent / "halumem.yaml"
with open(_YAML_PATH, "r", encoding="utf-8") as f:
    _PROMPTS = yaml.safe_load(f)


async def evaluation_for_memory_integrity(
    extract_memories: str,
    target_memory: str,
):
    """
    Memory Integrity Evaluation
    extract_memories: A formatted string concatenating all memory points extracted by the memory system under evaluation.
    target_memory: The target key memory point.
    """

    prompt = _PROMPTS["EVALUATION_PROMPT_FOR_MEMORY_INTEGRITY"].format(
        memories=extract_memories,
        expected_memory_point=target_memory,
    )

    result = await llm_request_for_json(prompt)

    return result


async def evaluation_for_memory_accuracy(
    dialogue: str,
    golden_memories: str,
    candidate_memory: str,
):
    """
    Memory Accuracy Evaluation
    dialogue: The complete human-machine dialogue record.
    golden_memories: The core memory points for this dialogue segment in the evaluation set (the correct reference memories).
    candidate_memory: A specific memory point extracted by the memory system being evaluated.
    """

    prompt = _PROMPTS["EVALUATION_PROMPT_FOR_MEMORY_ACCURACY"].format(
        dialogue=dialogue,
        golden_memories=golden_memories,
        candidate_memory=candidate_memory,
    )

    result = await llm_request_for_json(prompt)

    return result


async def evaluation_for_update_memory(
    extract_memories: str,
    target_update_memory: str,
    original_memory: str,
):
    """
    Memory Update Evaluation
    extract_memories: A formatted string concatenating all memory points extracted by the memory system under evaluation.
    target_update_memory: The target updated memory point.
    original_memory: str: A formatted string concatenating all original memory points corresponding to the target updated memory point (i.e., all memories before the update).
    """

    prompt = _PROMPTS["EVALUATION_PROMPT_FOR_UPDATE_MEMORY"].format(
        memories=extract_memories,
        updated_memory=target_update_memory,
        original_memory=original_memory,
    )

    result = await llm_request_for_json(prompt)

    return result


async def evaluation_for_question(
    question: str,
    reference_answer: str,
    key_memory_points: str,
    response: str,
):
    """
    Question-Answering Evaluation
    question: The question string to be evaluated.
    reference_answer: The reference (gold-standard) answer.
    key_memory_points: The memory points used to derive the reference answer.
    response: The answer produced by the memory system.
    """

    prompt = _PROMPTS["EVALUATION_PROMPT_FOR_QUESTION"].format(
        question=question,
        reference_answer=reference_answer,
        key_memory_points=key_memory_points,
        response=response,
    )

    result = await llm_request_for_json(prompt)

    return result


async def evaluation_for_question2(
    question: str,
    reference_answer: str,
    key_memory_points: str,
    response: str,
    dialogue: str,
):
    """
    Question-Answering Evaluation with Dialogue Context (Version 2)
    question: The question string to be evaluated.
    reference_answer: The reference (gold-standard) answer.
    key_memory_points: The memory points used to derive the reference answer.
    response: The answer produced by the memory system.
    dialogue: The formatted dialogue history (role, content, time_created).
    """

    # prompt = _PROMPTS["EVALUATION_PROMPT_FOR_QUESTION2"].format(
    prompt = _PROMPTS["EVALUATION_PROMPT_FOR_QUESTION"].format(
        question=question,
        reference_answer=reference_answer,
        key_memory_points=key_memory_points,
        response=response,
        dialogue=dialogue,
    )

    result = await llm_request_for_json(prompt, model_name="qwen3-max")

    return result


async def answer_question_with_memories(
    question: str,
    memories: str,
    user_id: str = None,
):
    """
    Answer a question using retrieved memories with PROMPT_MEMZERO_JSON template.
    
    Args:
        question: The question to answer
        memories: The retrieved memories (formatted as context)
        user_id: Optional user ID for context formatting
    
    Returns:
        dict with 'reasoning' and 'answer' fields
    """
    # Format context with memories
    if user_id:
        context = _PROMPTS["TEMPLATE_MEMOS"].format(
            user_id=user_id,
            memories=memories
        )
    else:
        context = f"Memories:\n{memories}"
    
    # Use PROMPT_MEMZERO_JSON template for structured JSON response
    prompt = _PROMPTS["PROMPT_MEMZERO_JSON"].format(
        context=context,
        question=question
    )
    
    # result = await llm_request_for_json(prompt, model_name="qwen3-max")
    result = await llm_request_for_json(prompt, model_name="qwen3-30b-a3b-instruct-2507")

    return result
