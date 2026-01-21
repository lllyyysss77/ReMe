"""
Re-evaluate Question Answering results from data directory using LLM.

This script:
1. Loads existing QA records from data directory
2. Re-evaluates each system_response using multiple models in parallel
3. Uses both EVALUATION_PROMPT_FOR_QUESTION and EVALUATION_PROMPT_FOR_QUESTION2
4. Saves updated results with new evaluation metrics
"""

import asyncio
import json
import re
import yaml
from collections import defaultdict
from pathlib import Path
from typing import Any

from reme_ai.core_old.schema import Message
from reme_ai.core_old.utils import load_env
from reme_ai.reme import ReMe
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Load environment
load_env()

# Initialize ReMe singleton
reme = ReMe()

# Load prompts from YAML file
_YAML_PATH = Path(__file__).parent / "eval.yaml"
with open(_YAML_PATH, "r", encoding="utf-8") as f:
    _PROMPTS = yaml.safe_load(f)


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def llm_request(prompt: str, model_name: str = "qwen3-max", **kwargs) -> str:
    """Make an LLM request using ReMe's LLM."""
    assistant_message = await reme.llm.chat(
        messages=[
            Message(
                **{
                    "role": "user",
                    "content": prompt,
                },
            ),
        ],
        model_name=model_name,
        **kwargs,
    )
    return assistant_message.content


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
async def llm_request_for_json(prompt: str, model_name: str = "qwen3-max", **kwargs) -> dict:
    """Make an LLM request expecting JSON response."""
    content = await llm_request(prompt, model_name=model_name, **kwargs)

    match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON block found in model output: {content}")

    json_str = match.group(1).strip()
    return json.loads(json_str)


async def evaluate_qa_record(
    question: str,
    reference_answer: str,
    key_memory_points: str,
    response: str,
    dialogue: str = "",
    model_name: str = "qwen3-max",
    prompt_version: str = "v1"
) -> dict:
    """Evaluate a single QA record using LLM with specified prompt version.

    Args:
        question: The question to evaluate
        reference_answer: The reference answer
        key_memory_points: Key memory points
        response: System response to evaluate
        dialogue: Dialogue context (optional)
        model_name: LLM model name
        prompt_version: "v1" for EVALUATION_PROMPT_FOR_QUESTION,
                       "v2" for EVALUATION_PROMPT_FOR_QUESTION2

    Returns:
        dict with evaluation_result and reasoning
    """
    # Select prompt template
    if prompt_version == "v2":
        prompt_template = _PROMPTS["EVALUATION_PROMPT_FOR_QUESTION2"]
    else:
        prompt_template = _PROMPTS["EVALUATION_PROMPT_FOR_QUESTION"]

    # Format prompt
    prompt = prompt_template.format(
        question=question,
        reference_answer=reference_answer,
        key_memory_points=key_memory_points,
        response=response,
        dialogue=dialogue or "N/A"
    )

    result = await llm_request_for_json(prompt, model_name=model_name)
    return result


def load_from_data_dir(data_dir: str) -> list[dict]:
    """Load data from data directory (same as compute_qa_stats.py)."""
    data_path = Path(data_dir)

    # Try flat file structure first (conversation_{user}_session_{idx}.json)
    json_files = [f for f in data_path.iterdir() if f.is_file() and f.suffix == ".json"]

    if json_files:
        # Group files by user
        users_dict = defaultdict(list)

        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)
                user_name = session_data.get("user_name")
                if user_name:
                    users_dict[user_name].append({
                        "file": json_file,
                        "data": session_data
                    })

        # Sort sessions by session_idx for each user
        users_data = []
        for user_name, sessions in users_dict.items():
            sessions_sorted = sorted(
                sessions,
                key=lambda s: s["data"].get("session_idx", 0)
            )
            users_data.extend(sessions_sorted)

        return users_data

    return []


def format_dialogue_context(session_data: dict) -> str:
    """Format dialogue context from session data."""
    dialogue = session_data.get("session", {}).get("dialogue", [])
    if not dialogue:
        return "N/A"

    formatted_turns = []
    for turn in dialogue:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        timestamp = turn.get("timestamp", "")
        formatted_turns.append(
            f"Role: {role}\nContent: {content}\nTime: {timestamp}"
        )
    return "\n\n".join(formatted_turns)


async def reevaluate_session(
    session_file: Path,
    session_data: dict,
    models: list[str],
    prompt_versions: list[str],
    parallel: bool = True
) -> dict:
    """Re-evaluate all QA records in a session using multiple models and prompts.

    Args:
        session_file: Path to session file
        session_data: Session data dict
        models: List of model names to use for evaluation
        prompt_versions: List of prompt versions ("v1", "v2")
        parallel: If True, use asyncio.gather for parallel execution;
                 if False, execute sequentially

    Returns:
        Updated session data with evaluation results for each model+prompt combination

    Note:
        Request rate limiting is handled by base_llm.py's request_interval mechanism.
    """
    eval_results = session_data.get("session", {}).get("evaluation_results", {})
    qa_records = eval_results.get("question_answering_records", [])

    if not qa_records:
        print(f"  ⏭️  No QA records found")
        return session_data

    total_evals = len(models) * len(prompt_versions) * len(qa_records)
    print(f"  🔍 Re-evaluating {len(qa_records)} QA records with {len(models)} models × {len(prompt_versions)} prompts = {total_evals} evaluations...")

    # Format dialogue context once
    dialogue_context = format_dialogue_context(session_data)

    async def evaluate_single_combination(
        idx: int,
        qa: dict,
        model_name: str,
        prompt_version: str
    ) -> tuple[int, str, str, dict]:
        """Evaluate a single QA record with specific model and prompt.

        Note: Rate limiting is handled by BaseLLM's request_interval mechanism.
        """
        question = qa.get("question", "")
        reference_answer = qa.get("answer", "")

        # Get key memory points from evidence
        evidence = qa.get("evidence", [])
        key_memory_points = "\n".join([e.get("memory_content", "") for e in evidence])

        # Get system response
        system_response = qa.get("system_response", "")

        try:
            # Call LLM for evaluation
            eval_result = await evaluate_qa_record(
                question=question,
                reference_answer=reference_answer,
                key_memory_points=key_memory_points,
                response=system_response,
                dialogue=dialogue_context,
                model_name=model_name,
                prompt_version=prompt_version
            )

            result = {
                "result_type": eval_result.get("evaluation_result", "Invalid"),
                "reasoning": eval_result.get("reasoning", "")
            }

            return idx, model_name, prompt_version, result

        except Exception as e:
            print(f"    ❌ QA[{idx+1}] {model_name}/{prompt_version}: Error: {e}")
            return idx, model_name, prompt_version, {
                "result_type": "Error",
                "reasoning": f"Evaluation error: {str(e)}"
            }

    # Create all evaluation tasks (all combinations of models, prompts, and QA records)
    tasks = []
    for idx, qa in enumerate(qa_records):
        for model_name in models:
            for prompt_version in prompt_versions:
                tasks.append(evaluate_single_combination(idx, qa, model_name, prompt_version))

    # Execute evaluations based on parallel mode
    if parallel:
        print(f"  ⚡ Starting {len(tasks)} parallel evaluations (rate limited by LLM layer)...")
        results = await asyncio.gather(*tasks)
    else:
        print(f"  🔄 Starting {len(tasks)} sequential evaluations...")
        results = []
        for i, task in enumerate(tasks, 1):
            result = await task
            results.append(result)
            if i % 10 == 0 or i == len(tasks):
                print(f"    ⏳ Progress: {i}/{len(tasks)} evaluations completed")

    # Organize results by QA index, then by model and prompt
    # Structure: qa_records[idx]["evaluations"][model][prompt_version] = {result_type, reasoning}
    for idx, qa in enumerate(qa_records):
        if "evaluations" not in qa:
            qa["evaluations"] = {}

        # Initialize evaluations structure
        for model_name in models:
            if model_name not in qa["evaluations"]:
                qa["evaluations"][model_name] = {}

    # Fill in results
    completed_count = 0
    for qa_idx, model_name, prompt_version, result in results:
        qa_records[qa_idx]["evaluations"][model_name][prompt_version] = result
        completed_count += 1
        if completed_count % 10 == 0 or completed_count == len(results):
            print(f"    ✅ Completed {completed_count}/{len(results)} evaluations")

    # Set default result_type to first model's v1 result for compatibility
    if models and prompt_versions:
        default_model = models[0]
        default_prompt = prompt_versions[0]
        for qa in qa_records:
            default_eval = qa["evaluations"].get(default_model, {}).get(default_prompt, {})
            qa["result_type"] = default_eval.get("result_type", "Invalid")
            qa["question_answering_reasoning"] = default_eval.get("reasoning", "")

    # Update session data
    if "session" not in session_data:
        session_data["session"] = {}
    if "evaluation_results" not in session_data["session"]:
        session_data["session"]["evaluation_results"] = {}

    session_data["session"]["evaluation_results"]["question_answering_records"] = qa_records

    # Save updated session data
    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)

    print(f"  💾 Updated session saved with all evaluations")

    return session_data


def compute_qa_metrics(qa_records: list[dict]) -> dict[str, Any]:
    """Compute question answering metrics."""
    total = len(qa_records)
    if total == 0:
        return {
            "correct_qa_ratio(all)": 0,
            "hallucination_qa_ratio(all)": 0,
            "omission_qa_ratio(all)": 0,
            "correct_qa_ratio(valid)": 0,
            "hallucination_qa_ratio(valid)": 0,
            "omission_qa_ratio(valid)": 0,
            "qa_valid_num": 0,
            "qa_num": 0
        }

    correct = hallucination = omission = valid = 0

    for qa in qa_records:
        result_type = qa.get("result_type", "")
        if result_type == "Correct":
            correct += 1
            valid += 1
        elif result_type == "Hallucination":
            hallucination += 1
            valid += 1
        elif result_type == "Omission":
            omission += 1
            valid += 1

    metrics = {
        "correct_qa_ratio(all)": correct / total,
        "hallucination_qa_ratio(all)": hallucination / total,
        "omission_qa_ratio(all)": omission / total,
        "correct_qa_ratio(valid)": correct / valid if valid > 0 else 0,
        "hallucination_qa_ratio(valid)": hallucination / valid if valid > 0 else 0,
        "omission_qa_ratio(valid)": omission / valid if valid > 0 else 0,
        "qa_valid_num": valid,
        "qa_num": total
    }

    return metrics


async def main(
    data_dir: str = "./data",
    models: list[str] = None,
    prompt_versions: list[str] = None,
    parallel: bool = True
):
    """Main function to re-evaluate QA records from data directory with multiple models and prompts.

    Args:
        data_dir: Path to data directory
        models: List of model names (e.g., ["qwen3-max", "qwen-flash"])
        prompt_versions: List of prompt versions (e.g., ["v1", "v2"])
        parallel: If True, use parallel execution; if False, use sequential execution

    Note:
        Request rate limiting is automatically handled by base_llm.py's request_interval mechanism.
    """
    data_path = Path(data_dir)

    if not data_path.exists() or not data_path.is_dir():
        print(f"❌ Error: Directory not found: {data_dir}")
        return

    # Default values
    if models is None:
        models = ["qwen3-max"]
    if prompt_versions is None:
        prompt_versions = ["v1"]

    print("=" * 80)
    print("RE-EVALUATING QA RECORDS WITH MULTIPLE MODELS & PROMPTS")
    print(f"Models: {', '.join(models)}")
    print(f"Prompts: {', '.join(['EVALUATION_PROMPT_FOR_QUESTION' if v == 'v1' else 'EVALUATION_PROMPT_FOR_QUESTION2' for v in prompt_versions])}")
    print(f"Execution mode: {'Parallel' if parallel else 'Sequential'}")
    print("Note: Request rate limiting handled by LLM layer (base_llm.py)")
    print("=" * 80 + "\n")

    # Load data from directory
    sessions = load_from_data_dir(data_dir)

    if not sessions:
        print(f"❌ No session files found in {data_dir}")
        return

    print(f"📂 Found {len(sessions)} session files\n")

    # Process each session
    all_qa_records = []

    for idx, session_info in enumerate(sessions, 1):
        session_file = session_info["file"]
        session_data = session_info["data"]
        user_name = session_data.get("user_name", "Unknown")
        session_idx = session_data.get("session_idx", 0)

        print(f"[{idx}/{len(sessions)}] {user_name} - Session {session_idx}")

        updated_session = await reevaluate_session(
            session_file=session_file,
            session_data=session_data,
            models=models,
            prompt_versions=prompt_versions,
            parallel=parallel
        )

        # Collect QA records for metrics
        eval_results = updated_session.get("session", {}).get("evaluation_results", {})
        qa_records = eval_results.get("question_answering_records", [])
        all_qa_records.extend(qa_records)

        print()

    # Compute and display metrics for each model+prompt combination
    print("=" * 80)
    print("UPDATED METRICS (BY MODEL & PROMPT)")
    print("=" * 80 + "\n")

    for model_name in models:
        for prompt_version in prompt_versions:
            prompt_name = "EVALUATION_PROMPT_FOR_QUESTION" if prompt_version == "v1" else "EVALUATION_PROMPT_FOR_QUESTION2"
            print(f"\n📊 {model_name} / {prompt_name}:")
            print("─" * 80)

            # Extract QA records for this model+prompt combination
            model_qa_records = []
            for qa in all_qa_records:
                eval_data = qa.get("evaluations", {}).get(model_name, {}).get(prompt_version, {})
                if eval_data:
                    # Create a copy with the specific evaluation result
                    qa_copy = {
                        **qa,
                        "result_type": eval_data.get("result_type", "Invalid"),
                        "question_answering_reasoning": eval_data.get("reasoning", "")
                    }
                    model_qa_records.append(qa_copy)

            if model_qa_records:
                metrics = compute_qa_metrics(model_qa_records)

                print(f"  Correct (all):         {metrics['correct_qa_ratio(all)']:.4f}")
                print(f"  Hallucination (all):   {metrics['hallucination_qa_ratio(all)']:.4f}")
                print(f"  Omission (all):        {metrics['omission_qa_ratio(all)']:.4f}")
                print(f"  Correct (valid):       {metrics['correct_qa_ratio(valid)']:.4f}")
                print(f"  Hallucination (valid): {metrics['hallucination_qa_ratio(valid)']:.4f}")
                print(f"  Omission (valid):      {metrics['omission_qa_ratio(valid)']:.4f}")
                print(f"  Valid/Total:           {metrics['qa_valid_num']}/{metrics['qa_num']}")

    # Save detailed results with all evaluations
    report_file = data_path.parent / "reme_eval_stat_result_detailed.json"

    # Create summary for each model+prompt combination
    evaluation_summary = {}
    for model_name in models:
        evaluation_summary[model_name] = {}
        for prompt_version in prompt_versions:
            prompt_name = "EVALUATION_PROMPT_FOR_QUESTION" if prompt_version == "v1" else "EVALUATION_PROMPT_FOR_QUESTION2"

            # Extract QA records for this combination
            model_qa_records = []
            for qa in all_qa_records:
                eval_data = qa.get("evaluations", {}).get(model_name, {}).get(prompt_version, {})
                if eval_data:
                    qa_copy = {
                        **qa,
                        "result_type": eval_data.get("result_type", "Invalid"),
                        "question_answering_reasoning": eval_data.get("reasoning", "")
                    }
                    model_qa_records.append(qa_copy)

            metrics = compute_qa_metrics(model_qa_records)
            evaluation_summary[model_name][prompt_name] = {
                "metrics": metrics,
                "qa_records": model_qa_records
            }

    final_results = {
        "evaluation_summary": evaluation_summary,
        "all_qa_records_with_evaluations": all_qa_records
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Detailed results saved: {report_file}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-evaluate QA records from data directory using multiple models and prompts in parallel. "
                    "Request rate limiting is automatically handled by base_llm.py's request_interval mechanism."
    )
    parser.add_argument(
        "data_dir",
        nargs='?',
        default="./data",
        type=str,
        help="Path to data directory containing user session files (default: ./data)"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs='+',
        default=["gpt-5.1-2025-11-13", "gemini-3-pro-preview"],
        help="LLM model names for evaluation (space-separated, default: qwen3-max)"
    )
    # ["gpt-4o-2024-11-20"], ["qwen3-max", "qwen-flash", "qwen3-30b-a3b-instruct-2507", "qwen3-30b-a3b-instruct-2507"]
    parser.add_argument(
        "--prompts",
        type=str,
        nargs='+',
        choices=["v1", "v2"],
        default=["v1", "v2"],
        help="Prompt versions to use: v1=EVALUATION_PROMPT_FOR_QUESTION, v2=EVALUATION_PROMPT_FOR_QUESTION2 (default: v1)"
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Use sequential execution instead of parallel (default: parallel)"
    )

    args = parser.parse_args()

    asyncio.run(main(
        data_dir=args.data_dir,
        models=args.models,
        prompt_versions=args.prompts,
        parallel=not args.serial
    ))
