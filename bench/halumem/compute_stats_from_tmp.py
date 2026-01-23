"""
Compute statistics from existing tmp JSON files (Stage 1 results).

This script assumes that process_user_stage1 has already been run and JSON files
are available in the tmp directory. It will:
1. Load all JSON files from tmp directory
2. Generate the combined JSONL file
3. Run Stage 2 evaluation (extraction only, no new API calls)
4. Aggregate results and compute metrics

Usage:
    python bench/halumem/compute_stats_from_tmp.py --tmp_dir bench_results/reme_simple_v4/tmp
"""

import asyncio
import json
import os
import time
from datetime import datetime

from loguru import logger

from eval_tools import (
    evaluation_for_memory_accuracy,
    evaluation_for_memory_integrity,
    evaluation_for_question,
    evaluation_for_update_memory,
)


def compute_f1(precision: float, recall: float) -> float:
    """Compute F1-score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def iter_jsonl(file_path: str):
    """Iterate over lines in a JSONL file."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


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


async def main_async(tmp_dir: str):
    """Main function to compute statistics from tmp directory."""
    start_time = time.time()

    # Determine paths
    parent_dir = os.path.dirname(tmp_dir)
    frame = "reme"

    output_file_stage1 = os.path.join(parent_dir, f"{frame}_eval_results.jsonl")
    output_file_stage2 = os.path.join(parent_dir, f"{frame}_eval_stat_result.json")

    print("\n" + "=" * 80)
    print("LOADING STAGE 1 RESULTS FROM TMP DIRECTORY")
    print(f"Tmp Directory: {tmp_dir}")
    print("=" * 80)

    # Step 1: Combine all tmp JSON files into the stage1 output JSONL
    json_files = [f for f in os.listdir(tmp_dir) if f.endswith(".json")]
    print(f"\n📁 Found {len(json_files)} JSON files in tmp directory")

    with open(output_file_stage1, "w", encoding="utf-8") as f_out:
        for file_name in json_files:
            file_path = os.path.join(tmp_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f_in:
                data = json.load(f_in)
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"✅ Combined results saved to: {output_file_stage1}")

    # Step 2: Run Stage 2 evaluation (extraction only)
    print("\n" + "=" * 80)
    print("STAGE 2: EXTRACTING AND AGGREGATING EVALUATION RESULTS")
    print("=" * 80)

    tmp_dir2 = os.path.join(parent_dir, "tmp2")
    os.makedirs(tmp_dir2, exist_ok=True)

    start_stage2 = time.time()

    # Load all users and process
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


def main(tmp_dir: str):
    """Synchronous entry point."""
    asyncio.run(main_async(tmp_dir))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute statistics from existing Stage 1 tmp results")
    parser.add_argument(
        "--tmp_dir",
        type=str,
        required=True,
        help="Path to tmp directory containing Stage 1 JSON results (e.g., bench_results/reme/tmp)",
    )
    args = parser.parse_args()

    main(tmp_dir=args.tmp_dir)
