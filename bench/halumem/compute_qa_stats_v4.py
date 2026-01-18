"""
Compute Question Answering statistics from eval_reme_simple_v4.py results.

This script processes the output from eval_reme_simple_v4.py and computes
comprehensive QA metrics.

Usage:
    python bench/halumem/compute_qa_stats_v4.py --results_file bench_results/reme_simple_v4/eval_results.jsonl
"""

import json
import os
from pathlib import Path
from typing import Any

from loguru import logger


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
    
    correct = 0
    hallucination = 0
    omission = 0
    valid = 0
    
    for qa in qa_records:
        result_type = qa.get("result_type", "")
        
        if result_type in ["Correct", "Hallucination", "Omission"]:
            valid += 1
            if result_type == "Correct":
                correct += 1
            elif result_type == "Hallucination":
                hallucination += 1
            elif result_type == "Omission":
                omission += 1
    
    metrics = {
        "correct_qa_ratio(all)": correct / total,
        "hallucination_qa_ratio(all)": hallucination / total,
        "omission_qa_ratio(all)": omission / total,
        "qa_valid_num": valid,
        "qa_num": total
    }
    
    if valid > 0:
        metrics.update({
            "correct_qa_ratio(valid)": correct / valid,
            "hallucination_qa_ratio(valid)": hallucination / valid,
            "omission_qa_ratio(valid)": omission / valid
        })
    else:
        metrics.update({
            "correct_qa_ratio(valid)": 0,
            "hallucination_qa_ratio(valid)": 0,
            "omission_qa_ratio(valid)": 0
        })
    
    return metrics


def compute_time_metrics(results_file: str) -> dict[str, float]:
    """Compute timing metrics from evaluation results."""
    add_duration = 0
    search_duration = 0
    
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            user_data = json.loads(line)
            
            for session in user_data.get("sessions", []):
                add_duration += session.get("add_dialogue_duration_ms", 0)
                
                eval_results = session.get("evaluation_results", {})
                for qa in eval_results.get("question_answering_records", []):
                    search_duration += qa.get("search_duration_ms", 0)
    
    # Convert to minutes
    return {
        "add_dialogue_duration_time": add_duration / 1000 / 60,
        "search_memory_duration_time": search_duration / 1000 / 60,
        "total_duration_time": (add_duration + search_duration) / 1000 / 60
    }


def load_from_tmp_dir(tmp_dir: str) -> tuple[str, list[dict]]:
    """Load data from tmp directory and generate eval_results.jsonl file."""
    tmp_path = Path(tmp_dir)
    parent_dir = tmp_path.parent
    eval_results_file = parent_dir / "eval_results.jsonl"
    
    print(f"\n📁 Loading data from tmp directory: {tmp_dir}")
    print(f"📝 Will generate: {eval_results_file}")
    
    # Collect all user directories
    user_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
    print(f"   Found {len(user_dirs)} user directories")
    
    users_data = []
    
    for user_dir in user_dirs:
        user_name = user_dir.name
        
        # Load all session files for this user (sorted by session number)
        session_files = sorted(
            [f for f in user_dir.iterdir() 
             if f.name.startswith("session_") and f.suffix == ".json"],
            key=lambda f: int(f.stem.split("_")[1])  # Sort by session number
        )
        
        if not session_files:
            print(f"   ⚠️  No session files found for user: {user_name}")
            continue
        
        # Load first session to get user metadata
        with open(session_files[0], "r", encoding="utf-8") as f:
            first_session = json.load(f)
        
        user_data = {
            "uuid": first_session["uuid"],
            "user_name": first_session["user_name"],
            "sessions": []
        }
        
        # Load all sessions
        for session_file in session_files:
            with open(session_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)
                # Remove redundant user metadata
                session_data.pop("uuid", None)
                session_data.pop("user_name", None)
                user_data["sessions"].append(session_data)
        
        users_data.append(user_data)
        print(f"   ✓ Loaded user {user_name}: {len(session_files)} sessions")
    
    # Write to eval_results.jsonl
    with open(eval_results_file, "w", encoding="utf-8") as f:
        for user_data in users_data:
            f.write(json.dumps(user_data, ensure_ascii=False) + "\n")
    
    print(f"   ✅ Generated: {eval_results_file}")
    
    return str(eval_results_file), users_data


def main(input_path: str):
    """Main function to compute statistics from eval results."""
    
    if not os.path.exists(input_path):
        logger.error(f"Input path not found: {input_path}")
        return
    
    print("\n" + "=" * 80)
    print("COMPUTING QUESTION ANSWERING STATISTICS - REME V4")
    print("=" * 80)
    
    # Determine if input is a directory (tmp) or file (eval_results.jsonl)
    if os.path.isdir(input_path):
        results_file, users_data = load_from_tmp_dir(input_path)
    else:
        results_file = input_path
        users_data = None
        print(f"\n📁 Using existing results file: {results_file}")
    
    # Collect all QA records with metadata
    qa_records = []
    qa_records_with_metadata = []  # Store records with user/session/question info
    user_count = 0
    session_count = 0
    
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            user_data = json.loads(line)
            user_count += 1
            user_name = user_data.get("user_name", "Unknown")
            
            valid_session_idx = 0  # Track the index of valid (non-skipped) sessions
            for original_idx, session in enumerate(user_data.get("sessions", [])):
                if session.get("is_generated_qa_session"):
                    continue
                
                session_count += 1
                eval_results = session.get("evaluation_results", {})
                session_qa_records = eval_results.get("question_answering_records", [])
                
                # Add records with metadata
                for qa_idx, qa in enumerate(session_qa_records):
                    qa_records.append(qa)
                    qa_records_with_metadata.append({
                        "user_name": user_name,
                        "session_idx": valid_session_idx,
                        "original_session_idx": original_idx,
                        "question_idx": qa_idx,
                        "qa_record": qa
                    })
                
                valid_session_idx += 1
    
    print(f"\n📊 Data loaded:")
    print(f"  Users: {user_count}")
    print(f"  Sessions: {session_count}")
    print(f"  QA Records: {len(qa_records)}")
    
    # Compute metrics
    print("\n🔄 Computing metrics...")
    qa_metrics = compute_qa_metrics(qa_records)
    time_metrics = compute_time_metrics(results_file)
    
    final_results = {
        "overall_score": {
            "question_answering": qa_metrics,
            "time_consuming": time_metrics
        },
        "question_answering_records": qa_records
    }
    
    # Save final report
    output_dir = Path(results_file).parent
    report_file = output_dir / "reme_eval_stat_result.json"
    
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    
    print(f"\n✅ Statistics saved to: {report_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY - REME V4")
    print("=" * 80)
    
    print("\n📊 Question Answering:")
    print(f"  Correct (all):       {qa_metrics['correct_qa_ratio(all)']:.4f}")
    print(f"  Hallucination (all): {qa_metrics['hallucination_qa_ratio(all)']:.4f}")
    print(f"  Omission (all):      {qa_metrics['omission_qa_ratio(all)']:.4f}")
    print(f"  Correct (valid):     {qa_metrics['correct_qa_ratio(valid)']:.4f}")
    print(f"  Hallucination (valid): {qa_metrics['hallucination_qa_ratio(valid)']:.4f}")
    print(f"  Omission (valid):    {qa_metrics['omission_qa_ratio(valid)']:.4f}")
    print(f"  Valid/Total:         {qa_metrics['qa_valid_num']}/{qa_metrics['qa_num']}")
    
    print(f"\n⏱️  Time Metrics:")
    print(f"  Memory Addition:  {time_metrics['add_dialogue_duration_time']:.2f} min")
    print(f"  Memory Search:    {time_metrics['search_memory_duration_time']:.2f} min")
    print(f"  Total:            {time_metrics['total_duration_time']:.2f} min")
    
    # Print non-Correct QA records
    print("\n" + "=" * 80)
    print("NON-CORRECT QA RECORDS")
    print("=" * 80)
    
    non_correct_records = [
        record for record in qa_records_with_metadata 
        if record["qa_record"].get("result_type") not in ["Correct", ""]
    ]
    
    if non_correct_records:
        print(f"\nFound {len(non_correct_records)} non-correct records:\n")
        for record in non_correct_records:
            user_name = record["user_name"]
            session_idx = record["session_idx"]
            original_idx = record["original_session_idx"]
            question_idx = record["question_idx"]
            qa = record["qa_record"]
            result_type = qa.get("result_type", "Unknown")
            question = qa.get("question", "N/A")
            answer = qa.get("answer", "N/A")
            
            print(f"👤 User: {user_name}")
            print(f"📅 Session: {original_idx} (valid session index: {session_idx})")
            print(f"❓ Question #{question_idx}")
            print(f"🏷️  Result Type: {result_type}")
            print(f"💬 Question: {question}")
            print(f"💡 Answer: {answer}")
            print("-" * 80)
    else:
        print("\n✅ All QA records are Correct!")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute QA statistics from eval_reme_simple_v4.py results"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        required=False,
        help="Path to eval_results.jsonl file (e.g., bench_results/reme_simple_v4/eval_results.jsonl)"
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        required=False,
        help="Path to tmp directory (e.g., bench_results/reme_simple_v4/tmp)"
    )
    
    args = parser.parse_args()
    
    # Determine input path
    if args.tmp_dir:
        input_path = args.tmp_dir
    elif args.results_file:
        input_path = args.results_file
    else:
        parser.error("Either --results_file or --tmp_dir must be provided")
    
    main(input_path=input_path)
