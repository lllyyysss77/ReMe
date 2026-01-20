"""
Compute Question Answering statistics from evaluation results in tmp directory.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


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


def compute_time_metrics(users_data: list[dict]) -> dict[str, float]:
    """Compute timing metrics from evaluation results."""
    add_duration = search_duration = 0
    
    for user_data in users_data:
        for session in user_data.get("sessions", []):
            add_duration += session.get("add_dialogue_duration_ms", 0)
            eval_results = session.get("session", {}).get("evaluation_results", {})
            for qa in eval_results.get("question_answering_records", []):
                search_duration += qa.get("search_duration_ms", 0)
    
    return {
        "add_dialogue_duration_time": add_duration / 1000 / 60,
        "search_memory_duration_time": search_duration / 1000 / 60,
        "total_duration_time": (add_duration + search_duration) / 1000 / 60
    }


def load_from_tmp_dir(tmp_dir: str) -> list[dict]:
    """Load data from tmp directory."""
    tmp_path = Path(tmp_dir)
    
    # Try flat file structure first (conversation_{user}_session_{idx}.json)
    json_files = [f for f in tmp_path.iterdir() if f.is_file() and f.suffix == ".json"]
    
    if json_files:
        # Group files by user
        users_dict = defaultdict(list)
        
        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)
                user_name = session_data.get("user_name")
                if user_name:
                    users_dict[user_name].append(session_data)
        
        # Sort sessions by session_idx for each user
        users_data = []
        for user_name, sessions in users_dict.items():
            sessions_sorted = sorted(sessions, key=lambda s: s.get("session_idx", 0))
            if sessions_sorted:
                user_data = {
                    "uuid": sessions_sorted[0].get("uuid"),
                    "user_name": user_name,
                    "sessions": []
                }
                for session_data in sessions_sorted:
                    session_copy = session_data.copy()
                    session_copy.pop("uuid", None)
                    session_copy.pop("user_name", None)
                    user_data["sessions"].append(session_copy)
                users_data.append(user_data)
        
        return users_data
    
    # Fallback to directory structure (user_name/session_{idx}.json)
    user_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
    
    users_data = []
    for user_dir in user_dirs:
        session_files = sorted(
            [f for f in user_dir.iterdir() if "session_" in f.name and f.suffix == ".json"],
            key=lambda f: int(f.stem.split("_")[-1])
        )
        
        if not session_files:
            continue
        
        with open(session_files[0], "r", encoding="utf-8") as f:
            first_session = json.load(f)
        
        user_data = {
            "uuid": first_session["uuid"],
            "user_name": first_session["user_name"],
            "sessions": []
        }
        
        for session_file in session_files:
            with open(session_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)
                session_data.pop("uuid", None)
                session_data.pop("user_name", None)
                user_data["sessions"].append(session_data)
        
        users_data.append(user_data)
    
    return users_data


def main(tmp_dir: str):
    """Main function to compute statistics from tmp directory."""
    tmp_path = Path(tmp_dir)
    
    if not tmp_path.exists() or not tmp_path.is_dir():
        print(f"❌ Error: Directory not found: {tmp_dir}")
        return
    
    # Load data from tmp directory
    users_data = load_from_tmp_dir(tmp_dir)
    
    # Collect QA records with metadata
    qa_records = []
    qa_with_metadata = []
    user_count = session_count = 0
    
    for user_data in users_data:
        user_count += 1
        user_name = user_data.get("user_name", "Unknown")
        
        valid_session_idx = 0
        for session in user_data.get("sessions", []):
            if session.get("is_generated_qa_session"):
                continue
            
            session_count += 1
            eval_results = session.get("session", {}).get("evaluation_results", {})
            
            for qa_idx, qa in enumerate(eval_results.get("question_answering_records", [])):
                qa_records.append(qa)
                qa_with_metadata.append({
                    "user_name": user_name,
                    "session_idx": valid_session_idx,
                    "question_idx": qa_idx,
                    "qa_record": qa
                })
            
            valid_session_idx += 1
    
    # Compute metrics
    qa_metrics = compute_qa_metrics(qa_records)
    time_metrics = compute_time_metrics(users_data)
    
    # Save results
    output_dir = tmp_path.parent
    report_file = output_dir / "reme_eval_stat_result.json"
    
    final_results = {
        "overall_score": {
            "question_answering": qa_metrics,
            "time_consuming": time_metrics
        },
        "question_answering_records": qa_records
    }
    
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    
    # Print summary
    print(f"\n📊 Data: {user_count} users, {session_count} sessions, {len(qa_records)} QA records")
    print(f"\n✅ Metrics:")
    print(f"  Correct: {qa_metrics['correct_qa_ratio(all)']:.4f} | Hallucination: {qa_metrics['hallucination_qa_ratio(all)']:.4f} | Omission: {qa_metrics['omission_qa_ratio(all)']:.4f}")
    print(f"  Valid: {qa_metrics['qa_valid_num']}/{qa_metrics['qa_num']}")
    print(f"\n⏱️  Time: {time_metrics['total_duration_time']:.2f} min (Add: {time_metrics['add_dialogue_duration_time']:.2f} | Search: {time_metrics['search_memory_duration_time']:.2f})")
    print(f"\n💾 Results saved: {report_file}")
    
    # Print error records
    print(f"\n{'='*80}\n❌ ERROR RECORDS ({len([r for r in qa_with_metadata if r['qa_record'].get('result_type') not in ['Correct', '']])} errors)\n{'='*80}")
    
    error_records = [r for r in qa_with_metadata if r["qa_record"].get("result_type") not in ["Correct", ""]]
    
    if error_records:
        for idx, record in enumerate(error_records, 1):
            qa = record["qa_record"]
            print(f"\n[{idx}] {qa.get('result_type', 'Unknown')} - {record['user_name']} (S{record['session_idx']}/Q{record['question_idx']})")
            print(f"  Q: {qa.get('question', 'N/A')}")
            print(f"  Expected: {qa.get('answer', 'N/A')}")
            print(f"  Got: {qa.get('system_response', 'N/A')}")
    
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute QA statistics from tmp directory")
    parser.add_argument(
        "tmp_dir",
        nargs='?',
        default="./data",
        type=str,
        help="Path to tmp directory containing user session data (default: ./data)")
    
    args = parser.parse_args()
    main(tmp_dir=args.tmp_dir)
