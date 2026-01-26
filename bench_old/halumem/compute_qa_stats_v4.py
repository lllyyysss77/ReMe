"""
Compute Question Answering statistics from eval_reme_simple_v4.py results.

Usage:
    python bench/halumem/compute_qa_stats_v4.py --results_file bench_results/reme_simple_v4/eval_results.jsonl
    python bench/halumem/compute_qa_stats_v4.py --tmp_dir bench_results/reme_simple_v4/tmp
"""

import json
import os
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


def compute_time_metrics(results_file: str) -> dict[str, float]:
    """Compute timing metrics from evaluation results."""
    add_duration = search_duration = 0

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

    return {
        "add_dialogue_duration_time": add_duration / 1000 / 60,
        "search_memory_duration_time": search_duration / 1000 / 60,
        "total_duration_time": (add_duration + search_duration) / 1000 / 60
    }


def load_from_tmp_dir(tmp_dir: str) -> str:
    """Load data from tmp directory and generate eval_results.jsonl file."""
    tmp_path = Path(tmp_dir)
    eval_results_file = tmp_path.parent / "eval_results.jsonl"

    print(f"\n📁 Loading from: {tmp_dir}")
    print(f"📝 Generating: {eval_results_file}")

    user_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
    print(f"   Found {len(user_dirs)} users")

    users_data = []
    for user_dir in user_dirs:
        session_files = sorted(
            [f for f in user_dir.iterdir() if f.name.startswith("session_") and f.suffix == ".json"],
            key=lambda f: int(f.stem.split("_")[1])
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
        print(f"   ✓ {user_dir.name}: {len(session_files)} sessions")

    with open(eval_results_file, "w", encoding="utf-8") as f:
        for user_data in users_data:
            f.write(json.dumps(user_data, ensure_ascii=False) + "\n")

    print(f"   ✅ Generated: {eval_results_file}")
    return str(eval_results_file)


def main(input_path: str):
    """Main function to compute statistics from eval results."""

    if not os.path.exists(input_path):
        print(f"❌ Error: Path not found: {input_path}")
        return

    print("\n" + "=" * 80)
    print("REME V4 - QUESTION ANSWERING STATISTICS")
    print("=" * 80)

    # Load or generate eval_results.jsonl
    if os.path.isdir(input_path):
        results_file = load_from_tmp_dir(input_path)
    else:
        results_file = input_path
        print(f"\n📁 Using: {results_file}")

    # Collect QA records with metadata
    qa_records = []
    qa_with_metadata = []
    user_count = session_count = 0

    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            user_data = json.loads(line)
            user_count += 1
            user_name = user_data.get("user_name", "Unknown")

            valid_session_idx = 0
            for original_idx, session in enumerate(user_data.get("sessions", [])):
                if session.get("is_generated_qa_session"):
                    continue

                session_count += 1
                eval_results = session.get("evaluation_results", {})

                for qa_idx, qa in enumerate(eval_results.get("question_answering_records", [])):
                    qa_records.append(qa)
                    qa_with_metadata.append({
                        "user_name": user_name,
                        "session_idx": valid_session_idx,
                        "question_idx": qa_idx,
                        "qa_record": qa
                    })

                valid_session_idx += 1

    print(f"\n📊 Data Summary:")
    print(f"  Users: {user_count}")
    print(f"  Sessions: {session_count}")
    print(f"  QA Records: {len(qa_records)}")

    # Compute metrics
    qa_metrics = compute_qa_metrics(qa_records)
    time_metrics = compute_time_metrics(results_file)

    # Save results
    output_dir = Path(results_file).parent
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

    print(f"\n✅ Results saved to: {report_file}")

    # Print metrics
    print("\n" + "=" * 80)
    print("📊 QUESTION ANSWERING METRICS")
    print("=" * 80)
    print(f"\n  Correct (all):       {qa_metrics['correct_qa_ratio(all)']:.4f}")
    print(f"  Hallucination (all): {qa_metrics['hallucination_qa_ratio(all)']:.4f}")
    print(f"  Omission (all):      {qa_metrics['omission_qa_ratio(all)']:.4f}")
    print(f"  Correct (valid):     {qa_metrics['correct_qa_ratio(valid)']:.4f}")
    print(f"  Hallucination (valid): {qa_metrics['hallucination_qa_ratio(valid)']:.4f}")
    print(f"  Omission (valid):    {qa_metrics['omission_qa_ratio(valid)']:.4f}")
    print(f"  Valid/Total:         {qa_metrics['qa_valid_num']}/{qa_metrics['qa_num']}")

    print(f"\n⏱️  TIME METRICS")
    print(f"  Memory Addition:  {time_metrics['add_dialogue_duration_time']:.2f} min")
    print(f"  Memory Search:    {time_metrics['search_memory_duration_time']:.2f} min")
    print(f"  Total:            {time_metrics['total_duration_time']:.2f} min")

    # Print error records
    print("\n" + "=" * 80)
    print("❌ ERROR RECORDS (Non-Correct)")
    print("=" * 80)

    error_records = [r for r in qa_with_metadata if r["qa_record"].get("result_type") not in ["Correct", ""]]

    if not error_records:
        print("\n✅ All QA records are correct!")
    else:
        print(f"\nFound {len(error_records)} error records:\n")

        for idx, record in enumerate(error_records, 1):
            qa = record["qa_record"]

            print(f"\n{'━' * 80}")
            print(f"❌ ERROR #{idx}")
            print(f"{'━' * 80}")
            print(f"👤 User: {record['user_name']}")
            print(f"📅 Session: {record['session_idx']} | Question: {record['question_idx']}")
            print(f"🏷️  Result Type: {qa.get('result_type', 'Unknown')}")
            print(f"\n❓ Question:")
            print(f"   {qa.get('question', 'N/A')}")
            print(f"\n✅ Expected Answer:")
            print(f"   {qa.get('answer', 'N/A')}")
            print(f"\n🤖 System Response:")
            print(f"   {qa.get('system_response', 'N/A')}")
            print(f"\n💭 Reasoning:")
            reason = qa.get('question_answering_reasoning', 'N/A')
            # Wrap long reasoning text
            if len(reason) > 80:
                words = reason.split()
                lines = []
                current_line = "   "
                for word in words:
                    if len(current_line) + len(word) + 1 <= 80:
                        current_line += word + " "
                    else:
                        lines.append(current_line.rstrip())
                        current_line = "   " + word + " "
                if current_line.strip():
                    lines.append(current_line.rstrip())
                print("\n".join(lines))
            else:
                print(f"   {reason}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute QA statistics from eval_reme_simple_v4.py results")
    parser.add_argument("--results_file", type=str, help="Path to eval_results.jsonl file")
    parser.add_argument("--tmp_dir", type=str, help="Path to tmp directory")

    args = parser.parse_args()

    if args.tmp_dir:
        main(input_path=args.tmp_dir)
    elif args.results_file:
        main(input_path=args.results_file)
    else:
        parser.error("Either --results_file or --tmp_dir must be provided")
