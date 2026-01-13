"""
HaluMem Benchmark Evaluator - Baseline (Direct QA without Memory System)

A simple baseline evaluation pipeline that:
1. Loads HaluMem benchmark data
2. Directly uses dialogue history to answer questions (no memory system)
3. Evaluates question answering performance
4. Generates comprehensive metrics

Usage:
    python bench/halumem/eval_baseline_simple.py \
        --data_path /path/to/HaluMem-Medium.jsonl \
        --user_num 100 --max_concurrency 20
"""

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from eval_tools import evaluation_for_question2
from llms import llm_request_for_json


# ==================== Configuration ====================

@dataclass
class EvalConfig:
    """Evaluation configuration parameters."""
    data_path: str
    user_num: int = 1
    max_concurrency: int = 2
    output_dir: str = "bench_results/baseline_simple"


# ==================== Utilities ====================

class DataLoader:
    """Handles loading and parsing of HaluMem data."""
    
    @staticmethod
    def load_jsonl(file_path: str) -> list[dict]:
        """Load all entries from a JSONL file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line.strip()) for line in f if line.strip()]
    
    @staticmethod
    def extract_user_name(persona_info: str) -> str:
        """Extract user name from persona info string."""
        match = re.search(r"Name:\s*(.*?); Gender:", persona_info)
        if not match:
            raise ValueError(f"No name found in persona_info: {persona_info}")
        return match.group(1).strip()
    
    @staticmethod
    def format_dialogue_messages(dialogue: list[dict]) -> list[dict]:
        """Format dialogue into ReMe message format."""
        return [
            {
                "role": turn["role"],
                "content": turn["content"],
                "time_created": datetime.strptime(
                    turn["timestamp"], "%b %d, %Y, %H:%M:%S"
                )
                .replace(tzinfo=timezone.utc)
                .strftime("%Y-%m-%d %H:%M:%S"),
            }
            for turn in dialogue
        ]
    
    @staticmethod
    def format_dialogue_for_eval(dialogue: list[dict], user_name: str = None) -> str:
        """Format dialogue into string for evaluation."""
        formatted_turns = []
        for turn in dialogue:
            timestamp = datetime.strptime(
                turn["timestamp"], "%b %d, %Y, %H:%M:%S"
            ).replace(tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            
            # Use user_name if role is 'user' and user_name is provided
            role = user_name if turn['role'] == 'user' and user_name else turn['role']
            
            formatted_turns.append(
                f"Role: {role}\n"
                f"Content: {turn['content']}\n"
                f"Time: {timestamp}"
            )
        return "\n\n".join(formatted_turns)


class FileManager:
    """Manages file I/O operations."""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.tmp_dir = self.base_dir / "tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
    
    def get_user_dir(self, user_name: str) -> Path:
        """Get the directory path for a user."""
        user_dir = self.tmp_dir / user_name
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir
    
    def get_session_file(self, user_name: str, session_id: int) -> Path:
        """Get the file path for a specific session."""
        return self.get_user_dir(user_name) / f"session_{session_id}.json"
    
    def save_session(self, user_name: str, session_id: int, data: dict):
        """Save session data to file."""
        file_path = self.get_session_file(user_name, session_id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Saved session {session_id} to {file_path}")
    
    def load_session(self, user_name: str, session_id: int) -> dict | None:
        """Load session data from file."""
        file_path = self.get_session_file(user_name, session_id)
        if not file_path.exists():
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def user_has_cache(self, user_name: str) -> bool:
        """Check if user has cached results."""
        user_dir = self.get_user_dir(user_name)
        return any(f.name.startswith("session_") and f.suffix == ".json" 
                  for f in user_dir.iterdir())
    
    def combine_results(self, output_file: str):
        """Combine all user session files into a single JSONL file."""
        with open(output_file, "w", encoding="utf-8") as f_out:
            for user_dir in self.tmp_dir.iterdir():
                if not user_dir.is_dir():
                    continue
                
                session_files = sorted([
                    f for f in user_dir.iterdir() 
                    if f.name.startswith("session_") and f.suffix == ".json"
                ])
                
                if not session_files:
                    continue
                
                # Load first session to get user metadata
                with open(session_files[0], "r", encoding="utf-8") as f_in:
                    first_session = json.load(f_in)
                
                user_data = {
                    "uuid": first_session["uuid"],
                    "user_name": first_session["user_name"],
                    "sessions": []
                }
                
                # Load all sessions
                for session_file in session_files:
                    with open(session_file, "r", encoding="utf-8") as f_in:
                        session_data = json.load(f_in)
                        # Remove redundant user metadata
                        session_data.pop("uuid", None)
                        session_data.pop("user_name", None)
                        user_data["sessions"].append(session_data)
                
                f_out.write(json.dumps(user_data, ensure_ascii=False) + "\n")


# ==================== Question Answering Prompt ====================

BASELINE_QA_PROMPT = """You are a helpful AI assistant. Based on the dialogue history provided below, please answer the question.

**Dialogue History:**
{dialogue}

**Question:**
{question}

**Instructions:**
- Carefully read through the dialogue history
- Answer the question based ONLY on information present in the dialogue
- If the information needed to answer the question is NOT in the dialogue, respond with "I don't know" or "The information is not available in the dialogue"
- Do NOT make up or hallucinate information that is not explicitly mentioned in the dialogue
- Provide your reasoning process before giving the final answer

**Response Format:**
Please respond in JSON format with the following structure:
```json
{{
    "reasoning": "Your step-by-step reasoning process",
    "answer": "Your final answer (or 'I don't know' if information is not available)"
}}
```"""


# ==================== Evaluation ====================

class BaselineQuestionAnsweringEvaluator:
    """Evaluates question answering performance using direct LLM inference (no memory system)."""
    
    def __init__(self):
        pass
    
    async def answer_question(
        self,
        question: str,
        formatted_dialogue: str
    ) -> tuple[str, str, float]:
        """
        Answer a question using the dialogue history directly.
        
        Returns:
            tuple: (answer, reasoning, duration_ms)
        """
        start = time.time()
        
        # Format prompt
        prompt = BASELINE_QA_PROMPT.format(
            dialogue=formatted_dialogue,
            question=question
        )
        
        # Get answer from LLM
        try:
            # model_name = "qwen3-max"
            model_name = "qwen3-30b-a3b-instruct-2507"
            result = await llm_request_for_json(prompt, model_name=model_name)
            answer = result.get("answer", "I don't know")
            reasoning = result.get("reasoning", "")
        except Exception as e:
            logger.error(f"Error getting answer from LLM: {e}")
            answer = "Error: Failed to get answer"
            reasoning = str(e)
        
        duration_ms = (time.time() - start) * 1000
        return answer, reasoning, duration_ms
    
    async def evaluate_questions(
        self,
        questions: list[dict],
        user_name: str,
        uuid: str,
        session_id: int,
        formatted_dialogue: str
    ) -> list[dict]:
        """Evaluate all questions for a session."""
        results = []
        
        for qa in questions:
            # Get answer directly from LLM
            answer, reasoning, duration_ms = await self.answer_question(
                question=qa["question"],
                formatted_dialogue=formatted_dialogue
            )
            
            # Evaluate response
            evidence_text = "\n".join([e["memory_content"] for e in qa["evidence"]])
            eval_result = await evaluation_for_question2(
                qa["question"],
                qa["answer"],
                evidence_text,
                answer,
                formatted_dialogue
            )
            
            # Build result record
            qa_result = {
                **qa,
                "uuid": uuid,
                "session_id": session_id,
                "system_response": answer,
                "reasoning": reasoning,
                "answer_duration_ms": duration_ms,
                "result_type": eval_result.get("evaluation_result"),
                "question_answering_reasoning": eval_result.get("reasoning", "")
            }
            results.append(qa_result)
        
        return results


class MetricsAggregator:
    """Aggregates evaluation metrics."""
    
    @staticmethod
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
    
    @staticmethod
    def compute_time_metrics(eval_results_file: str) -> dict[str, float]:
        """Compute timing metrics from evaluation results."""
        answer_duration = 0
        
        with open(eval_results_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                user_data = json.loads(line)
                
                for session in user_data["sessions"]:
                    eval_results = session.get("evaluation_results", {})
                    for qa in eval_results.get("question_answering_records", []):
                        answer_duration += qa.get("answer_duration_ms", 0)
        
        # Convert to minutes
        return {
            "answer_duration_time": answer_duration / 1000 / 60,
            "total_duration_time": answer_duration / 1000 / 60
        }


# ==================== Main Pipeline ====================

class HaluMemBaselineEvaluator:
    """Main evaluator orchestrating the baseline evaluation pipeline."""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.file_manager = FileManager(config.output_dir)
        self.qa_evaluator = BaselineQuestionAnsweringEvaluator()
        self.data_loader = DataLoader()
    
    async def process_session(
        self,
        session: dict,
        session_id: int,
        user_name: str,
        uuid: str
    ) -> dict:
        """Process a single session."""
        session_data = {
            "uuid": uuid,
            "user_name": user_name,
            "session_id": session_id,
            "memory_points": session["memory_points"]
        }
        
        # Skip generated QA sessions
        if session.get("is_generated_qa_session", False):
            session_data["is_generated_qa_session"] = True
            return session_data
        
        # Store dialogue
        dialogue = session["dialogue"]
        session_data["dialogue"] = dialogue
        
        # Evaluate questions if present
        if "questions" in session:
            formatted_dialogue = self.data_loader.format_dialogue_for_eval(dialogue, user_name)
            qa_results = await self.qa_evaluator.evaluate_questions(
                questions=session["questions"],
                user_name=user_name,
                uuid=uuid,
                session_id=session_id,
                formatted_dialogue=formatted_dialogue
            )
            
            session_data["evaluation_results"] = {
                "question_answering_records": qa_results
            }
        
        return session_data
    
    async def process_user(self, user_data: dict) -> dict:
        """Process all sessions for a user."""
        user_name = self.data_loader.extract_user_name(user_data["persona_info"])
        uuid = user_data["uuid"]
        
        total_sessions = len(user_data["sessions"])
        logger.info(f"Processing user: {user_name} ({total_sessions} sessions)")
        
        # Semaphore for concurrency control within user sessions
        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        completed_count = [0]  # Use list to allow modification in nested async function
        
        async def process_session_with_log(idx: int, session: dict):
            async with semaphore:
                session_data = await self.process_session(
                    session=session,
                    session_id=idx,
                    user_name=user_name,
                    uuid=uuid
                )
                
                self.file_manager.save_session(user_name, idx, session_data)
                
                # Update and log completion
                completed_count[0] += 1
                print(f"✅ {user_name} complete {completed_count[0]}/{total_sessions}")
        
        # Process all sessions in parallel
        tasks = [
            process_session_with_log(idx, session)
            for idx, session in enumerate(user_data["sessions"])
        ]
        await asyncio.gather(*tasks)
        
        return {"uuid": uuid, "user_name": user_name, "status": "ok"}
    
    async def run_evaluation(self):
        """Run the complete evaluation pipeline."""
        start_time = time.time()
        
        # Load user data
        all_users = self.data_loader.load_jsonl(self.config.data_path)
        users_to_process = all_users[:self.config.user_num]
        
        print("\n" + "=" * 80)
        print("HALUMEM BASELINE EVALUATION - DIRECT QA WITHOUT MEMORY SYSTEM")
        print(f"Users: {len(users_to_process)} | Session Concurrency: {self.config.max_concurrency}")
        print("=" * 80 + "\n")
        
        # Process users sequentially (for loop)
        for idx, user_data in enumerate(users_to_process, 1):
            user_name = self.data_loader.extract_user_name(user_data["persona_info"])
            
            # Check cache
            if self.file_manager.user_has_cache(user_name):
                print(f"⚡ [{idx}/{len(users_to_process)}] Skipping {user_name} (cached)")
                continue
            
            print(f"🔄 [{idx}/{len(users_to_process)}] Processing {user_name}...")
            await self.process_user(user_data)
            print(f"✅ [{idx}/{len(users_to_process)}] User {user_name} completed\n")
        
        # Combine results
        output_file = os.path.join(self.config.output_dir, "eval_results.jsonl")
        self.file_manager.combine_results(output_file)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Processing completed in {elapsed:.2f}s")
        print(f"📁 Results: {output_file}\n")
        
        # Aggregate metrics
        await self.aggregate_and_report(output_file)
    
    async def aggregate_and_report(self, results_file: str):
        """Aggregate results and generate final report."""
        print("=" * 80)
        print("AGGREGATING METRICS")
        print("=" * 80 + "\n")
        
        # Collect all QA records
        qa_records = []
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                user_data = json.loads(line)
                
                for session in user_data["sessions"]:
                    if session.get("is_generated_qa_session"):
                        continue
                    
                    eval_results = session.get("evaluation_results", {})
                    qa_records.extend(
                        eval_results.get("question_answering_records", [])
                    )
        
        # Compute metrics
        qa_metrics = MetricsAggregator.compute_qa_metrics(qa_records)
        time_metrics = MetricsAggregator.compute_time_metrics(results_file)
        
        final_results = {
            "overall_score": {
                "question_answering": qa_metrics,
                "time_consuming": time_metrics
            },
            "question_answering_records": qa_records
        }
        
        # Save final report
        report_file = os.path.join(self.config.output_dir, "eval_statistics.json")
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)
        
        print(f"📊 Statistics saved to: {report_file}\n")
        
        # Print summary
        self._print_summary(qa_metrics, time_metrics)
    
    def _print_summary(self, qa_metrics: dict, time_metrics: dict):
        """Print evaluation summary."""
        print("=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80 + "\n")
        
        print("📊 Question Answering:")
        print(f"  Correct (all):       {qa_metrics['correct_qa_ratio(all)']:.4f}")
        print(f"  Hallucination (all): {qa_metrics['hallucination_qa_ratio(all)']:.4f}")
        print(f"  Omission (all):      {qa_metrics['omission_qa_ratio(all)']:.4f}")
        print(f"  Correct (valid):     {qa_metrics['correct_qa_ratio(valid)']:.4f}")
        print(f"  Hallucination (valid): {qa_metrics['hallucination_qa_ratio(valid)']:.4f}")
        print(f"  Omission (valid):    {qa_metrics['omission_qa_ratio(valid)']:.4f}")
        print(f"  Valid/Total:         {qa_metrics['qa_valid_num']}/{qa_metrics['qa_num']}")
        
        print(f"\n⏱️  Time Metrics:")
        print(f"  Answer Duration:  {time_metrics['answer_duration_time']:.2f} min")
        print(f"  Total:            {time_metrics['total_duration_time']:.2f} min")
        print("\n" + "=" * 80)


# ==================== Entry Point ====================

def main(
    data_path: str,
    user_num: int = 1,
    max_concurrency: int = 2
):
    """Main entry point."""
    config = EvalConfig(
        data_path=data_path,
        user_num=user_num,
        max_concurrency=max_concurrency
    )
    
    evaluator = HaluMemBaselineEvaluator(config)
    asyncio.run(evaluator.run_evaluation())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate Baseline (Direct QA) on HaluMem benchmark"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to HaluMem JSONL file"
    )
    parser.add_argument(
        "--user_num",
        type=int,
        default=1,
        help="Number of users to evaluate (default: 1)"
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=2,
        help="Maximum concurrent user processing (default: 2)"
    )
    
    args = parser.parse_args()
    
    main(
        data_path=args.data_path,
        user_num=args.user_num,
        max_concurrency=args.max_concurrency
    )
