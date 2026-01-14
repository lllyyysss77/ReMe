"""
分析 bench_results/reme_simple/tmp 目录下的评估结果

统计所有用户session中的result_type分布，并输出非Correct结果的详细位置信息。
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def analyze_results(tmp_dir: str = "bench_results/reme_simple/tmp"):
    """
    分析评估结果目录。
    
    Args:
        tmp_dir: 临时结果目录路径
    """
    tmp_path = Path(tmp_dir)
    
    if not tmp_path.exists():
        print(f"❌ 目录不存在: {tmp_dir}")
        return
    
    # 统计数据
    result_counter = Counter()
    non_correct_results = []  # 存储非Correct结果的详细信息
    
    # 遍历所有用户目录
    user_dirs = sorted([d for d in tmp_path.iterdir() if d.is_dir()])
    
    if not user_dirs:
        print(f"❌ {tmp_dir} 下没有用户目录")
        return
    
    print(f"📁 找到 {len(user_dirs)} 个用户目录\n")
    print("=" * 80)
    print("开始分析...")
    print("=" * 80 + "\n")
    
    total_sessions = 0
    total_questions = 0
    
    # 遍历每个用户目录
    for user_dir in user_dirs:
        user_name = user_dir.name
        
        # 获取该用户的所有session文件
        session_files = sorted([
            f for f in user_dir.iterdir() 
            if f.name.startswith("session_") and f.suffix == ".json"
        ])
        
        if not session_files:
            continue
        
        # 遍历每个session
        for session_file in session_files:
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
                
                session_id = session_data.get("session_id", -1)
                total_sessions += 1
                
                # 跳过生成的QA session
                if session_data.get("is_generated_qa_session", False):
                    continue
                
                # 获取评估结果
                eval_results = session_data.get("evaluation_results", {})
                qa_records = eval_results.get("question_answering_records", [])
                
                # 分析每个问题的结果
                for qa_idx, qa_record in enumerate(qa_records):
                    result_type = qa_record.get("result_type", "Unknown")
                    
                    # 统计result_type
                    result_counter[result_type] += 1
                    total_questions += 1
                    
                    # 如果不是Correct，记录详细信息
                    if result_type != "Correct":
                        non_correct_results.append({
                            "user_name": user_name,
                            "session_id": session_id,
                            "question_id": qa_idx,
                            "result_type": result_type,
                            "question": qa_record.get("question", ""),
                            "answer": qa_record.get("answer", ""),
                            "system_response": qa_record.get("system_response", "")
                        })
                
            except Exception as e:
                print(f"⚠️  读取文件失败: {session_file}, 错误: {e}")
                continue
    
    # 输出统计结果
    print("\n" + "=" * 80)
    print("统计结果")
    print("=" * 80 + "\n")
    
    print(f"📊 总用户数: {len(user_dirs)}")
    print(f"📊 总Session数: {total_sessions}")
    print(f"📊 总问题数: {total_questions}\n")
    
    if total_questions == 0:
        print("❌ 没有找到任何问题数据")
        return
    
    # 输出result_type分布
    print("=" * 80)
    print("Result Type 分布")
    print("=" * 80 + "\n")
    
    # 按数量降序排列
    sorted_results = sorted(result_counter.items(), key=lambda x: x[1], reverse=True)
    
    for result_type, count in sorted_results:
        ratio = count / total_questions * 100
        print(f"  {result_type:20s}: {count:5d} ({ratio:6.2f}%)")
    
    # 输出非Correct结果的详细信息
    if non_correct_results:
        print("\n" + "=" * 80)
        print(f"非 Correct 结果详情 (共 {len(non_correct_results)} 条)")
        print("=" * 80 + "\n")
        
        for idx, result in enumerate(non_correct_results, 1):
            print(f"[{idx}] {result['result_type']}")
            print(f"    用户: {result['user_name']}")
            print(f"    位置: Session {result['session_id']}, Question {result['question_id']}")
            print(f"    问题: {result['question']}")
            print(f"    正确答案: {result['answer']}")
            print(f"    系统回答: {result['system_response'][:200]}{'...' if len(result['system_response']) > 200 else ''}")
            print()
    
    else:
        print("\n🎉 所有问题都是 Correct!")
    
    # 保存详细报告到文件
    report_file = Path(tmp_dir).parent / "analysis_report.json"
    report_data = {
        "summary": {
            "total_users": len(user_dirs),
            "total_sessions": total_sessions,
            "total_questions": total_questions,
            "result_type_distribution": dict(result_counter),
            "result_type_ratio": {
                result_type: count / total_questions 
                for result_type, count in result_counter.items()
            }
        },
        "non_correct_results": non_correct_results
    }
    
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    print("=" * 80)
    print(f"📄 详细报告已保存到: {report_file}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="分析 ReMe 评估结果中的 result_type 分布"
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        default="bench_results/reme_simple/tmp",
        help="临时结果目录路径 (默认: bench_results/reme_simple/tmp)"
    )
    
    args = parser.parse_args()
    analyze_results(args.tmp_dir)
