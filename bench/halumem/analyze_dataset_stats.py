"""
HaluMem Dataset Statistics Analyzer

统计 HaluMem 数据集的各项指标：
- 每个用户的 session 数量
- 每个 session 的对话数量
- 每个 session 的对话总长度

Usage:
    python bench/halumem/analyze_dataset_stats.py --data_path /path/to/HaluMem-Medium.jsonl
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class UserStats:
    """单个用户的统计数据"""
    user_name: str
    uuid: str
    num_sessions: int
    dialogues_per_session: list[int]  # 每个 session 的对话数量
    dialogue_lengths_per_session: list[int]  # 每个 session 的对话总长度（字符数）
    num_chunks_after_split: int  # 按 5000 字符分割后的 chunk 数量


@dataclass
class DatasetStats:
    """整体数据集统计"""
    total_users: int
    total_sessions: int
    total_dialogues: int
    
    avg_sessions_per_user: float
    avg_dialogues_per_session: float
    avg_dialogue_length_per_session: float
    
    # 详细分布
    sessions_per_user_list: list[int]
    dialogues_per_session_list: list[int]
    dialogue_lengths_per_session_list: list[int]
    
    # Content 统计
    total_contents: int  # 所有对话回合的 content 总数
    content_sizes: list[int]  # 每个 content 的大小（字符数）
    min_content_size: int
    max_content_size: int
    percentiles: dict[str, float]  # 分位点统计（全部）
    
    # 按 role 分类的 Content 统计
    total_user_contents: int
    total_assistant_contents: int
    user_percentiles: dict[str, float]  # user 角色的分位点
    assistant_percentiles: dict[str, float]  # assistant 角色的分位点
    
    # Session 分割统计
    total_chunks_after_split: int  # 按 5000 字符分割后的总 chunk 数
    chunks_per_user_list: list[int]  # 每个用户分割后的 chunk 数量
    avg_chunks_per_user: float  # 平均每个用户的 chunk 数量


class DatasetAnalyzer:
    """数据集分析器"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.user_stats_list: list[UserStats] = []
        self.all_content_sizes: list[int] = []  # 收集所有 content 的大小
        self.user_content_sizes: list[int] = []  # user 角色的 content 大小
        self.assistant_content_sizes: list[int] = []  # assistant 角色的 content 大小
    
    @staticmethod
    def extract_user_name(persona_info: str) -> str:
        """从 persona_info 中提取用户名"""
        match = re.search(r"Name:\s*(.*?);", persona_info)
        if not match:
            return "Unknown"
        return match.group(1).strip()
    
    @staticmethod
    def calculate_dialogue_length(dialogue: list[dict]) -> int:
        """计算对话的总长度（字符数）"""
        total_length = 0
        for turn in dialogue:
            content = turn.get("content", "")
            total_length += len(content)
        return total_length
    
    @staticmethod
    def split_session_into_chunks(dialogue: list[dict], max_length: int = 5000) -> int:
        """
        将一个 session 按照 max_length 分割成多个 chunks。
        规则：
        1. 每次添加 2 个对话回合（user-assistant 对）
        2. 如果添加后超过 max_length，就开始新的 chunk
        3. 但是每个 chunk 至少包含 2 个对话回合
        
        返回分割后的 chunk 数量
        """
        if not dialogue:
            return 0
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        # 每次处理 2 个对话回合
        i = 0
        while i < len(dialogue):
            # 取 2 个对话回合（如果不足 2 个，取剩余的）
            pair = dialogue[i:i+2]
            pair_length = sum(len(turn.get("content", "")) for turn in pair)
            
            # 如果当前 chunk 为空，直接添加（保证至少 2 个）
            if not current_chunk:
                current_chunk.extend(pair)
                current_length += pair_length
                i += len(pair)
            else:
                # 如果添加这一对后会超过限制
                if current_length + pair_length > max_length:
                    # 保存当前 chunk，开始新的 chunk
                    chunks.append(current_chunk)
                    current_chunk = pair
                    current_length = pair_length
                    i += len(pair)
                else:
                    # 否则添加到当前 chunk
                    current_chunk.extend(pair)
                    current_length += pair_length
                    i += len(pair)
        
        # 添加最后一个 chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return len(chunks)
    
    def load_and_analyze(self):
        """加载并分析数据集"""
        logger.info(f"Loading data from: {self.data_path}")
        
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    user_data = json.loads(line)
                    self._analyze_user(user_data)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing line {line_num}: {e}")
                    continue
        
        logger.info(f"Analyzed {len(self.user_stats_list)} users")
    
    def _analyze_user(self, user_data: dict):
        """分析单个用户的数据"""
        user_name = self.extract_user_name(user_data.get("persona_info", ""))
        uuid = user_data.get("uuid", "")
        sessions = user_data.get("sessions", [])
        
        dialogues_per_session = []
        dialogue_lengths_per_session = []
        total_chunks = 0
        
        for session in sessions:
            dialogue = session.get("dialogue", [])
            num_dialogues = len(dialogue)
            dialogue_length = self.calculate_dialogue_length(dialogue)
            
            dialogues_per_session.append(num_dialogues)
            dialogue_lengths_per_session.append(dialogue_length)
            
            # 计算这个 session 分割后的 chunk 数量
            num_chunks = self.split_session_into_chunks(dialogue, max_length=5000)
            total_chunks += num_chunks
            
            # 收集每个 content 的大小，并按 role 分类
            for turn in dialogue:
                content = turn.get("content", "")
                content_size = len(content)
                role = turn.get("role", "")
                
                self.all_content_sizes.append(content_size)
                
                if role == "user":
                    self.user_content_sizes.append(content_size)
                elif role == "assistant":
                    self.assistant_content_sizes.append(content_size)
        
        user_stats = UserStats(
            user_name=user_name,
            uuid=uuid,
            num_sessions=len(sessions),
            dialogues_per_session=dialogues_per_session,
            dialogue_lengths_per_session=dialogue_lengths_per_session,
            num_chunks_after_split=total_chunks
        )
        
        self.user_stats_list.append(user_stats)
    
    def compute_dataset_stats(self) -> DatasetStats:
        """计算整体数据集统计"""
        total_users = len(self.user_stats_list)
        
        sessions_per_user_list = [u.num_sessions for u in self.user_stats_list]
        total_sessions = sum(sessions_per_user_list)
        
        dialogues_per_session_list = []
        dialogue_lengths_per_session_list = []
        
        for user in self.user_stats_list:
            dialogues_per_session_list.extend(user.dialogues_per_session)
            dialogue_lengths_per_session_list.extend(user.dialogue_lengths_per_session)
        
        total_dialogues = sum(dialogues_per_session_list)
        
        # 计算平均值
        avg_sessions_per_user = total_sessions / total_users if total_users > 0 else 0
        avg_dialogues_per_session = (
            total_dialogues / total_sessions if total_sessions > 0 else 0
        )
        avg_dialogue_length_per_session = (
            sum(dialogue_lengths_per_session_list) / len(dialogue_lengths_per_session_list)
            if dialogue_lengths_per_session_list else 0
        )
        
        # Content 统计
        total_contents = len(self.all_content_sizes)
        min_content_size = min(self.all_content_sizes) if self.all_content_sizes else 0
        max_content_size = max(self.all_content_sizes) if self.all_content_sizes else 0
        
        # 计算分位点 (10%, 15%, 20%, ..., 95%)
        percentile_points = list(range(10, 100, 5))  # 10, 15, 20, ..., 95
        
        # 全部 content 的分位点
        percentiles = {}
        if self.all_content_sizes:
            content_array = np.array(self.all_content_sizes)
            for p in percentile_points:
                percentiles[f"p{p}"] = float(np.percentile(content_array, p))
        
        # user 角色的分位点
        user_percentiles = {}
        if self.user_content_sizes:
            user_array = np.array(self.user_content_sizes)
            for p in percentile_points:
                user_percentiles[f"p{p}"] = float(np.percentile(user_array, p))
        
        # assistant 角色的分位点
        assistant_percentiles = {}
        if self.assistant_content_sizes:
            assistant_array = np.array(self.assistant_content_sizes)
            for p in percentile_points:
                assistant_percentiles[f"p{p}"] = float(np.percentile(assistant_array, p))
        
        # Session 分割统计
        chunks_per_user_list = [u.num_chunks_after_split for u in self.user_stats_list]
        total_chunks_after_split = sum(chunks_per_user_list)
        avg_chunks_per_user = (
            total_chunks_after_split / total_users if total_users > 0 else 0
        )
        
        return DatasetStats(
            total_users=total_users,
            total_sessions=total_sessions,
            total_dialogues=total_dialogues,
            avg_sessions_per_user=avg_sessions_per_user,
            avg_dialogues_per_session=avg_dialogues_per_session,
            avg_dialogue_length_per_session=avg_dialogue_length_per_session,
            sessions_per_user_list=sessions_per_user_list,
            dialogues_per_session_list=dialogues_per_session_list,
            dialogue_lengths_per_session_list=dialogue_lengths_per_session_list,
            total_contents=total_contents,
            content_sizes=self.all_content_sizes,
            min_content_size=min_content_size,
            max_content_size=max_content_size,
            percentiles=percentiles,
            total_user_contents=len(self.user_content_sizes),
            total_assistant_contents=len(self.assistant_content_sizes),
            user_percentiles=user_percentiles,
            assistant_percentiles=assistant_percentiles,
            total_chunks_after_split=total_chunks_after_split,
            chunks_per_user_list=chunks_per_user_list,
            avg_chunks_per_user=avg_chunks_per_user
        )
    
    @staticmethod
    def _print_percentiles(percentiles: dict[str, float]):
        """打印分位点统计（辅助函数）"""
        if not percentiles:
            print("    (无数据)")
            return
        
        sorted_percentiles = sorted(percentiles.keys(), key=lambda x: int(x[1:]))
        
        # 每行显示 5 个分位点，让输出更紧凑
        for i in range(0, len(sorted_percentiles), 5):
            line_items = []
            for percentile_key in sorted_percentiles[i:i+5]:
                percentile_value = percentiles[percentile_key]
                p_num = percentile_key[1:]  # 去掉 'p' 前缀
                line_items.append(f"{p_num}%: {percentile_value:.0f}")
            print(f"    {' | '.join(line_items)}")
    
    def print_summary(self, stats: DatasetStats):
        """打印统计摘要"""
        print("\n" + "=" * 80)
        print("HALUMEM DATASET STATISTICS")
        print("=" * 80 + "\n")
        
        print("📊 总体统计:")
        print(f"  总用户数:              {stats.total_users}")
        print(f"  总 Session 数:         {stats.total_sessions}")
        print(f"  总对话数:              {stats.total_dialogues}")
        
        print(f"\n📈 平均值:")
        print(f"  每个用户的平均 Session 数:           {stats.avg_sessions_per_user:.2f}")
        print(f"  每个 Session 的平均对话数:           {stats.avg_dialogues_per_session:.2f}")
        print(f"  每个 Session 的平均对话长度（字符）: {stats.avg_dialogue_length_per_session:.2f}")
        
        print(f"\n📊 分布统计:")
        if stats.sessions_per_user_list:
            print(f"  每用户 Session 数 - 最小: {min(stats.sessions_per_user_list)}, "
                  f"最大: {max(stats.sessions_per_user_list)}")
        
        if stats.dialogues_per_session_list:
            print(f"  每 Session 对话数 - 最小: {min(stats.dialogues_per_session_list)}, "
                  f"最大: {max(stats.dialogues_per_session_list)}")
        
        if stats.dialogue_lengths_per_session_list:
            print(f"  每 Session 对话长度 - 最小: {min(stats.dialogue_lengths_per_session_list)}, "
                  f"最大: {max(stats.dialogue_lengths_per_session_list)}")
        
        print(f"\n💬 Content 详细统计:")
        print(f"  总 Content 数量:       {stats.total_contents}")
        print(f"    User 消息数:         {stats.total_user_contents}")
        print(f"    Assistant 消息数:    {stats.total_assistant_contents}")
        print(f"  Content 大小（字符数）:")
        print(f"    最小值:              {stats.min_content_size}")
        print(f"    最大值:              {stats.max_content_size}")
        
        if stats.content_sizes:
            avg_content_size = sum(stats.content_sizes) / len(stats.content_sizes)
            print(f"    平均值:              {avg_content_size:.2f}")
        
        print(f"\n📈 Content 大小分位点 (全部):")
        self._print_percentiles(stats.percentiles)
        
        print(f"\n📈 Content 大小分位点 (User 角色):")
        self._print_percentiles(stats.user_percentiles)
        
        print(f"\n📈 Content 大小分位点 (Assistant 角色):")
        self._print_percentiles(stats.assistant_percentiles)
        
        print(f"\n✂️  Session 分割统计 (按 5000 字符分割):")
        print(f"  原始 Session 总数:      {stats.total_sessions}")
        print(f"  分割后 Chunk 总数:      {stats.total_chunks_after_split}")
        print(f"  每个用户平均 Chunk 数:  {stats.avg_chunks_per_user:.2f}")
        print(f"  Chunk/Session 比例:     {stats.total_chunks_after_split / stats.total_sessions:.2f}x")
        
        print("\n" + "=" * 80)
    
    def print_per_user_stats(self):
        """打印每个用户的详细统计"""
        print("\n" + "=" * 80)
        print("PER-USER STATISTICS")
        print("=" * 80 + "\n")
        
        for idx, user_stats in enumerate(self.user_stats_list, 1):
            avg_dialogues = (
                sum(user_stats.dialogues_per_session) / len(user_stats.dialogues_per_session)
                if user_stats.dialogues_per_session else 0
            )
            avg_length = (
                sum(user_stats.dialogue_lengths_per_session) / len(user_stats.dialogue_lengths_per_session)
                if user_stats.dialogue_lengths_per_session else 0
            )
            
            print(f"[{idx}] {user_stats.user_name} (UUID: {user_stats.uuid[:8]}...)")
            print(f"    Session 数: {user_stats.num_sessions}")
            print(f"    分割后 Chunk 数: {user_stats.num_chunks_after_split}")
            print(f"    平均每 Session 对话数: {avg_dialogues:.2f}")
            print(f"    平均每 Session 对话长度: {avg_length:.2f} 字符")
            print()
    
    def print_user_split_summary(self):
        """打印每个用户的分割统计摘要（表格形式）"""
        print("\n" + "=" * 80)
        print("PER-USER SESSION SPLIT SUMMARY (按 5000 字符分割)")
        print("=" * 80 + "\n")
        
        # 表头
        print(f"{'序号':<6} {'用户名':<25} {'原始Sessions':<15} {'分割后Chunks':<15} {'比例':<10}")
        print("-" * 80)
        
        # 每个用户的数据
        for idx, user_stats in enumerate(self.user_stats_list, 1):
            ratio = (
                user_stats.num_chunks_after_split / user_stats.num_sessions
                if user_stats.num_sessions > 0 else 0
            )
            print(f"{idx:<6} {user_stats.user_name[:24]:<25} {user_stats.num_sessions:<15} "
                  f"{user_stats.num_chunks_after_split:<15} {ratio:.2f}x")
        
        print("-" * 80)
        
        # 总计
        total_sessions = sum(u.num_sessions for u in self.user_stats_list)
        total_chunks = sum(u.num_chunks_after_split for u in self.user_stats_list)
        overall_ratio = total_chunks / total_sessions if total_sessions > 0 else 0
        
        print(f"{'总计':<6} {'':<25} {total_sessions:<15} {total_chunks:<15} {overall_ratio:.2f}x")
        print("=" * 80)
    
    def save_results(self, output_path: str, stats: DatasetStats):
        """保存统计结果到 JSON 文件"""
        results = {
            "summary": {
                "total_users": stats.total_users,
                "total_sessions": stats.total_sessions,
                "total_dialogues": stats.total_dialogues,
                "avg_sessions_per_user": stats.avg_sessions_per_user,
                "avg_dialogues_per_session": stats.avg_dialogues_per_session,
                "avg_dialogue_length_per_session": stats.avg_dialogue_length_per_session,
                "session_split_stats": {
                    "total_chunks_after_split": stats.total_chunks_after_split,
                    "avg_chunks_per_user": stats.avg_chunks_per_user,
                    "chunk_to_session_ratio": (
                        stats.total_chunks_after_split / stats.total_sessions
                        if stats.total_sessions > 0 else 0
                    )
                },
                "content_stats": {
                    "total_contents": stats.total_contents,
                    "total_user_contents": stats.total_user_contents,
                    "total_assistant_contents": stats.total_assistant_contents,
                    "min_content_size": stats.min_content_size,
                    "max_content_size": stats.max_content_size,
                    "avg_content_size": (
                        sum(stats.content_sizes) / len(stats.content_sizes)
                        if stats.content_sizes else 0
                    ),
                    "percentiles_all": stats.percentiles,
                    "percentiles_user": stats.user_percentiles,
                    "percentiles_assistant": stats.assistant_percentiles
                }
            },
            "per_user_stats": [
                {
                    "user_name": u.user_name,
                    "uuid": u.uuid,
                    "num_sessions": u.num_sessions,
                    "num_chunks_after_split": u.num_chunks_after_split,
                    "avg_dialogues_per_session": (
                        sum(u.dialogues_per_session) / len(u.dialogues_per_session)
                        if u.dialogues_per_session else 0
                    ),
                    "avg_dialogue_length_per_session": (
                        sum(u.dialogue_lengths_per_session) / len(u.dialogue_lengths_per_session)
                        if u.dialogue_lengths_per_session else 0
                    ),
                    "dialogues_per_session": u.dialogues_per_session,
                    "dialogue_lengths_per_session": u.dialogue_lengths_per_session
                }
                for u in self.user_stats_list
            ]
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to: {output_path}")


def main(data_path: str, output_path: str = None, show_per_user: bool = False):
    """主函数"""
    # 检查文件是否存在
    if not Path(data_path).exists():
        logger.error(f"File not found: {data_path}")
        return
    
    # 创建分析器并执行分析
    analyzer = DatasetAnalyzer(data_path)
    analyzer.load_and_analyze()
    
    # 计算统计数据
    stats = analyzer.compute_dataset_stats()
    
    # 打印摘要
    analyzer.print_summary(stats)
    
    # 打印每个用户的分割统计摘要（始终显示）
    analyzer.print_user_split_summary()
    
    # 打印每个用户的详细统计（可选）
    if show_per_user:
        analyzer.print_per_user_stats()
    
    # 保存结果到文件
    if output_path:
        analyzer.save_results(output_path, stats)
    else:
        # 默认保存到与数据文件相同目录
        default_output = str(Path(data_path).parent / "dataset_statistics.json")
        analyzer.save_results(default_output, stats)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze HaluMem dataset statistics"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to HaluMem JSONL file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save statistics JSON (default: dataset_statistics.json in same dir)"
    )
    parser.add_argument(
        "--show_per_user",
        action="store_true",
        help="Show detailed statistics for each user"
    )
    
    args = parser.parse_args()
    
    main(
        data_path=args.data_path,
        output_path=args.output_path,
        show_per_user=args.show_per_user
    )
