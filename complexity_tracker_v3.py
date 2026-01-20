"""
复杂度追踪器
"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, Optional, List

# 全局OFE计数器
_global_ofe_lock = threading.Lock()
_global_ofe_counter = {'count': 0}


def get_global_ofe() -> int:
    """获取全局OFE计数"""
    with _global_ofe_lock:
        return _global_ofe_counter['count']


def increment_global_ofe() -> int:
    """增加全局OFE计数，返回增加后的值"""
    with _global_ofe_lock:
        _global_ofe_counter['count'] += 1
        return _global_ofe_counter['count']


def reset_global_ofe():
    """重置全局OFE计数（仅用于测试）"""
    with _global_ofe_lock:
        _global_ofe_counter['count'] = 0


# ComplexityTracker - 复杂度追踪器
class ComplexityTracker:
    """
    复杂度追踪器 - 追踪OFE(SINR调用)和LLM调用次数

    v3.0改进：
    - 使用全局计数器追踪OFE
    - 线程安全
    - 在多线程环境中可靠工作
    """

    def __init__(self, name: str = "Algorithm"):
        self.name = name
        self.tracker_id = id(self)  # 追踪器ID用于调试
        self.llm_calls = 0  # LLM调用次数
        self.all_scores = []  # 所有评估分数
        self.initial_score = None
        self.start_time = None
        self.end_time = None
        self.start_ofe = None  # 开始时的全局OFE
        self.last_evaluation_ofe = 0  # 最后一次评估的OFE
        self._accumulated_ofe = 0  # 累积OFE（用于最终统计）

        # 轨迹数据
        self.trajectory = []  # [(ofe, llm_calls, score, timestamp), ...]
        self.best_score_trajectory = []  # 最优分数轨迹 (兼容旧版本)

        # 显示设置
        self.display_interval = 100  # 每100次OFE显示一次进度

    def start(self):
        """开始追踪"""
        self.start_time = time.time()
        self.start_ofe = get_global_ofe()  # 记录开始时的全局OFE

    @property
    def ofe_count(self) -> int:
        """获取当前OFE计数（从全局计数器）"""
        if self.start_ofe is None:
            return 0
        return get_global_ofe() - self.start_ofe

    def record_sinr_call(self):
        """
        记录一次SINR调用 (这是真正的OFE)

        注意：v3.0中这个方法只是为了兼容性保留
        实际的OFE计数由全局计数器管理
        """
        increment_global_ofe()

        # 实时显示
        if self.ofe_count % self.display_interval == 0:
            self.print_progress()

    def record_llm_call(self):
        """记录一次LLM调用"""
        self.llm_calls += 1

    def record_evaluation(self, score: float, verbose: bool = True):
        """记录一次算法评估的分数"""
        self.all_scores.append(score)

        # 设置初始分数
        if self.initial_score is None and score > float('-inf'):
            self.initial_score = score

        # 记录轨迹
        self.trajectory.append({
            'ofe': self.ofe_count,
            'llm_calls': self.llm_calls,
            'score': score,
            'timestamp': time.time() - self.start_time if self.start_time else 0
        })

        # 更新最优分数轨迹 (兼容旧版本)
        current_best = self.get_final_score()
        self.best_score_trajectory.append({
            'ofe': self.ofe_count,
            'llm_calls': self.llm_calls,
            'best_score': current_best,
            'timestamp': time.time() - self.start_time if self.start_time else 0
        })

        if verbose:
            self.print_progress()

    def print_progress(self):
        """打印当前进度"""
        best_score = self.get_final_score()
        print(f"[OFE={self.ofe_count:6d}] Best: {best_score:.6f}, LLM: {self.llm_calls}")

    def end(self):
        """结束追踪"""
        self.end_time = time.time()

    def get_final_score(self) -> float:
        """获取最终最优分数"""
        if not self.all_scores:
            return float('-inf')
        valid_scores = [s for s in self.all_scores if s > float('-inf')]
        if not valid_scores:
            return float('-inf')
        return max(valid_scores)

    def get_improvement(self) -> float:
        """获取性能提升"""
        if self.initial_score is None:
            return 0.0
        return self.get_final_score() - self.initial_score

    def get_efficiency(self) -> float:
        """获取效率 (提升/OFE)"""
        if self.ofe_count == 0:
            return 0.0
        improvement = self.get_improvement()
        return improvement / self.ofe_count

    def get_duration(self) -> Optional[float]:
        """获取运行时长(秒)"""
        if self.start_time is None:
            return None
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    def get_success_rate(self) -> float:
        """获取成功率"""
        if not self.all_scores:
            return 0.0
        valid = [s for s in self.all_scores if s > float('-inf')]
        return len(valid) / len(self.all_scores)

    def to_dict(self) -> Dict:
        """导出为字典"""
        return {
            'metadata': {
                'name': self.name,
                'ofe_count': self.ofe_count,  # 使用property获取
                'llm_calls': self.llm_calls,
                'duration': self.get_duration(),
                'timestamp': time.time()
            },
            'performance': {
                'initial_score': self.initial_score,
                'final_score': self.get_final_score(),
                'best_score': self.get_final_score(),
                'improvement': self.get_improvement(),
                'all_scores': self.all_scores
            },
            'efficiency': {
                'efficiency': self.get_efficiency(),
                'success_rate': self.get_success_rate(),
                'avg_score': sum([s for s in self.all_scores if s > float('-inf')]) / len([s for s in self.all_scores if s > float('-inf')]) if [s for s in self.all_scores if s > float('-inf')] else 0.0
            },
            'trajectory': self.trajectory
        }

    def save_to_file(self, filepath: Path):
        """保存到JSON文件"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_trajectory_csv(self, filepath: Path):
        """保存轨迹数据到CSV"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write('ofe,llm_calls,score,timestamp\n')
            for point in self.trajectory:
                f.write(f"{point['ofe']},{point['llm_calls']},{point['score']},{point['timestamp']}\n")

    def print_summary(self):
        """打印统计摘要 (兼容旧版本接口)"""
        print("\n" + "=" * 80)
        print(f" 复杂度统计摘要 - {self.name}")
        print("=" * 80)

        stats = self.to_dict()

        print(f"\n元数据:")
        print(f"  总LLM调用: {stats['metadata']['llm_calls']}")
        print(f"  运行时长: {stats['metadata']['duration']:.1f}秒")

        print(f"\n性能:")
        print(f"  初始分数: {stats['performance']['initial_score']:.6f}" if stats['performance']['initial_score'] else "  初始分数: N/A")
        print(f"  最终分数: {stats['performance']['final_score']:.6f}")
        print(f"  性能提升: {stats['performance']['improvement']:.6f}")

        print(f"\n注意: OFE数据请查看 terminal 输出中的 [OFE=xxx] 值")

        print("=" * 80)

    # 兼容旧版本的方法别名
    def save_json(self, filepath):
        """保存为JSON (兼容旧版本接口)"""
        self.save_to_file(Path(filepath))

    def save_csv(self, filepath):
        """保存轨迹数据为CSV (兼容旧版本接口)"""
        self.save_trajectory_csv(Path(filepath))


# TrackedSINRFunction - 包装SINR函数以追踪调用次数
class TrackedSINRFunction:
    """
    包装SINR函数以追踪调用次数

    v3.0改进：
    - 使用全局计数器
    - 线程安全
    - 不依赖tracker对象
    """

    def __init__(self, sinr_func, tracker: ComplexityTracker):
        self.sinr_func = sinr_func
        self.tracker = tracker  # 保留用于兼容性，但不依赖它

    def __call__(self, *args, **kwargs):
        """调用SINR函数并记录"""
        # 增加全局计数器
        increment_global_ofe()

        # 调用原始函数
        result = self.sinr_func(*args, **kwargs)

        return result


# TrackedLLM - 追踪LLM调用次数的包装器
class TrackedLLM:
    """追踪LLM调用次数的包装器"""

    def __init__(self, llm, tracker: ComplexityTracker):
        self.llm = llm
        self.tracker = tracker

    def __call__(self, *args, **kwargs):
        """调用LLM并记录"""
        self.tracker.record_llm_call()
        return self.llm(*args, **kwargs)

    def draw_sample(self, *args, **kwargs):
        """拦戮draw_sample方法（EoH使用的主要方法）"""
        self.tracker.record_llm_call()
        return self.llm.draw_sample(*args, **kwargs)

    def __getattr__(self, name):
        """转发所有其他属性访问到原始LLM"""
        return getattr(self.llm, name)


# TrackedEvaluation - 追踪评估的包装器
class TrackedEvaluation:
    """
    追踪评估的包装器 - v3.0版本

    注意：这个类只记录最终的评估分数，不记录OFE
    OFE由全局计数器和TrackedSINRFunction记录
    """

    def __init__(self, evaluation, tracker: ComplexityTracker, tracked_sinr_func):
        self.evaluation = evaluation
        self.tracker = tracker
        self.tracked_sinr_func = tracked_sinr_func

        # 替换evaluation中的SINR函数
        self._inject_tracked_sinr()

    def _inject_tracked_sinr(self):
        """
        将tracked SINR函数注入到evaluation模块中
        这样算法内部和评估时的SINR调用都会被追踪
        """
        # 导入utility_objective_functions模块
        import utility_objective_functions

        # 替换模块中的函数
        utility_objective_functions.sinr_balancing_power_constraint = self.tracked_sinr_func

        # 替换evaluation对象可能缓存的引用
        if hasattr(self.evaluation, 'sinr_balancing_power_constraint'):
            self.evaluation.sinr_balancing_power_constraint = self.tracked_sinr_func

        # 关键修复：替换evaluation.py模块中的SINR引用
        # evaluation.py在模块级别 import 了 sinr_balancing_power_constraint
        # 我们需要找到这个模块并替换它的引用
        import sys
        for module_name, module in sys.modules.items():
            if module and hasattr(module, 'sinr_balancing_power_constraint'):
                # 检查是否是原始的SINR函数
                if module.sinr_balancing_power_constraint != self.tracked_sinr_func:
                    module.sinr_balancing_power_constraint = self.tracked_sinr_func

    def evaluate_program(self, program_str, program_callable=None, **kwargs):
        """评估程序并记录分数"""
        # 评估前记录OFE
        ofe_before = self.tracker.ofe_count

        # 评估（SINR调用会袾TrackedSINRFunction自动记录到全局计数器）
        result = self.evaluation.evaluate_program(program_str, program_callable, **kwargs)

        # 评估后记录OFE
        ofe_after = self.tracker.ofe_count

        # 只打印OFE，不做追踪（用户会从 terminal 输出中手动查看）

        # 只记录最终分数
        self.tracker.record_evaluation(result, verbose=False)

        # 显示OFE和LLM calls计数
        print(f"[OFE={ofe_after:4d}, LLM calls={self.tracker.llm_calls:3d}] {result:.6f}")

        return result

    def __getattr__(self, name):
        """转发所有其他属性访问到原始evaluation"""
        return getattr(self.evaluation, name)


# BudgetLimitedEvaluation - 带预算限制的评估包装器
class BudgetExceededException(Exception):
    """预算超出异常"""
    pass


class BudgetLimitedEvaluation(TrackedEvaluation):
    """
    带预算限制的评估包装器 - v3.0版本
    """

    def __init__(self, evaluation, max_ofe: int, tracker: ComplexityTracker, tracked_sinr_func):
        self.max_ofe = max_ofe
        super().__init__(evaluation, tracker, tracked_sinr_func)

    def evaluate_program(self, program_str, program_callable=None, **kwargs):
        """评估程序并检查预算"""
        # 检查预算
        if self.tracker.ofe_count >= self.max_ofe:
            raise BudgetExceededException(f"OFE budget exceeded: {self.tracker.ofe_count} >= {self.max_ofe}")

        # 调用父类方法
        return super().evaluate_program(program_str, program_callable, **kwargs)
