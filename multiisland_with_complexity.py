"""
多岛进化算法 - 带复杂度追踪版本
追踪OFE,支持预算限制
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# LLM4AD imports
from llm4ad.method.eoh import EoH

# Local imports
from complexity_tracker_v3 import (
    ComplexityTracker,
    TrackedLLM,
    TrackedEvaluation,
    TrackedSINRFunction,
    BudgetLimitedEvaluation,
    BudgetExceededException
)


class MultiIslandEoH_WithComplexity:
    """
    多岛进化算法 - 带复杂度追踪

    特性:
    1. 追踪OFE和LLM calls
    2. 记录性能轨迹
    3. 支持预算限制
    4. 保存所有算法到单独文件
    5. 生成复杂度统计报告
    """

    def __init__(
        self,
        llm,
        evaluation,
        island_configs: List[Dict],
        eoh_params: Dict,
        log_dir: str = "logs/multiisland_complexity",
        max_ofe_budget: Optional[int] = None,  # 可选的预算限制
        track_complexity: bool = True  # 是否追踪复杂度
    ):
        self.llm = llm
        self.evaluation = evaluation
        self.island_configs = island_configs
        self.eoh_params = eoh_params
        self.log_dir = Path(log_dir)
        self.max_ofe_budget = max_ofe_budget
        self.track_complexity = track_complexity

        # 创建输出目录
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 全局最优
        self.global_best_code = None
        self.global_best_score = float('-inf')

        # 复杂度追踪器列表
        self.trackers = []

    def _generate_hash(self, code: str) -> str:
        """生成代码的16位哈希"""
        return hashlib.md5(code.encode()).hexdigest()[:16]

    def _generate_filename(self, generation: int, code: str, score: float, ofe: int = None) -> str:
        """生成文件名: gen{generation:03d}_ofe{ofe:04d}_{hash}_score{score}.py"""
        code_hash = self._generate_hash(code)
        if score == float('-inf') or score != score:  # 处理-inf, inf, nan
            score_str = "score-inf"
        else:
            score_str = f"score{score:.6f}"

        # 添加OFE字段
        if ofe is not None:
            return f"gen{generation:03d}_ofe{ofe:04d}_{code_hash}_{score_str}.py"
        else:
            return f"gen{generation:03d}_{code_hash}_{score_str}.py"

    def _run_single_island(
        self,
        island_id: int,
        island_config: Dict
    ) -> Tuple[str, float, ComplexityTracker]:
        """
        运行单个岛屿的进化
        返回: (best_code, best_score, tracker)
        """
        island_name = island_config['name']

        print(f"\n{'='*80}")
        print(f" Island {island_id}: {island_name}")
        print(f"{'='*80}")

        # 创建岛屿目录
        island_dir = self.log_dir / f"island_{island_id}_{island_name}"
        island_dir.mkdir(parents=True, exist_ok=True)

        # 创建algorithms目录
        algorithms_dir = island_dir / "algorithms"
        algorithms_dir.mkdir(parents=True, exist_ok=True)

        # 重置全局OFE计数器（每个岛屿从0开始）
        from complexity_tracker_v3 import reset_global_ofe
        reset_global_ofe()

        # 创建复杂度追踪器
        tracker = ComplexityTracker(name=island_name)
        tracker.start()

        # 包装SINR函数 (v2.0核心改进)
        from utility_objective_functions import sinr_balancing_power_constraint
        tracked_sinr = TrackedSINRFunction(sinr_balancing_power_constraint, tracker)

        # 包装LLM和evaluation
        tracked_llm = TrackedLLM(self.llm, tracker)

        if self.max_ofe_budget:
            # 有预算限制
            tracked_eval = BudgetLimitedEvaluation(self.evaluation, self.max_ofe_budget, tracker, tracked_sinr)
            print(f"  预算限制: {self.max_ofe_budget} OFE")
        else:
            # 无预算限制
            tracked_eval = TrackedEvaluation(self.evaluation, tracker, tracked_sinr)
            print(f"  无预算限制")

        # 算法列表
        algorithms_list = []
        evaluation_counter = [0]
        generation_counter = [0]
        max_samples = self.eoh_params['max_samples']

        # 包装evaluation以保存每个算法
        original_evaluate = tracked_eval.evaluate_program

        def wrapped_evaluate(program_str, program_callable=None, **kwargs):
            """包装的评估函数,保存每个算法到单独文件"""
            # 评估
            result = original_evaluate(program_str, program_callable, **kwargs)

            # 更新计数器
            evaluation_counter[0] += 1
            generation_counter[0] = (evaluation_counter[0] - 1) // max_samples

            # 获取当前OFE计数（使用评估时保存的OFE）
            current_ofe = getattr(tracker, 'last_evaluation_ofe', tracker.ofe_count)

            # 生成文件名(包含OFE)
            filename = self._generate_filename(generation_counter[0], program_str, result, ofe=current_ofe)
            filepath = algorithms_dir / filename

            # 保存算法到文件
            with open(filepath, 'w') as f:
                f.write(program_str)

            # 记录到列表
            algorithms_list.append({
                'filename': filename,
                'generation': generation_counter[0],
                'evaluation_id': evaluation_counter[0],
                'ofe': current_ofe,
                'score': result,
                'code_hash': self._generate_hash(program_str),
                'timestamp': time.time()
            })

            # 简洁输出
            if result > float('-inf'):
                print(f"  ✓ Gen{generation_counter[0]:03d} Eval{evaluation_counter[0]:03d}: {result:.6f}")
            else:
                print(f"  ✗ Gen{generation_counter[0]:03d} Eval{evaluation_counter[0]:03d}: FAILED")

            return result

        # 临时替换evaluation方法
        tracked_eval.evaluate_program = wrapped_evaluate

        # 创建EoH实例
        try:
            print(f"\n▶️  开始进化...")

            island_eoh = EoH(
                llm=tracked_llm,
                evaluation=tracked_eval,
                **self.eoh_params
            )

            # 运行进化
            result = island_eoh.run()

            # 结束追踪
            tracker.end()

            print(f"\n✓ 进化完成,用时 {tracker.get_duration():.1f}s")

            # 提取最优解
            best_code = None
            best_score = float('-inf')

            if hasattr(island_eoh, '_population') and island_eoh._population:
                try:
                    pop_list = list(island_eoh._population)
                    if pop_list:
                        print(f"  ✓ 从 _population 中提取到 {len(pop_list)} 个候选")

                        for prog in pop_list:
                            if hasattr(prog, 'score') and hasattr(prog, 'body'):
                                if prog.score > best_score:
                                    best_score = prog.score
                                    best_code = prog.body
                except Exception as e:
                    print(f"  ⚠️  提取_population失败: {e}")

            # 如果没有从population提取到,使用tracker的最优分数
            if best_score == float('-inf'):
                best_score = tracker.get_final_score()
                print(f"  ⚠️  使用tracker的最优分数: {best_score:.6f}")

            # 复杂度数据不保存，用户从 terminal 输出中手动统计

            # 保存算法列表
            with open(island_dir / "algorithms_list.json", 'w') as f:
                json.dump({
                    'metadata': {
                        'total_algorithms': len(algorithms_list),
                        'best_score': best_score,
                        'timestamp': time.time()
                    },
                    'algorithms': algorithms_list
                }, f, indent=2)

            print(f"  ✓ 算法列表已保存: algorithms_list.json ({len(algorithms_list)} 个算法)")

            # 保存最优算法
            if best_code:
                best_file = island_dir / "best_autoport.py"
                with open(best_file, 'w') as f:
                    f.write(best_code)
                print(f"  ✓ 最优算法已保存: best_autoport.py")

                # 保存最优算法摘要
                with open(island_dir / "best_summary.json", 'w') as f:
                    json.dump({
                        'score': best_score,
                        'code_hash': self._generate_hash(best_code),
                        'code_length': len(best_code),
                        'timestamp': time.time()
                    }, f, indent=2)

            return best_code, best_score, tracker

        except BudgetExceededException as e:
            # 预算超出
            tracker.end()
            print(f"  预算超出: {e}")
            print(f"  Best Score: {tracker.get_final_score():.6f}")

            # 复杂度数据不保存，用户从 terminal 输出中手动统计

            return None, tracker.get_final_score(), tracker

        except Exception as e:
            # 其他错误
            tracker.end()
            print(f"\n❌ [{island_name}] Failed: {e}")
            import traceback
            traceback.print_exc()

            # 复杂度数据不保存，用户从 terminal 输出中手动统计

            return None, float('-inf'), tracker

    def run(self) -> Tuple[str, float]:
        """
        运行多岛进化实验
        返回: (global_best_code, global_best_score)
        """
        print(f"\n{'='*80}")
        print(f"Multi-Island EoH - With Complexity Tracking")
        print(f"{'='*80}")
        print(f" 追踪OFE和LLM calls")
        print(f" 记录性能轨迹")
        if self.max_ofe_budget:
            print(f" 预算限制: {self.max_ofe_budget} OFE")
        print(f"{'='*80}")

        print(f"顺序运行 {len(self.island_configs)} 个岛屿...")

        # 运行每个岛屿
        island_results = []

        for island_id, island_config in enumerate(self.island_configs):
            print(f"\n{'='*80}")
            print(f"进度: {island_id + 1}/{len(self.island_configs)}")
            print(f"{'='*80}")

            best_code, best_score, tracker = self._run_single_island(island_id, island_config)

            island_results.append({
                'island_id': island_id,
                'island_name': island_config['name'],
                'best_score': best_score,
                'ofe_count': tracker.ofe_count,
                'llm_calls': tracker.llm_calls,
                'efficiency': tracker.get_efficiency(),
                'duration': tracker.get_duration()
            })

            # 更新全局最优
            if best_score > self.global_best_score:
                self.global_best_score = best_score
                self.global_best_code = best_code

            # 保存tracker
            self.trackers.append(tracker)

        # 保存最终结果
        self._save_final_results(island_results)

        return self.global_best_code, self.global_best_score

    def _save_final_results(self, island_results: List[Dict]):
        """保存最终结果"""
        print(f"\n{'='*80}")
        print(f" 最终结果")
        print(f"{'='*80}")

        # 保存全局最优算法
        if self.global_best_code:
            global_best_file = self.log_dir / "global_best_autoport.py"
            with open(global_best_file, 'w') as f:
                f.write(self.global_best_code)
            print(f"✓ 全局最优算法已保存: {global_best_file}")

        # 保存最终结果JSON
        final_results = {
            'metadata': {
                'num_islands': len(self.island_configs),
                'max_ofe_budget': self.max_ofe_budget,
                'timestamp': time.time()
            },
            'global_best': {
                'score': self.global_best_score,
                'code_hash': self._generate_hash(self.global_best_code) if self.global_best_code else None
            },
            'islands': island_results
        }

        with open(self.log_dir / "final_results.json", 'w') as f:
            json.dump(final_results, f, indent=2)

        print(f"✓ 最终结果已保存: final_results.json")

        # 合并所有轨迹数据
        self._merge_trajectory_data()

    def _merge_trajectory_data(self):
        """合并所有岛屿的轨迹数据到一个CSV"""
        # 不生成 all_trajectories.csv，用户从 terminal 输出中手动统计
        pass
