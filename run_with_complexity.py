"""
运行带复杂度追踪的多岛进化实验 - 修复日志显示问题
"""

import sys
import argparse
import logging

# 在导入其他模块之前，先配置日志级别
# 这样可以抑制第三方库（如httpx, openai）的INFO日志
logging.basicConfig(
    level=logging.WARNING,  # 只显示WARNING及以上级别
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 抑制特定库的日志
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

# LLM4AD imports
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI

# Local imports
from evaluation import FasPortRateEvaluation
from multiisland_with_complexity import MultiIslandEoH_WithComplexity

def main():
    parser = argparse.ArgumentParser(description='Multi-Island EoH - With Complexity Tracking')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (10 gens, 30 samples per gen)')
    parser.add_argument('--ultra-quick', action='store_true', help='Ultra-quick test mode (1 algorithm per island, 1 island only)')
    parser.add_argument('--api-key', type=str, default='sk-8251c8f00d16462f905dd30a4c218f52',
                        help='DeepSeek API key')
    parser.add_argument('--output-dir', type=str, default='logs/multiisland_complexity',
                        help='Output directory')
    parser.add_argument('--max-ofe', type=int, default=None,
                        help='Maximum OFE budget (optional)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show all logs including HTTP requests')
    args = parser.parse_args()

    # 如果用户指定了verbose，显示所有日志
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger('httpx').setLevel(logging.INFO)
        logging.getLogger('openai').setLevel(logging.INFO)

    # 配置参数
    if args.ultra_quick:
        max_gens = 1
        max_samples = 1
        pop_size = 1
        num_islands = 1
        print("Ultra-Quick Test Mode (With Complexity Tracking)")
        print("超极速模式: 只跑1个岛, 每个岛只评估1个算法")
        print("目的: 纯功能测试, 验证系统是否跑通")
    elif args.quick:
        max_gens = 10
        max_samples = 30
        pop_size = 3
        num_islands = 3
        print("Quick Test Mode (With Complexity Tracking)")
        print("快速模式: 10代, 每代30个样本, 种群大小3")
    else:
        max_gens = 50
        max_samples = 100
        pop_size = 3
        num_islands = 3
        print("Standard Mode (With Complexity Tracking)")

    print("=" * 80)
    print("Multi-Island Evolution of Heuristics - Complexity Tracking")
    print("=" * 80)
    print(f"Islands: {num_islands}")
    print(f"Max Generations: {max_gens}")
    print(f"Max Samples: {max_samples}")
    print(f"Pop Size (top_seed): {pop_size}")
    print(f"Total Evaluations: {num_islands} islands × {max_gens} gens × {max_samples} samples = {num_islands * max_gens * max_samples}")
    if args.max_ofe:
        print(f"Max OFE Budget: {args.max_ofe:,}")
        print(f"预计总运行时间: {args.max_ofe * 3 / 60:.1f} 分钟 (假设每次评估3分钟)")
    else:
        total_evals = num_islands * max_gens * max_samples
        print(f"预计总运行时间: {total_evals * 3 / 60:.1f} 分钟 (假设每次评估3分钟)")
        print(f"预计总运行时间: {total_evals * 1 / 60:.1f} 分钟 (假设每次评估1分钟)")
        print(f"预计总运行时间: {total_evals * 0.5 / 60:.1f} 分钟 (假设每次评估0.5分钟)")
    print("=" * 80)

    # 初始化LLM
    print("\n初始化LLM客户端 (Local vLLM - OpenAI Compatible)...")
    llm = OpenAIAPI(
        # 这里必须带 http:// 和 /v1，OpenAI 客户端会自动处理
        base_url='http://localhost:8000/v1', 
        
        # 本地随意填
        api_key='EMPTY',
        
        # 您的模型名
        model='xingyaoww/CodeActAgent-Mistral-7b-v0.1',
        
        timeout=600,
    )
    print("✓ LLM客户端初始化完成")

    # 初始化评估任务
    print("\n初始化评估任务...")
    evaluation = FasPortRateEvaluation(timeout_seconds=180)
    print("✓ 评估任务初始化完成 (单次评估超时: 180秒)")

    # 岛屿配置
    all_island_configs = [
        {'name': 'Exploit'},
        {'name': 'Explore'},
        {'name': 'Diversity'}
    ]
    island_configs = all_island_configs[:num_islands]  # 根据模式选择岛屿数量

    # EoH参数
    eoh_params = {
        'max_gens': max_gens,
        'max_samples': max_samples,
        'pop_size': pop_size,
        'num_samplers': 1,
        'num_evaluators': 1,
        'initial_sample_nums_max': 10  # 增加初始化采样次数,避免初始化失败
    }

    # 创建多岛进化实例
    print("\n创建多岛进化实例 (With Complexity Tracking)...")
    multiisland = MultiIslandEoH_WithComplexity(
        llm=llm,
        evaluation=evaluation,
        island_configs=island_configs,
        eoh_params=eoh_params,
        log_dir=args.output_dir,
        max_ofe_budget=args.max_ofe,
        track_complexity=True
    )
    print("✓ 多岛进化实例创建完成")

    # 运行实验
    print("\n" + "=" * 80)
    print("开始多岛进化实验 (With Complexity Tracking)...")
    print("=" * 80)
    
    # 刷新输出，确保上面的内容立即显示
    sys.stdout.flush()

    try:
        best_code, best_score = multiisland.run()

        print("\n" + "=" * 80)
        print("实验完成!")
        print("=" * 80)
        if best_score > float('-inf'):
            print(f"Global Best Score: {best_score:.6f}")
        else:
            print("Global Best Score: 未找到有效解")

        # 显示复杂度统计摘要
        print("\n" + "=" * 80)
        print("输出文件:")
        print("=" * 80)
        print(f"  {args.output_dir}/")
        print(f"    ├── island_0_Exploit/")
        print(f"    │   ├── algorithms/          ← 所有算法文件")
        print(f"    │   ├── algorithms_list.json")
        print(f"    │   ├── best_autoport.py")
        print(f"    │   └── best_summary.json")
        print(f"    ├── island_1_Explore/")
        print(f"    ├── island_2_Diversity/")
        print(f"    ├── final_results.json")
        print(f"    └── global_best_autoport.py")
        print(f"")
        print(f"  注意: OFE数据请查看 terminal 输出中的 [OFE=xxx] 值")

        print("\n所有数据已保存!")

        return best_code, best_score

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断实验")
        print(f"部分结果可能已保存到: {args.output_dir}")
        return None, float('-inf')

    except Exception as e:
        print(f"\n\n实验失败: {e}")
        import traceback
        traceback.print_exc()
        return None, float('-inf')

if __name__ == "__main__":
    main()
