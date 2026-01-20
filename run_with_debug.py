"""
带调试输出的运行脚本
"""

import sys
import argparse
import logging

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,  # 改为DEBUG级别
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# 禁用httpx的INFO日志
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

from llm4ad.tools.llm.llm_api_openai import OpenAIAPI
from evaluation import FasPortRateEvaluation
from multiisland_with_complexity import MultiIslandEoH_WithComplexity

def main():
    parser = argparse.ArgumentParser(description='Multi-Island EoH - Debug Mode')
    parser.add_argument('--ultra-quick', action='store_true', help='Ultra-quick test mode')
    args = parser.parse_args()

    print("=" * 80, flush=True)
    print("Debug Mode - Ultra Quick Test", flush=True)
    print("=" * 80, flush=True)

    # 最小配置
    max_gens = 1
    max_samples = 1
    pop_size = 1
    num_islands = 1

    print(f"\n配置:", flush=True)
    print(f"  Islands: {num_islands}", flush=True)
    print(f"  Generations: {max_gens}", flush=True)
    print(f"  Samples: {max_samples}", flush=True)
    print(f"  Pop Size: {pop_size}", flush=True)

    # 初始化LLM
    print("\n[1/4] 初始化LLM...", flush=True)
    try:
        llm = OpenAIAPI(
            base_url='http://localhost:8000/v1',
            api_key='EMPTY',
            model='xingyaoww/CodeActAgent-Mistral-7b-v0.1',
            timeout=180
        )
        print("✓ LLM初始化完成", flush=True)
    except Exception as e:
        print(f"✗ LLM初始化失败: {e}", flush=True)
        return

    # 测试LLM
    print("\n[2/4] 测试LLM响应...", flush=True)
    try:
        print("  发送测试请求...", flush=True)
        sys.stdout.flush()
        
        response = llm.query("Say 'Hello' and nothing else.", max_new_tokens=50)
        print(f"✓ LLM响应: {response}", flush=True)
    except Exception as e:
        print(f"✗ LLM测试失败: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return

    # 初始化评估
    print("\n[3/4] 初始化评估任务...", flush=True)
    try:
        evaluation = FasPortRateEvaluation(timeout_seconds=180)
        print("✓ 评估任务初始化完成", flush=True)
    except Exception as e:
        print(f"✗ 评估初始化失败: {e}", flush=True)
        return

    # 创建多岛实例
    print("\n[4/4] 创建多岛进化实例...", flush=True)
    try:
        island_configs = [{'name': 'Test'}]
        
        eoh_params = {
            'max_gens': max_gens,
            'max_samples': max_samples,
            'pop_size': pop_size,
            'num_samplers': 1,
            'num_evaluators': 1,
            'initial_sample_nums_max': 1  # 最小化初始化
        }

        multiisland = MultiIslandEoH_WithComplexity(
            llm=llm,
            evaluation=evaluation,
            island_configs=island_configs,
            eoh_params=eoh_params,
            log_dir='logs/debug_test',
            max_ofe_budget=None,
            track_complexity=True
        )
        print("✓ 多岛实例创建完成", flush=True)
    except Exception as e:
        print(f"✗ 多岛实例创建失败: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return

    # 运行实验
    print("\n" + "=" * 80, flush=True)
    print("开始运行实验...", flush=True)
    print("=" * 80, flush=True)
    sys.stdout.flush()

    try:
        print("\n调用 multiisland.run()...", flush=True)
        sys.stdout.flush()
        
        best_code, best_score = multiisland.run()

        print("\n" + "=" * 80, flush=True)
        print("实验完成!", flush=True)
        print("=" * 80, flush=True)
        print(f"Best Score: {best_score:.6f}", flush=True)

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断", flush=True)
    except Exception as e:
        print(f"\n\n✗ 实验失败: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

