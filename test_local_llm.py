"""
测试本地LLM是否能正确响应
"""

import sys
from openai import OpenAI

print("=" * 80)
print("测试本地LLM响应")
print("=" * 80)

# 初始化客户端
print("\n1. 初始化OpenAI客户端...")
client = OpenAI(
    base_url='http://localhost:8000/v1',
    api_key='EMPTY'
)
print("✓ 客户端初始化完成")

# 测试1: 简单对话
print("\n2. 测试简单对话...")
try:
    response = client.chat.completions.create(
        model='xingyaoww/CodeActAgent-Mistral-7b-v0.1',
        messages=[
            {'role': 'user', 'content': 'Say "Hello World" and nothing else.'}
        ],
        max_tokens=50,
        temperature=0.7,
        timeout=60
    )
    
    content = response.choices[0].message.content
    print(f"✓ LLM响应: {content}")
    
except Exception as e:
    print(f"✗ 错误: {e}")
    sys.exit(1)

# 测试2: 代码生成
print("\n3. 测试代码生成...")
try:
    response = client.chat.completions.create(
        model='xingyaoww/CodeActAgent-Mistral-7b-v0.1',
        messages=[
            {'role': 'user', 'content': 'Write a Python function that adds two numbers. Only output the code, no explanation.'}
        ],
        max_tokens=200,
        temperature=0.7,
        timeout=60
    )
    
    content = response.choices[0].message.content
    print(f"✓ LLM响应:\n{content}")
    
except Exception as e:
    print(f"✗ 错误: {e}")
    sys.exit(1)

# 测试3: 长文本生成（模拟EoH的prompt）
print("\n4. 测试长文本生成（模拟EoH）...")
try:
    prompt = """
You are an expert algorithm designer. Your task is to improve the following Python function.

Current function:
```python
def select_ports(K, N_selected, N_Ports, Pt, n, H, noise):
    import numpy as np
    port_sample = np.zeros((n, N_selected), dtype=int)
    for j in range(n):
        p = np.random.choice(N_Ports, N_selected, replace=False)
        port_sample[j,:] = p
    return port_sample
```

Improve this function to select better ports. Output ONLY the improved function code, nothing else.
"""
    
    print("发送prompt...")
    sys.stdout.flush()
    
    response = client.chat.completions.create(
        model='xingyaoww/CodeActAgent-Mistral-7b-v0.1',
        messages=[
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=500,
        temperature=0.7,
        timeout=120
    )
    
    content = response.choices[0].message.content
    print(f"✓ LLM响应 ({len(content)} 字符):")
    print(content[:200] + "..." if len(content) > 200 else content)
    
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ 所有测试通过！本地LLM工作正常。")
print("=" * 80)
print("\n如果这个测试通过，但EoH还是卡住，问题可能在:")
print("  1. llm4ad库的实现")
print("  2. EoH的prompt太长")
print("  3. 超时设置")
print("  4. 响应解析")
