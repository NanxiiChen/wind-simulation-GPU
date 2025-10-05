import torch
import torch.func as func
import time
import numpy as np

def test_vmap_behavior():
    print("测试PyTorch vmap行为...")
    
    # 测试1: 简单的vmap操作
    def simple_func(x):
        return torch.sum(x**2)
    
    # 创建测试数据
    data = torch.randn(100, 1000)
    
    print("测试1: 简单vmap操作")
    
    # 方法1: vmap
    start = time.time()
    result1 = func.vmap(simple_func)(data)
    time1 = time.time() - start
    print(f"vmap时间: {time1:.4f}s")
    
    # 方法2: 循环
    start = time.time()
    result2 = torch.stack([simple_func(x) for x in data])
    time2 = time.time() - start
    print(f"循环时间: {time2:.4f}s")
    
    print(f"结果相等: {torch.allclose(result1, result2)}")
    print(f"vmap vs 循环加速比: {time2/time1:.2f}x")
    
    # 测试2: 复杂操作 (模拟你的风场计算)
    print("\n测试2: 复杂矩阵操作")
    
    def complex_func(freq, positions):
        """模拟build_spectrum_for_position"""
        n = positions.shape[0]
        # 模拟计算
        result = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                result[i, j] = torch.sin(freq * (i+j)) * torch.exp(-freq/10)
        return result
    
    # 创建测试数据
    frequencies = torch.linspace(0.1, 10, 50)
    positions = torch.randn(20, 3)
    
    print(f"频率点数: {len(frequencies)}, 位置点数: {positions.shape[0]}")
    
    # 方法1: vmap
    start = time.time()
    try:
        result1 = func.vmap(complex_func, in_dims=(0, None))(frequencies, positions)
        time1 = time.time() - start
        print(f"vmap时间: {time1:.4f}s")
        vmap_success = True
    except Exception as e:
        print(f"vmap失败: {e}")
        vmap_success = False
        time1 = np.inf
    
    # 方法2: 循环
    start = time.time()
    result2 = torch.stack([complex_func(freq, positions) for freq in frequencies])
    time2 = time.time() - start
    print(f"循环时间: {time2:.4f}s")
    
    if vmap_success:
        print(f"结果相等: {torch.allclose(result1, result2)}")
        print(f"vmap vs 循环加速比: {time2/time1:.2f}x")
    
    return vmap_success

if __name__ == "__main__":
    test_vmap_behavior()
