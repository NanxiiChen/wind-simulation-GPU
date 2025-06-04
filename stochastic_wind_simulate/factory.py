def get_simulator(backend="jax", **kwargs):
    """
    获取指定后端的风场模拟器
    
    参数:
        backend: 'jax' 或 'torch'
        **kwargs: 传递给模拟器的参数
    """
    if backend.lower() == "jax":
        from .jax_backend.simulator import JaxWindSimulator
        return JaxWindSimulator(**kwargs)
    elif backend.lower() == "torch":
        from .torch_backend.simulator import TorchWindSimulator
        return TorchWindSimulator(**kwargs)
    else:
        raise ValueError(f"不支持的后端: {backend}，目前支持 'jax' 和 'torch'")

def get_visualizer(backend="jax", **kwargs):
    """
    获取指定后端的可视化器
    
    参数:
        backend: 'jax' 或 'torch'
        **kwargs: 传递给可视化器的参数
    """
    if backend.lower() == "jax":
        from .jax_backend.visualizer import JaxWindVisualizer
        return JaxWindVisualizer(**kwargs)
    elif backend.lower() == "torch":
        from .torch_backend.visualizer import TorchWindVisualizer
        return TorchWindVisualizer(**kwargs)
    else:
        raise ValueError(f"不支持的后端: {backend}，目前支持 'jax' 和 'torch'")