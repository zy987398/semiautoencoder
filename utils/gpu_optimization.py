"""
GPU优化工具函数
"""
import torch
import numpy as np
from typing import Optional, Tuple
import gc


def check_gpu_availability() -> Tuple[bool, Optional[str]]:
    """
    检查GPU可用性
    
    Returns:
        is_available: GPU是否可用
        device_info: GPU信息
    """
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
        memory_cached = torch.cuda.memory_reserved(current_device) / 1024**3
        
        info = f"""
GPU Information:
- Device Count: {device_count}
- Current Device: {current_device}
- Device Name: {device_name}
- Total Memory: {memory_total:.2f} GB
- Allocated Memory: {memory_allocated:.2f} GB
- Cached Memory: {memory_cached:.2f} GB
        """
        return True, info
    else:
        return False, "No GPU available. Using CPU."


def get_optimal_batch_size(model: torch.nn.Module, 
                          input_shape: Tuple[int, ...],
                          device: torch.device,
                          max_batch_size: int = 1024,
                          safety_factor: float = 0.9) -> int:
    """
    自动确定最优批大小以最大化GPU利用率
    
    Args:
        model: 模型
        input_shape: 输入形状（不包括batch维度）
        device: 设备
        max_batch_size: 最大批大小
        safety_factor: 安全系数（0-1）
        
    Returns:
        optimal_batch_size: 最优批大小
    """
    if device.type == 'cpu':
        return 64  # CPU默认批大小
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 二分搜索找到最大可行批大小
    low = 1
    high = max_batch_size
    optimal_batch_size = 1
    
    while low <= high:
        mid = (low + high) // 2
        try:
            # 尝试前向传播
            dummy_input = torch.randn(mid, *input_shape, device=device)
            model.train()
            with torch.cuda.amp.autocast():  # 使用混合精度
                _ = model(dummy_input)
            
            # 尝试反向传播
            loss = torch.randn(1, device=device)
            loss.backward()
            
            # 如果成功，增加批大小
            optimal_batch_size = mid
            low = mid + 1
            
            # 清理
            del dummy_input, loss
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                high = mid - 1
                torch.cuda.empty_cache()
                gc.collect()
            else:
                raise e
    
    # 应用安全系数
    safe_batch_size = int(optimal_batch_size * safety_factor)
    print(f"Optimal batch size for GPU: {safe_batch_size}")
    return safe_batch_size


def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        print("GPU memory cleared")


def move_data_to_device(data, device: torch.device):
    """
    将数据移动到指定设备
    
    Args:
        data: 数据（numpy array, torch tensor, list, dict等）
        device: 目标设备
        
    Returns:
        移动后的数据
    """
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [move_data_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_data_to_device(item, device) for item in data)
    elif isinstance(data, dict):
        return {key: move_data_to_device(value, device) for key, value in data.items()}
    else:
        return data


class GPUDataLoader:
    """
    GPU优化的数据加载器包装器
    """
    def __init__(self, dataloader, device: torch.device):
        self.dataloader = dataloader
        self.device = device
    
    def __iter__(self):
        for batch in self.dataloader:
            yield move_data_to_device(batch, self.device)
    
    def __len__(self):
        return len(self.dataloader)


def enable_mixed_precision_training():
    """
    启用混合精度训练设置
    
    Returns:
        scaler: GradScaler对象
    """
    # 检查GPU是否支持混合精度
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 7:  # Volta架构及以上
            print("Mixed precision training enabled (GPU supports it)")
            return torch.cuda.amp.GradScaler()
        else:
            print("GPU does not support mixed precision training efficiently")
            return None
    else:
        print("Mixed precision training not available (no GPU)")
        return None


def optimize_model_for_gpu(model: torch.nn.Module) -> torch.nn.Module:
    """
    优化模型以获得更好的GPU性能
    
    Args:
        model: 原始模型
        
    Returns:
        优化后的模型
    """
    if torch.cuda.is_available():
        # 启用cudnn基准测试
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # 如果模型参数较少，考虑使用DataParallel
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            model = torch.nn.DataParallel(model)
    
    return model


def profile_gpu_usage(func):
    """
    装饰器：分析函数的GPU使用情况
    """
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated()
            
            result = func(*args, **kwargs)
            
            torch.cuda.synchronize()
            end_memory = torch.cuda.memory_allocated()
            
            print(f"Function {func.__name__} GPU memory usage: "
                  f"{(end_memory - start_memory) / 1024**2:.2f} MB")
            
            return result
        else:
            return func(*args, **kwargs)
    
    return wrapper
