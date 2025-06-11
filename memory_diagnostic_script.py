"""
快速内存诊断和修复脚本
运行此脚本来诊断和解决内存问题
"""

import numpy as np
import pandas as pd
import psutil
import torch
import gc
import os
import sys
from pathlib import Path

def diagnose_system():
    """诊断系统资源"""
    print("="*60)
    print("SYSTEM RESOURCE DIAGNOSIS")
    print("="*60)
    
    # CPU信息
    print("\n1. CPU Information:")
    print(f"   - CPU Count: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    print(f"   - CPU Usage: {psutil.cpu_percent(interval=1)}%")
    
    # 内存信息
    memory = psutil.virtual_memory()
    print("\n2. Memory Information:")
    print(f"   - Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"   - Available RAM: {memory.available / (1024**3):.1f} GB")
    print(f"   - Used RAM: {memory.used / (1024**3):.1f} GB ({memory.percent}%)")
    print(f"   - Free RAM: {memory.free / (1024**3):.1f} GB")
    
    # GPU信息
    print("\n3. GPU Information:")
    if torch.cuda.is_available():
        print(f"   - GPU Available: Yes")
        print(f"   - GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"   - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        print(f"   - GPU Memory Allocated: {torch.cuda.memory_allocated() / (1024**2):.1f} MB")
        print(f"   - GPU Memory Cached: {torch.cuda.memory_reserved() / (1024**2):.1f} MB")
    else:
        print("   - GPU Available: No")
    
    # 诊断结果
    risk_level = "LOW"
    if memory.percent > 90:
        risk_level = "CRITICAL"
    elif memory.percent > 80:
        risk_level = "HIGH"
    elif memory.percent > 70:
        risk_level = "MEDIUM"
    
    print(f"\n4. Risk Assessment: {risk_level}")
    
    return {
        'memory_percent': memory.percent,
        'available_gb': memory.available / (1024**3),
        'risk_level': risk_level,
        'has_gpu': torch.cuda.is_available()
    }


def analyze_data_size(labeled_file, unlabeled_file, target_column='log_target_length'):
    """分析数据规模"""
    print("\n" + "="*60)
    print("DATA SIZE ANALYSIS")
    print("="*60)
    
    try:
        # 读取数据维度
        df_labeled = pd.read_csv(labeled_file, nrows=5)
        df_unlabeled = pd.read_csv(unlabeled_file, nrows=5)
        
        # 计算行数
        n_labeled = sum(1 for _ in open(labeled_file)) - 1
        n_unlabeled = sum(1 for _ in open(unlabeled_file)) - 1
        
        # 特征数
        n_features = len(df_labeled.columns) - 1  # 减去目标列
        
        print(f"\n1. Data Dimensions:")
        print(f"   - Labeled samples: {n_labeled:,}")
        print(f"   - Unlabeled samples: {n_unlabeled:,}")
        print(f"   - Features: {n_features}")
        print(f"   - Total samples: {n_labeled + n_unlabeled:,}")
        
        # 估算内存需求
        # 原始数据
        data_memory = (n_labeled + n_unlabeled) * n_features * 4 / (1024**2)  # float32
        
        # 多项式特征 (degree=2)
        poly_features = n_features * (n_features + 1) // 2
        poly_memory = (n_labeled + n_unlabeled) * poly_features * 4 / (1024**2)
        
        # 模型内存（估算）
        vae_memory = 100  # MB
        ensemble_memory = 500  # MB
        
        total_memory = data_memory + poly_memory + vae_memory + ensemble_memory
        
        print(f"\n2. Estimated Memory Requirements:")
        print(f"   - Raw data: {data_memory:.1f} MB")
        print(f"   - Polynomial features: {poly_memory:.1f} MB")
        print(f"   - VAE model: ~{vae_memory} MB")
        print(f"   - Ensemble models: ~{ensemble_memory} MB")
        print(f"   - Total estimated: {total_memory:.1f} MB")
        
        return {
            'n_labeled': n_labeled,
            'n_unlabeled': n_unlabeled,
            'n_features': n_features,
            'total_memory_mb': total_memory
        }
        
    except Exception as e:
        print(f"Error analyzing data: {e}")
        return None


def generate_optimized_config(system_info, data_info):
    """生成优化的配置文件"""
    print("\n" + "="*60)
    print("GENERATING OPTIMIZED CONFIGURATION")
    print("="*60)
    
    available_ram = system_info['available_gb']
    total_samples = data_info['n_labeled'] + data_info['n_unlabeled']
    
    # 决定配置级别
    if available_ram < 4 or total_samples > 100000:
        config_level = "MINIMAL"
    elif available_ram < 8 or total_samples > 50000:
        config_level = "CONSERVATIVE"
    else:
        config_level = "STANDARD"
    
    print(f"\nRecommended configuration level: {config_level}")
    
    config_content = f"""# Auto-generated optimized configuration
import torch
import psutil

SEED = 42
DEVICE = 'cpu'  # Using CPU to avoid GPU memory issues

# Memory mode: {config_level}
MEMORY_MODE = '{config_level.lower()}'

# Optimized configurations based on your system
AUTOENCODER_CONFIG = {{
    'latent_dims': {[16, 8] if config_level == "MINIMAL" else [32, 16]},
    'dropout_rate': 0.2,
    'epochs': {100 if config_level == "MINIMAL" else 150},
    'batch_size': {64 if config_level == "MINIMAL" else 128},
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'scheduler_patience': 10,
    'scheduler_factor': 0.5,
    'kl_weight': 1e-3,
    'use_improved_vae': False,
    'use_attention': False,
    'use_residual': False
}}

ENSEMBLE_CONFIG = {{
    {'# Only LightGBM for minimal config' if config_level == "MINIMAL" else ''}
    'LightGBM': {{
        'n_estimators': {200 if config_level == "MINIMAL" else 400},
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'verbosity': -1,
        'force_col_wise': True,
        'device_type': 'cpu',
        'num_threads': 2,
    }}{',' if config_level != "MINIMAL" else ''}
    {'''
    'XGBoost': {
        'n_estimators': 300,
        'learning_rate': 0.1,
        'max_depth': 4,
        'min_child_weight': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'verbosity': 0,
        'tree_method': 'hist',
        'nthread': 2,
    }''' if config_level != "MINIMAL" else ''}
}}

SEMI_SUPERVISED_CONFIG = {{
    'validation_split': 0.2,
    'n_cv_folds': {2 if config_level == "MINIMAL" else 3},
    'poly_degree': {'1' if total_samples > 100000 else '2'},  # Reduce polynomial degree for large datasets
    'poly_interaction_only': True,
    'poly_include_bias': False,
    'use_advanced_pseudo_labeling': False,
    'use_active_learning': False,
    'use_clustering': False
}}

PSEUDO_LABEL_CONFIG = {{
    'confidence_threshold': 0.3,
    'reconstruction_threshold': 0.8,
    'ensemble_agreement_threshold': 0.85,
    'min_pseudo_labels': {10 if config_level == "MINIMAL" else 30},
    'iqr_multiplier': 1.5
}}

SELF_TRAINING_CONFIG = {{
    'n_iterations': {1 if config_level == "MINIMAL" else 2},
    'high_confidence_percentile': 40,
}}

OPTIMIZATION_CONFIG = {{
    'chunk_size': {min(5000, total_samples // 20)},
    'n_bags': {2 if config_level == "MINIMAL" else 3},
    'bag_subsample_ratio': 0.6,
    'dataloader_num_workers': 0,
    'n_jobs': 2,
}}

MEMORY_CONFIG = {{
    'enable_gc_collection': True,
    'gc_collect_frequency': 5,
    'clear_gpu_cache': False,  # Not using GPU
    'log_memory_usage': True,
    'memory_limit_mb': {int(available_ram * 1024 * 0.7)},  # Use 70% of available RAM
}}

print(f"Configuration loaded: {{MEMORY_MODE}} mode")
print(f"Memory limit: {{MEMORY_CONFIG['memory_limit_mb']}} MB")
"""
    
    # 保存配置
    config_path = "config_optimized.py"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"\nOptimized configuration saved to: {config_path}")
    print("\nKey optimizations:")
    print(f"  - Batch size: {64 if config_level == 'MINIMAL' else 128}")
    print(f"  - Chunk size: {min(5000, total_samples // 20)}")
    print(f"  - Number of models: {'1' if config_level == 'MINIMAL' else '2'}")
    print(f"  - Polynomial degree: {'1' if total_samples > 100000 else '2'}")
    print(f"  - Memory limit: {int(available_ram * 1024 * 0.7)} MB")
    
    return config_level


def create_minimal_test_script():
    """创建最小化测试脚本"""
    script_content = '''#!/usr/bin/env python3
"""
Minimal test script for memory-constrained systems
"""
import numpy as np
import gc
import psutil
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

# 监控内存
def log_memory(stage):
    mem = psutil.Process().memory_info().rss / 1024**2
    print(f"[{stage}] Memory: {mem:.1f} MB")

# 简化的管道
def minimal_pipeline(X_labeled, y_labeled, X_unlabeled, max_samples=5000):
    print("Running minimal pipeline...")
    log_memory("Start")
    
    # 限制数据大小
    if len(X_labeled) > max_samples:
        indices = np.random.choice(len(X_labeled), max_samples, replace=False)
        X_labeled = X_labeled[indices]
        y_labeled = y_labeled[indices]
    
    if len(X_unlabeled) > max_samples:
        indices = np.random.choice(len(X_unlabeled), max_samples, replace=False)
        X_unlabeled = X_unlabeled[indices]
    
    # 标准化
    scaler = StandardScaler()
    X_all = np.vstack([X_labeled, X_unlabeled])
    scaler.fit(X_all[:1000])  # Fit on subset
    
    X_labeled = scaler.transform(X_labeled)
    X_unlabeled = scaler.transform(X_unlabeled)
    
    log_memory("After preprocessing")
    
    # 简单模型
    model = SGDRegressor(max_iter=1000, tol=1e-3)
    model.fit(X_labeled, y_labeled)
    
    log_memory("After training")
    
    # 简单伪标签
    predictions = model.predict(X_unlabeled[:1000])
    
    print("Minimal pipeline completed!")
    return model

if __name__ == "__main__":
    # 测试用小数据
    X_labeled = np.random.randn(1000, 10)
    y_labeled = np.random.randn(1000)
    X_unlabeled = np.random.randn(2000, 10)
    
    model = minimal_pipeline(X_labeled, y_labeled, X_unlabeled)
    print("Test successful!")
'''
    
    with open("test_minimal.py", 'w') as f:
        f.write(script_content)
    
    print("\nMinimal test script created: test_minimal.py")
    print("Run it with: python test_minimal.py")


def main():
    """主诊断流程"""
    print("\n" + "="*70)
    print("SUBSURFACE CRACK PREDICTION MODEL - MEMORY DIAGNOSTIC TOOL")
    print("="*70)
    
    # 1. 系统诊断
    system_info = diagnose_system()
    
    # 2. 给出立即的建议
    print("\n" + "="*60)
    print("IMMEDIATE RECOMMENDATIONS")
    print("="*60)
    
    if system_info['risk_level'] in ['HIGH', 'CRITICAL']:
        print("\n⚠️  HIGH MEMORY USAGE DETECTED!")
        print("\nImmediate actions:")
        print("1. Close unnecessary applications")
        print("2. Run: python -m torch.cuda.empty_cache()")
        print("3. Restart Python kernel/interpreter")
    
    # 3. 数据分析
    print("\nDo you want to analyze your data files? (y/n): ", end='')
    if input().lower() == 'y':
        labeled_file = input("Enter path to labeled data file: ") or "data/labeled_log.csv"
        unlabeled_file = input("Enter path to unlabeled data file: ") or "data/unlabeled.csv"
        
        data_info = analyze_data_size(labeled_file, unlabeled_file)
        
        if data_info:
            # 4. 生成优化配置
            print("\nGenerate optimized configuration? (y/n): ", end='')
            if input().lower() == 'y':
                config_level = generate_optimized_config(system_info, data_info)
                
                # 5. 创建测试脚本
                create_minimal_test_script()
    
    # 6. 最终建议
    print("\n" + "="*60)
    print("FINAL RECOMMENDATIONS")
    print("="*60)
    
    print("\n1. For immediate relief:")
    print("   - Use the generated config_optimized.py")
    print("   - Run test_minimal.py to verify basic functionality")
    print("   - Consider reducing data size or using sampling")
    
    print("\n2. For long-term solution:")
    print("   - Upgrade RAM if consistently hitting limits")
    print("   - Use cloud computing with more resources")
    print("   - Implement data streaming/chunking")
    
    print("\n3. Quick fixes in existing code:")
    print("   - Replace config.py with config_optimized.py")
    print("   - Add gc.collect() after major operations")
    print("   - Reduce batch sizes and number of models")
    
    print("\nDiagnostic complete!")


if __name__ == "__main__":
    main()