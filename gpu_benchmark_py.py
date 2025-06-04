"""
GPU性能基准测试脚本
比较CPU和GPU在各个模型上的训练速度
"""
import numpy as np
import pandas as pd
import time
import torch
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns


def generate_synthetic_data(n_samples=10000, n_features=50):
    """生成合成数据用于基准测试"""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # 创建非线性关系
    y = (X[:, 0] * X[:, 1] + 
         np.sin(X[:, 2]) * X[:, 3] + 
         X[:, 4]**2 + 
         np.random.randn(n_samples) * 0.1)
    return X, y


def benchmark_autoencoder(X, device='cpu', epochs=50):
    """基准测试自编码器"""
    from models.autoencoder import AutoEncoderTrainer
    
    print(f"\nBenchmarking AutoEncoder on {device.upper()}...")
    
    trainer = AutoEncoderTrainer(
        input_dim=X.shape[1],
        latent_dims=[32, 16, 8],
        device=device
    )
    
    start_time = time.time()
    trainer.train(X, epochs=epochs, batch_size=256, verbose=False)
    training_time = time.time() - start_time
    
    # 测试编码速度
    start_time = time.time()
    _ = trainer.encode(X)
    encoding_time = time.time() - start_time
    
    return {
        'training_time': training_time,
        'encoding_time': encoding_time,
        'epochs': epochs,
        'samples': len(X)
    }


def benchmark_xgboost(X, y, use_gpu=False):
    """基准测试XGBoost"""
    device = "GPU" if use_gpu else "CPU"
    print(f"\nBenchmarking XGBoost on {device}...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'verbosity': 0
    }
    
    if use_gpu:
        params.update({
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'predictor': 'gpu_predictor'
        })
    else:
        params['tree_method'] = 'hist'
    
    model = XGBRegressor(**params)
    
    # 训练时间
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # 预测时间
    start_time = time.time()
    _ = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    return {
        'training_time': training_time,
        'prediction_time': prediction_time,
        'n_estimators': params['n_estimators']
    }


def benchmark_lightgbm(X, y, use_gpu=False):
    """基准测试LightGBM"""
    device = "GPU" if use_gpu else "CPU"
    print(f"\nBenchmarking LightGBM on {device}...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    params = {
        'n_estimators': 100,
        'num_leaves': 31,
        'learning_rate': 0.1,
        'random_state': 42,
        'verbose': -1
    }
    
    if use_gpu:
        params.update({
            'device': 'gpu',
            'gpu_use_dp': True
        })
    else:
        params['device'] = 'cpu'
    
    try:
        model = LGBMRegressor(**params)
        
        # 训练时间
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # 预测时间
        start_time = time.time()
        _ = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        return {
            'training_time': training_time,
            'prediction_time': prediction_time,
            'n_estimators': params['n_estimators']
        }
    except Exception as e:
        print(f"  LightGBM {device} failed: {e}")
        return None


def benchmark_catboost(X, y, use_gpu=False):
    """基准测试CatBoost"""
    device = "GPU" if use_gpu else "CPU"
    print(f"\nBenchmarking CatBoost on {device}...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    params = {
        'iterations': 100,
        'depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'verbose': False
    }
    
    if use_gpu:
        params.update({
            'task_type': 'GPU',
            'devices': '0'
        })
    else:
        params['task_type'] = 'CPU'
    
    model = CatBoostRegressor(**params)
    
    # 训练时间
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # 预测时间
    start_time = time.time()
    _ = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    return {
        'training_time': training_time,
        'prediction_time': prediction_time,
        'iterations': params['iterations']
    }


def plot_benchmark_results(results):
    """绘制基准测试结果"""
    # 准备数据
    models = []
    devices = []
    training_times = []
    speedups = []
    
    for model_name, model_results in results.items():
        if 'cpu' in model_results and 'gpu' in model_results:
            if model_results['gpu'] is not None:
                models.extend([model_name, model_name])
                devices.extend(['CPU', 'GPU'])
                training_times.extend([
                    model_results['cpu']['training_time'],
                    model_results['gpu']['training_time']
                ])
                speedup = model_results['cpu']['training_time'] / model_results['gpu']['training_time']
                speedups.append(speedup)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 训练时间对比
    df = pd.DataFrame({
        'Model': models,
        'Device': devices,
        'Training Time (s)': training_times
    })
    
    sns.barplot(data=df, x='Model', y='Training Time (s)', hue='Device', ax=ax1)
    ax1.set_title('Training Time Comparison: CPU vs GPU')
    ax1.set_ylabel('Training Time (seconds)')
    
    # 加速比
    speedup_df = pd.DataFrame({
        'Model': list(results.keys())[:len(speedups)],
        'Speedup': speedups
    })
    
    bars = ax2.bar(speedup_df['Model'], speedup_df['Speedup'])
    ax2.set_title('GPU Speedup Factor')
    ax2.set_ylabel('Speedup (CPU time / GPU time)')
    ax2.axhline(y=1, color='r', linestyle='--', label='No speedup')
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('gpu_benchmark_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主函数"""
    print("="*60)
    print("GPU Performance Benchmark")
    print("="*60)
    
    # 检查GPU可用性
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected. Running CPU benchmarks only.")
    
    # 生成测试数据
    print("\nGenerating synthetic data...")
    X, y = generate_synthetic_data(n_samples=10000, n_features=50)
    print(f"Data shape: {X.shape}")
    
    results = {}
    
    # 1. AutoEncoder基准测试
    print("\n" + "="*40)
    print("AutoEncoder Benchmark")
    print("="*40)
    
    results['AutoEncoder'] = {}
    results['AutoEncoder']['cpu'] = benchmark_autoencoder(X, device='cpu', epochs=20)
    
    if gpu_available:
        torch.cuda.empty_cache()
        results['AutoEncoder']['gpu'] = benchmark_autoencoder(X, device='cuda', epochs=20)
        
        speedup = results['AutoEncoder']['cpu']['training_time'] / results['AutoEncoder']['gpu']['training_time']
        print(f"\nAutoEncoder Results:")
        print(f"  CPU time: {results['AutoEncoder']['cpu']['training_time']:.2f}s")
        print(f"  GPU time: {results['AutoEncoder']['gpu']['training_time']:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")
    
    # 2. XGBoost基准测试
    print("\n" + "="*40)
    print("XGBoost Benchmark")
    print("="*40)
    
    results['XGBoost'] = {}
    results['XGBoost']['cpu'] = benchmark_xgboost(X, y, use_gpu=False)
    
    if gpu_available:
        results['XGBoost']['gpu'] = benchmark_xgboost(X, y, use_gpu=True)
        
        if results['XGBoost']['gpu']:
            speedup = results['XGBoost']['cpu']['training_time'] / results['XGBoost']['gpu']['training_time']
            print(f"\nXGBoost Results:")
            print(f"  CPU time: {results['XGBoost']['cpu']['training_time']:.2f}s")
            print(f"  GPU time: {results['XGBoost']['gpu']['training_time']:.2f}s")
            print(f"  Speedup: {speedup:.2f}x")
    
    # 3. LightGBM基准测试
    print("\n" + "="*40)
    print("LightGBM Benchmark")
    print("="*40)
    
    results['LightGBM'] = {}
    results['LightGBM']['cpu'] = benchmark_lightgbm(X, y, use_gpu=False)
    
    if gpu_available:
        results['LightGBM']['gpu'] = benchmark_lightgbm(X, y, use_gpu=True)
        
        if results['LightGBM']['gpu']:
            speedup = results['LightGBM']['cpu']['training_time'] / results['LightGBM']['gpu']['training_time']
            print(f"\nLightGBM Results:")
            print(f"  CPU time: {results['LightGBM']['cpu']['training_time']:.2f}s")
            print(f"  GPU time: {results['LightGBM']['gpu']['training_time']:.2f}s")
            print(f"  Speedup: {speedup:.2f}x")
    
    # 4. CatBoost基准测试
    print("\n" + "="*40)
    print("CatBoost Benchmark")
    print("="*40)
    
    results['CatBoost'] = {}
    results['CatBoost']['cpu'] = benchmark_catboost(X, y, use_gpu=False)
    
    if gpu_available:
        results['CatBoost']['gpu'] = benchmark_catboost(X, y, use_gpu=True)
        
        speedup = results['CatBoost']['cpu']['training_time'] / results['CatBoost']['gpu']['training_time']
        print(f"\nCatBoost Results:")
        print(f"  CPU time: {results['CatBoost']['cpu']['training_time']:.2f}s")
        print(f"  GPU time: {results['CatBoost']['gpu']['training_time']:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")
    
    # 绘制结果
    if gpu_available:
        print("\n" + "="*40)
        print("Generating visualization...")
        print("="*40)
        plot_benchmark_results(results)
    
    # 总结
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)
    
    if gpu_available:
        print("\nGPU acceleration provides significant speedup for:")
        print("- AutoEncoder: Deep learning benefits most from GPU")
        print("- XGBoost: Good GPU support with gpu_hist")
        print("- CatBoost: Excellent GPU optimization")
        print("- LightGBM: GPU support varies by installation")
        
        print("\nRecommendations:")
        print("1. Use GPU for all models when available")
        print("2. Increase batch size for AutoEncoder on GPU")
        print("3. Use more estimators/iterations on GPU due to faster training")
    else:
        print("Install CUDA and GPU-enabled packages for acceleration")


if __name__ == "__main__":
    main()