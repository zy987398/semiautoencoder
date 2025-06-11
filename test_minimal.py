#!/usr/bin/env python3
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
