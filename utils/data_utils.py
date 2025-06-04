"""
数据处理工具函数
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Any, Dict
from sklearn.model_selection import train_test_split


def load_data(labeled_file: str, 
              unlabeled_file: str, 
              target_column: str = 'target_length',
              feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载标记和未标记数据
    
    Args:
        labeled_file: 标记数据文件路径
        unlabeled_file: 未标记数据文件路径
        target_column: 目标列名
        feature_columns: 特征列名列表，如果为None则使用除目标列外的所有列
        
    Returns:
        X_labeled: 标记数据特征
        y_labeled: 标记数据目标
        X_unlabeled: 未标记数据特征
    """
    # 读取数据
    labeled_df = pd.read_csv(labeled_file)
    unlabeled_df = pd.read_csv(unlabeled_file)
    
    # 确定特征列
    if feature_columns is None:
        feature_columns = [col for col in labeled_df.columns if col != target_column]
    
    # 提取特征和目标
    X_labeled = labeled_df[feature_columns].values.astype(np.float32)
    y_labeled = labeled_df[target_column].values.astype(np.float32)
    X_unlabeled = unlabeled_df[feature_columns].values.astype(np.float32)
    
    return X_labeled, y_labeled, X_unlabeled


def prepare_data_splits(X: np.ndarray, 
                       y: np.ndarray, 
                       test_size: float = 0.2,
                       val_size: float = 0.2,
                       random_state: int = 42) -> Tuple[np.ndarray, ...]:
    """
    准备训练、验证和测试数据集
    
    Args:
        X: 特征数据
        y: 目标数据
        test_size: 测试集比例
        val_size: 验证集比例（相对于训练集）
        random_state: 随机种子
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # 先分出测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 再从剩余数据中分出验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def check_data_quality(X: np.ndarray, 
                      y: Optional[np.ndarray] = None,
                      verbose: bool = True) -> Dict[str, Any]:
    """
    检查数据质量
    
    Args:
        X: 特征数据
        y: 目标数据（可选）
        verbose: 是否打印信息
        
    Returns:
        quality_report: 数据质量报告
    """
    quality_report = {}
    
    # 检查特征数据
    quality_report['n_samples'] = X.shape[0]
    quality_report['n_features'] = X.shape[1]
    quality_report['has_nan'] = np.any(np.isnan(X))
    quality_report['has_inf'] = np.any(np.isinf(X))
    
    if quality_report['has_nan']:
        quality_report['nan_count'] = np.sum(np.isnan(X))
        quality_report['nan_columns'] = np.where(np.any(np.isnan(X), axis=0))[0].tolist()
    
    # 检查目标数据
    if y is not None:
        quality_report['target_has_nan'] = np.any(np.isnan(y))
        quality_report['target_has_inf'] = np.any(np.isinf(y))
        quality_report['target_min'] = np.min(y)
        quality_report['target_max'] = np.max(y)
        quality_report['target_mean'] = np.mean(y)
        quality_report['target_std'] = np.std(y)
    
    if verbose:
        print("=== Data Quality Report ===")
        print(f"Number of samples: {quality_report['n_samples']}")
        print(f"Number of features: {quality_report['n_features']}")
        print(f"Has NaN values: {quality_report['has_nan']}")
        print(f"Has Inf values: {quality_report['has_inf']}")
        
        if y is not None:
            print(f"\nTarget statistics:")
            print(f"  Min: {quality_report['target_min']:.4f}")
            print(f"  Max: {quality_report['target_max']:.4f}")
            print(f"  Mean: {quality_report['target_mean']:.4f}")
            print(f"  Std: {quality_report['target_std']:.4f}")
    
    return quality_report


def remove_outliers(X: np.ndarray, 
                   y: np.ndarray, 
                   iqr_multiplier: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用IQR方法移除异常值
    
    Args:
        X: 特征数据
        y: 目标数据
        iqr_multiplier: IQR倍数
        
    Returns:
        X_clean: 清理后的特征数据
        y_clean: 清理后的目标数据
    """
    # 计算目标值的IQR
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr
    
    # 过滤异常值
    mask = (y >= lower_bound) & (y <= upper_bound)
    X_clean = X[mask]
    y_clean = y[mask]
    
    print(f"Removed {len(y) - len(y_clean)} outliers ({(len(y) - len(y_clean)) / len(y) * 100:.2f}%)")
    
    return X_clean, y_clean