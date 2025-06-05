"""
可视化工具函数
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple


def plot_training_history(training_history: Dict[str, List], 
                         figsize: Tuple[int, int] = (15, 10),
                         save_path: Optional[str] = None) -> None:
    """
    绘制训练历史
    
    Args:
        training_history: 训练历史字典
        figsize: 图形大小
        save_path: 保存路径（可选）
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 自编码器损失
    if training_history.get('autoencoder_loss'):
        axes[0, 0].plot(training_history['autoencoder_loss'])
        axes[0, 0].set_title('AutoEncoder Reconstruction Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
    
    # 集成性能
    if training_history.get('ensemble_performance'):
        perfs = training_history['ensemble_performance']
        rmses = [p['val_rmse'] for p in perfs]
        r2s = [p['val_r2'] for p in perfs]
        
        ax = axes[0, 1]
        ax2 = ax.twinx()
        
        line1 = ax.plot(rmses, 'b-', label='RMSE')
        line2 = ax2.plot(r2s, 'r-', label='R²')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('RMSE', color='b')
        ax2.set_ylabel('R²', color='r')
        ax.set_title('Validation Metrics')
        ax.grid(True)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best')
    
    # 伪标签统计：覆盖率
    if training_history.get('pseudo_label_stats'):
        stats = training_history['pseudo_label_stats']
        coverage = [s['coverage_ratio'] for s in stats]
        
        axes[1, 0].plot(coverage, marker='o', markersize=8, linewidth=2)
        axes[1, 0].set_title('Pseudo-label Coverage Ratio')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Coverage')
        axes[1, 0].grid(True)
        axes[1, 0].set_ylim(0, 1)
    
    # 不确定性
    if training_history.get('pseudo_label_stats'):
        stats = training_history['pseudo_label_stats']
        mean_unc = [s['mean_uncertainty'] for s in stats if s['mean_uncertainty'] > 0]
        
        if mean_unc:
            axes[1, 1].plot(mean_unc, marker='o', markersize=8, linewidth=2)
            axes[1, 1].set_title('Mean Prediction Uncertainty')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Uncertainty')
            axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_predictions(y_true: np.ndarray, 
                    y_pred: np.ndarray,
                    uncertainties: Optional[np.ndarray] = None,
                    title: str = "Predictions vs True Values",
                    figsize: Tuple[int, int] = (10, 8),
                    save_path: Optional[str] = None) -> None:
    """
    绘制预测值与真实值的对比图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        uncertainties: 不确定性（可选）
        title: 图表标题
        figsize: 图形大小
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=figsize)
    
    if uncertainties is not None:
        # 按不确定性着色
        scatter = plt.scatter(y_true, y_pred, c=uncertainties, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Uncertainty')
    else:
        plt.scatter(y_true, y_pred, alpha=0.6)
    
    # 添加对角线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加R²分数
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_residuals(y_true: np.ndarray, 
                  y_pred: np.ndarray,
                  title: str = "Residual Plot",
                  figsize: Tuple[int, int] = (10, 6),
                  save_path: Optional[str] = None) -> None:
    """
    绘制残差图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        title: 图表标题
        figsize: 图形大小
        save_path: 保存路径（可选）
    """
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 残差散点图
    ax1.scatter(y_pred, residuals, alpha=0.6)
    ax1.axhline(y=0, color='r', linestyle='--', lw=2)
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted')
    ax1.grid(True, alpha=0.3)
    
    # 残差直方图
    ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Residual Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 添加正态分布曲线
    from scipy import stats
    mu, std = stats.norm.fit(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, stats.norm.pdf(x, mu, std), 'r-', lw=2, label=f'Normal(μ={mu:.2f}, σ={std:.2f})')
    ax2_twin.set_ylabel('Probability Density')
    ax2_twin.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_model_comparison(metrics: Dict[str, Dict[str, float]], 
                         title: str = "Model Performance Comparison",
                         figsize: Tuple[int, int] = (12, 6),
                         save_path: Optional[str] = None) -> None:
    """
    绘制模型性能对比图
    
    Args:
        metrics: 模型性能指标字典
        title: 图表标题
        figsize: 图形大小
        save_path: 保存路径（可选）
    """
    # 提取个体模型指标
    if 'individual_models' in metrics:
        model_metrics = metrics['individual_models']
        
        models = list(model_metrics.keys())
        rmse_values = [model_metrics[m]['rmse'] for m in models]
        r2_values = [model_metrics[m]['r2'] for m in models]
        
        # 添加集成模型
        if 'ensemble' in metrics:
            models.append('Ensemble')
            rmse_values.append(metrics['ensemble']['RMSE'])
            r2_values.append(metrics['ensemble']['R2'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # RMSE对比
        bars1 = ax1.bar(models, rmse_values, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('RMSE')
        ax1.set_title('RMSE Comparison')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 高亮最佳模型
        min_idx = np.argmin(rmse_values)
        bars1[min_idx].set_color('green')
        bars1[min_idx].set_alpha(0.9)
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars1, rmse_values)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        # R²对比
        bars2 = ax2.bar(models, r2_values, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('R²')
        ax2.set_title('R² Score Comparison')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 1)
        
        # 高亮最佳模型
        max_idx = np.argmax(r2_values)
        bars2[max_idx].set_color('green')
        bars2[max_idx].set_alpha(0.9)
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars2, r2_values)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def plot_uncertainty_analysis(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            uncertainties: np.ndarray,
                            title: str = "Uncertainty Analysis",
                            figsize: Tuple[int, int] = (15, 5),
                            save_path: Optional[str] = None) -> None:
    """
    绘制不确定性分析图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        uncertainties: 不确定性估计
        title: 图表标题
        figsize: 图形大小
        save_path: 保存路径（可选）
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # 1. 不确定性vs预测误差
    errors = np.abs(y_true - y_pred)
    ax1.scatter(uncertainties, errors, alpha=0.6)
    ax1.set_xlabel('Uncertainty')
    ax1.set_ylabel('Absolute Error')
    ax1.set_title('Uncertainty vs Prediction Error')
    ax1.grid(True, alpha=0.3)
    
    # 添加趋势线
    z = np.polyfit(uncertainties, errors, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(uncertainties.min(), uncertainties.max(), 100)
    ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'Trend: {z[0]:.2f}x + {z[1]:.2f}')
    ax1.legend()
    
    # 2. 不确定性分布
    ax2.hist(uncertainties, bins=30, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Uncertainty')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Uncertainty Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_unc = np.mean(uncertainties)
    std_unc = np.std(uncertainties)
    ax2.axvline(mean_unc, color='r', linestyle='--', label=f'Mean: {mean_unc:.3f}')
    ax2.axvline(mean_unc + std_unc, color='g', linestyle='--', label=f'±1 STD: {std_unc:.3f}')
    ax2.axvline(mean_unc - std_unc, color='g', linestyle='--')
    ax2.legend()
    
    # 3. 校准图
    n_bins = 10
    bin_edges = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
    bin_centers = []
    bin_errors = []
    bin_uncertainties = []
    
    for i in range(n_bins):
        mask = (uncertainties >= bin_edges[i]) & (uncertainties <= bin_edges[i + 1])
        if np.sum(mask) > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_errors.append(np.mean(errors[mask]))
            bin_uncertainties.append(np.mean(uncertainties[mask]))
    
    ax3.scatter(bin_uncertainties, bin_errors, s=100, alpha=0.7)
    ax3.plot([0, max(bin_uncertainties)], [0, max(bin_uncertainties)], 'r--', label='Perfect Calibration')
    ax3.set_xlabel('Mean Predicted Uncertainty')
    ax3.set_ylabel('Mean Actual Error')
    ax3.set_title('Uncertainty Calibration')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(feature_importance: Dict[str, np.ndarray],
                          feature_names: List[str],
                          top_n: int = 20,
                          title: str = "Feature Importance",
                          figsize: Tuple[int, int] = (10, 8),
                          save_path: Optional[str] = None) -> None:
    """
    绘制特征重要性图
    
    Args:
        feature_importance: 各模型的特征重要性字典
        feature_names: 特征名称列表
        top_n: 显示前N个重要特征
        title: 图表标题
        figsize: 图形大小
        save_path: 保存路径（可选）
    """
    # 计算平均特征重要性
    avg_importance = np.mean(list(feature_importance.values()), axis=0)
    
    # 获取top_n个最重要的特征
    top_indices = np.argsort(avg_importance)[-top_n:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_importance = avg_importance[top_indices]
    
    plt.figure(figsize=figsize)
    
    # 绘制条形图
    bars = plt.barh(range(len(top_features)), top_importance, alpha=0.7, edgecolor='black')
    
    # 为最重要的特征着色
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance Score')
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for i, (feature, importance) in enumerate(zip(top_features, top_importance)):
        plt.text(importance + 0.001, i, f'{importance:.4f}', 
                va='center', ha='left', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
