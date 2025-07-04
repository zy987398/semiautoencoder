"""
半监督学习主运行脚本
"""
import numpy as np
import pandas as pd
import torch
import json
import argparse
import time
from pathlib import Path
from typing import Optional
from sklearn.preprocessing import StandardScaler

# 设置随机种子
from config import SEED, DEVICE
torch.manual_seed(SEED)
np.random.seed(SEED)

# 导入自定义模块
from core.semi_supervised import EnhancedSemiSupervisedEnsemble
from utils.data_utils import load_data, prepare_data_splits, check_data_quality
from utils.visualization import (
    plot_training_history, plot_predictions, plot_residuals,
    plot_model_comparison, plot_uncertainty_analysis
)
from utils.gpu_optimization import check_gpu_availability, clear_gpu_memory
from config import (
    AUTOENCODER_CONFIG, ENSEMBLE_CONFIG, SEMI_SUPERVISED_CONFIG,
    PSEUDO_LABEL_CONFIG, SELF_TRAINING_CONFIG, 
    OPTIMIZATION_CONFIG, EVALUATION_CONFIG  # 新增导入
)


def run_full_pipeline(X_labeled: np.ndarray,
                     y_labeled: np.ndarray,
                     X_unlabeled: np.ndarray,
                     X_val: Optional[np.ndarray] = None,
                     y_val: Optional[np.ndarray] = None,
                     autoencoder_config: dict = None,
                     ensemble_config: dict = None,
                     semi_supervised_config: dict = None,
                     pseudo_label_config: dict = None,
                     self_training_config: dict = None,
                     optimization_config: dict = None,  # 新增参数
                     device: str = None,
                     verbose: bool = True) -> EnhancedSemiSupervisedEnsemble:
    """
    运行完整的半监督学习流程
    
    Args:
        ... (原有参数)
        optimization_config: 优化配置
        
    Returns:
        训练好的模型
    """
    # 获取配置
    opt_config = optimization_config or OPTIMIZATION_CONFIG
    ss_config = semi_supervised_config or SEMI_SUPERVISED_CONFIG
    
    # 创建模型实例（使用增强功能）
    model = EnhancedSemiSupervisedEnsemble(
        autoencoder_config=autoencoder_config,
        ensemble_config=ensemble_config,
        semi_supervised_config=semi_supervised_config,
        device=device,
        use_advanced_pseudo_labeling=ss_config.get('use_advanced_pseudo_labeling', True),
        use_improved_vae=autoencoder_config.get('use_improved_vae', True) if autoencoder_config else True
    )
    
    # Step 1: 特征工程和自编码器训练
    model.step1_feature_engineering_and_autoencoder(
        X_labeled, X_unlabeled, verbose=verbose
    )
    
    # Step 2: 初始集成模型训练
    model.step2_train_ensemble_with_encoded_features(
        X_labeled,
        y_labeled,
        X_val=X_val,
        y_val=y_val,
        verbose=verbose
    )
    
    # Step 3 & 4: 迭代自训练
    model.step4_iterative_self_training(
        X_labeled, y_labeled, X_unlabeled,
        X_val=X_val,
        y_val=y_val,
        self_training_config=self_training_config,
        pseudo_label_config=pseudo_label_config,
        verbose=verbose
    )
    
    return model


def main():
    """主函数"""
    # 记录开始时间
    total_start_time = time.time()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Semi-Supervised Learning Pipeline")
    parser.add_argument('--labeled_file', type=str, default='data/labeled_log.csv',
                       help='Path to labeled data file')
    parser.add_argument('--unlabeled_file', type=str, default='data/unlabeled.csv',
                       help='Path to unlabeled data file')
    parser.add_argument('--target_column', type=str, default='log_target_length',
                       help='Name of target column')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size ratio')
    parser.add_argument('--model_save_path', type=str, default='semi_supervised_model.pkl',
                       help='Path to save trained model')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed information')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                       help='Use GPU if available (default: True)')
    # 新增参数
    parser.add_argument('--use_advanced_pseudo_labeling', action='store_true', default=True,
                       help='Use advanced pseudo labeling strategy')
    parser.add_argument('--use_improved_vae', action='store_true', default=True,
                       help='Use improved VAE architecture with attention')
    parser.add_argument('--disable_gpu_optimization', action='store_true',
                       help='Disable GPU optimizations')
    args = parser.parse_args()
    
    # 设置设备
    if args.device is None:
        args.device = DEVICE
    
    # GPU信息
    if args.use_gpu and args.device == 'cuda':
        is_gpu_available, gpu_info = check_gpu_availability()
        if args.verbose:
            print(gpu_info)
        clear_gpu_memory()
    
    # 创建结果目录
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    
    # 加载数据
    print("Loading data...")
    X_labeled, y_labeled, X_unlabeled = load_data(
        args.labeled_file,
        args.unlabeled_file,
        args.target_column
    )


    # —— 特征标准化 ——  
    scaler = StandardScaler()
    # 把两部分拼一起再 fit
    X_all = np.vstack([X_labeled, X_unlabeled])
    X_all = scaler.fit_transform(X_all)
    # 再切回两份
    X_labeled   = X_all[:len(X_labeled)]
    X_unlabeled = X_all[len(X_labeled):]


    # 检查数据质量
    print("\nChecking data quality...")
    labeled_quality = check_data_quality(X_labeled, y_labeled, verbose=args.verbose)
    unlabeled_quality = check_data_quality(X_unlabeled, verbose=args.verbose)
    
    # 准备训练和测试数据
    print("\nPreparing data splits...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(
        X_labeled, y_labeled, test_size=args.test_size
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Unlabeled set: {len(X_unlabeled)} samples")
    
    # 更新配置（如果通过命令行参数指定）
    if args.use_advanced_pseudo_labeling:
        SEMI_SUPERVISED_CONFIG['use_advanced_pseudo_labeling'] = True
    if args.use_improved_vae:
        AUTOENCODER_CONFIG['use_improved_vae'] = True
    
    # 运行半监督学习流程
    print("\n" + "="*50)
    print("Running Semi-Supervised Learning Pipeline")
    print(f"  - Advanced Pseudo Labeling: {SEMI_SUPERVISED_CONFIG.get('use_advanced_pseudo_labeling', True)}")
    print(f"  - Improved VAE Architecture: {AUTOENCODER_CONFIG.get('use_improved_vae', True)}")
    print("="*50)
    
    model = run_full_pipeline(
        X_train,
        y_train,
        X_unlabeled,
        X_val=X_val,
        y_val=y_val,
        device=args.device,
        verbose=args.verbose,
        optimization_config=OPTIMIZATION_CONFIG  # 传递优化配置
    )
    
    # 评估模型
    print("\n" + "="*50)
    print("Evaluating Model Performance")
    print("="*50)
    
    # 在测试集上评估
    test_metrics, test_predictions, test_uncertainties = model.evaluate(X_test, y_test)
    
    # 打印评估结果
    print("\n=== Test Set Performance ===")
    print(f"RMSE: {test_metrics['ensemble']['RMSE']:.4f}")
    print(f"MAE: {test_metrics['ensemble']['MAE']:.4f}")
    print(f"R²: {test_metrics['ensemble']['R2']:.4f}")
    print(f"Mean Uncertainty: {test_metrics['uncertainty']['mean_uncertainty']:.4f}")
    
    print("\n=== Individual Model Performance ===")
    for model_name, metrics in test_metrics['individual_models'].items():
        print(f"{model_name}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
    
    # 保存结果
    print("\nSaving results...")
    
    # 保存模型
    model.save_model(args.model_save_path)
    
    # 保存评估指标
    with open(results_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    # 保存预测结果
    results_df = pd.DataFrame({
        'true_values': y_test,
        'predictions': test_predictions,
        'uncertainties': test_uncertainties
    })
    results_df.to_csv(results_dir / 'test_predictions.csv', index=False)
    
    # 生成可视化
    print("\nGenerating visualizations...")
    
    # 训练历史
    plot_training_history(
        model.training_history,
        save_path=results_dir / 'training_history.png'
    )
    
    # 预测对比图
    plot_predictions(
        y_test, test_predictions, test_uncertainties,
        title="Test Set: Predictions vs True Values",
        save_path=results_dir / 'predictions.png'
    )
    
    # 残差图
    plot_residuals(
        y_test, test_predictions,
        title="Test Set: Residual Analysis",
        save_path=results_dir / 'residuals.png'
    )
    
    # 模型对比
    plot_model_comparison(
        test_metrics,
        title="Model Performance Comparison",
        save_path=results_dir / 'model_comparison.png'
    )
    
    # 不确定性分析
    plot_uncertainty_analysis(
        y_test, test_predictions, test_uncertainties,
        title="Test Set: Uncertainty Analysis",
        save_path=results_dir / 'uncertainty_analysis.png'
    )
    
    print(f"\nAll results saved to: {results_dir}")
    
    # 打印总时间和设备信息
    total_time = time.time() - total_start_time
    print(f"\nTotal pipeline execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Device used: {args.device}")
    
    if args.device == 'cuda':
        print(f"GPU Memory Usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Peak GPU Memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    
    print("Pipeline completed successfully!")

    # 新增：打印高级伪标签生成的详细统计（如果使用）
    if hasattr(model, 'pseudo_label_generator') and model.pseudo_label_generator:
        summary = model.pseudo_label_generator.get_iteration_summary()
        if summary:
            print("\n=== Advanced Pseudo Label Generation Summary ===")
            print(f"Total iterations: {summary.get('n_iterations', 0)}")
            print(f"Total pseudo labels generated: {summary.get('total_pseudo_labels', 0)}")
            print(f"Mean selection rate: {summary.get('mean_selection_rate', 0):.3f}")
            
            # 保存高级统计信息
            with open(results_dir / 'pseudo_label_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
