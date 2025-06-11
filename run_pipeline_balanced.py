"""
快速修复脚本 - 处理极度数据不平衡问题
保存为: run_pipeline_balanced.py
"""
import numpy as np
import pandas as pd
import torch
import gc
import warnings
import argparse

warnings.filterwarnings('ignore')

# 使用优化的配置
import sys
sys.path.insert(0, '.')  # 确保能导入config_optimized
from config_optimized import *

# 添加关键的修改
SAMPLING_CONFIG = {
    'max_unlabeled_samples': 10000,  # 只使用1万个未标记样本
    'augment_labeled_factor': 3,      # 将标记数据增强3倍
    'use_gpu': True,                  # 使用GPU
}


def parse_args():
    """Parse command line options"""
    parser = argparse.ArgumentParser(
        description="Balanced semi-supervised learning pipeline")
    parser.add_argument(
        '--use_improved_vae', action='store_true',
        help='Use the improved VAE architecture')
    parser.add_argument(
        '--use_advanced_pseudo_labeling', action='store_true',
        help='Enable advanced pseudo label generation')
    return parser.parse_args()

def main():
    args = parse_args()
    print("="*60)
    print("BALANCED PIPELINE FOR SUBSURFACE CRACK PREDICTION")
    print("="*60)
    
    # 1. 加载数据
    print("\n1. Loading data...")
    labeled_file = 'data/labeled_log.csv'
    unlabeled_file = 'data/unlabeled.csv'
    
    # 读取标记数据
    df_labeled = pd.read_csv(labeled_file)
    X_labeled = df_labeled.drop('log_target_length', axis=1).values
    y_labeled = df_labeled['log_target_length'].values
    
    print(f"   Labeled samples: {len(X_labeled)}")
    
    # 2. 智能采样未标记数据
    print("\n2. Smart sampling unlabeled data...")
    
    # 分块读取未标记数据
    chunk_size = 50000
    sampled_data = []
    
    for chunk in pd.read_csv(unlabeled_file, chunksize=chunk_size):
        # 随机采样每个块
        n_sample = min(2000, len(chunk))
        sample_indices = np.random.choice(len(chunk), n_sample, replace=False)
        sampled_data.append(chunk.iloc[sample_indices].values)
        
        if len(sampled_data) * 2000 >= SAMPLING_CONFIG['max_unlabeled_samples']:
            break
    
    X_unlabeled = np.vstack(sampled_data)[:SAMPLING_CONFIG['max_unlabeled_samples']]
    print(f"   Sampled unlabeled: {len(X_unlabeled)}")
    
    # 3. 数据增强（因为标记数据太少）
    print("\n3. Augmenting labeled data...")
    X_labeled_list = [X_labeled]
    y_labeled_list = [y_labeled]
    
    for i in range(SAMPLING_CONFIG['augment_labeled_factor'] - 1):
        # 添加轻微噪声
        noise = np.random.normal(0, 0.01, X_labeled.shape)
        X_labeled_list.append(X_labeled + noise)
        y_labeled_list.append(y_labeled)
    
    X_labeled_aug = np.vstack(X_labeled_list)
    y_labeled_aug = np.hstack(y_labeled_list)
    print(f"   Augmented labeled: {len(X_labeled_aug)}")
    
    # 4. 特征标准化
    print("\n4. Standardizing features...")
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    
    # 合并所有数据来拟合scaler
    X_all = np.vstack([X_labeled_aug, X_unlabeled])
    scaler.fit(X_all)
    
    X_labeled_scaled = scaler.transform(X_labeled_aug)
    X_unlabeled_scaled = scaler.transform(X_unlabeled)
    
    # 5. 准备训练/验证/测试集
    print("\n5. Preparing data splits...")
    from sklearn.model_selection import train_test_split
    
    # 使用原始（未增强）数据作为测试集
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_labeled, y_labeled, test_size=0.2, random_state=SEED
    )
    
    # 从增强数据中创建训练和验证集
    train_indices = []
    val_indices = []
    
    for i in range(len(X_train_full)):
        # 原始样本用于训练
        train_indices.append(i)
        # 增强样本用于训练和验证
        for j in range(1, SAMPLING_CONFIG['augment_labeled_factor']):
            idx = i + j * len(X_labeled)
            if j == 1:
                val_indices.append(idx)
            else:
                train_indices.append(idx)
    
    X_train = X_labeled_scaled[train_indices]
    y_train = y_labeled_aug[train_indices]
    X_val = X_labeled_scaled[val_indices]
    y_val = y_labeled_aug[val_indices]
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 6. 设置GPU（如果可用）
    device = 'cpu'  # 默认CPU
    if SAMPLING_CONFIG['use_gpu'] and torch.cuda.is_available():
        device = 'cuda'
        print(f"\n6. Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        print("\n6. Using CPU")

    # 根据命令行参数更新配置
    if args.use_improved_vae:
        AUTOENCODER_CONFIG['use_improved_vae'] = True
    if args.use_advanced_pseudo_labeling:
        SEMI_SUPERVISED_CONFIG['use_advanced_pseudo_labeling'] = True
    
    # 7. 运行半监督学习
    print("\n7. Running semi-supervised learning...")
    from core.semi_supervised import EnhancedSemiSupervisedEnsemble
    
    # 修改配置以使用GPU
    if device == 'cuda':
        AUTOENCODER_CONFIG['device'] = 'cuda'
        # LightGBM GPU支持需要特殊编译，暂时保持CPU
    
    model = EnhancedSemiSupervisedEnsemble(
        autoencoder_config=AUTOENCODER_CONFIG,
        ensemble_config=ENSEMBLE_CONFIG,
        semi_supervised_config=SEMI_SUPERVISED_CONFIG,
        device=device,
        use_advanced_pseudo_labeling=args.use_advanced_pseudo_labeling,
        use_improved_vae=args.use_improved_vae
    )
    
    # Step 1: VAE训练
    print("\n   Step 1: Training VAE...")
    model.step1_feature_engineering_and_autoencoder(
        X_train, X_unlabeled_scaled, verbose=True
    )
    
    # 清理内存
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # Step 2: 集成模型训练
    print("\n   Step 2: Training ensemble...")
    model.step2_train_ensemble_with_encoded_features(
        X_train, y_train, X_val, y_val, verbose=True
    )
    
    # Step 3 & 4: 迭代自训练（简化版）
    print("\n   Step 3&4: Self-training...")
    
    # 只进行一次迭代，使用部分未标记数据
    pseudo_X, pseudo_y, _ = model.step3_generate_high_quality_pseudo_labels(
        X_unlabeled_scaled[:5000],  # 只使用5000个样本生成伪标签
        PSEUDO_LABEL_CONFIG,
        verbose=True
    )
    
    if len(pseudo_X) > 0:
        print(f"\n   Generated {len(pseudo_X)} pseudo labels")
        
        # 选择最置信的伪标签
        n_add = min(len(pseudo_X), 50)  # 最多添加50个
        X_combined = np.vstack([X_train, pseudo_X[:n_add]])
        y_combined = np.hstack([y_train, pseudo_y[:n_add]])
        
        # 重新训练
        print("   Retraining with pseudo labels...")
        model.step2_train_ensemble_with_encoded_features(
            X_combined, y_combined, X_val, y_val, verbose=False
        )
    
    # 8. 评估模型
    print("\n8. Evaluating model...")
    
    # 需要先对测试集进行预处理
    X_test_scaled = scaler.transform(X_test)
    
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    
    predictions = model.predict(X_test_scaled)
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"\nTest Performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # 9. 保存模型
    print("\n9. Saving model...")
    model.save_model('model_balanced.pkl')
    
    # 10. 内存使用总结
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024**2
    print(f"\nFinal memory usage: {memory_mb:.1f} MB")
    
    print("\nPipeline completed successfully!")
    
    # 可选：对所有未标记数据进行批量预测
    print("\n" + "="*60)
    print("Optional: Batch predict all unlabeled data? (y/n): ", end='')
    if input().lower() == 'y':
        batch_predict_all_unlabeled(model, unlabeled_file, scaler)


def batch_predict_all_unlabeled(model, unlabeled_file, scaler, 
                               output_file='predictions_all.csv'):
    """批量预测所有未标记数据"""
    print("\nBatch predicting all unlabeled data...")
    
    chunk_size = 10000
    predictions_list = []
    uncertainties_list = []
    
    total_chunks = sum(1 for _ in pd.read_csv(unlabeled_file, chunksize=chunk_size))
    
    for i, chunk in enumerate(pd.read_csv(unlabeled_file, chunksize=chunk_size)):
        # 预处理
        X_chunk = chunk.values
        X_chunk_scaled = scaler.transform(X_chunk)
        
        # 预测
        try:
            pred, unc, _ = model.predict(X_chunk_scaled, return_uncertainty=True)
            predictions_list.append(pred)
            uncertainties_list.append(unc)
        except:
            # 如果失败，使用简单预测
            pred = model.predict(X_chunk_scaled)
            predictions_list.append(pred)
            uncertainties_list.append(np.zeros_like(pred))
        
        # 进度
        progress = (i + 1) / total_chunks * 100
        print(f"  Progress: {progress:.1f}%", end='\r')
        
        # 定期清理内存
        if i % 10 == 0:
            gc.collect()
    
    print("\n  Saving predictions...")
    
    # 合并结果
    all_predictions = np.concatenate(predictions_list)
    all_uncertainties = np.concatenate(uncertainties_list)
    
    # 保存结果
    results_df = pd.DataFrame({
        'prediction': all_predictions,
        'uncertainty': all_uncertainties
    })
    results_df.to_csv(output_file, index=False)
    
    print(f"  Predictions saved to: {output_file}")
    print(f"  Total predictions: {len(results_df):,}")


if __name__ == "__main__":
    main()
