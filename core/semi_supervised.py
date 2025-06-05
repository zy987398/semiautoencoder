"""
半监督学习核心类
"""
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, List, Optional, Dict, Any
import joblib

from models.vae import VAETrainer
from models.ensemble import EnsembleUncertaintyEstimator
from config import (
    SEED, DEVICE, AUTOENCODER_CONFIG, ENSEMBLE_CONFIG,
    SEMI_SUPERVISED_CONFIG, PSEUDO_LABEL_CONFIG, SELF_TRAINING_CONFIG
)


class EnhancedSemiSupervisedEnsemble:
    """增强的半监督集成学习框架"""
    
    def __init__(self, 
                 autoencoder_config: Optional[Dict] = None,
                 ensemble_config: Optional[Dict] = None,
                 semi_supervised_config: Optional[Dict] = None,
                 device: Optional[str] = None):
        """
        Args:
            autoencoder_config: 自编码器配置
            ensemble_config: 集成模型配置
            semi_supervised_config: 半监督学习配置
            device: 计算设备
        """
        # 使用提供的配置或默认配置
        self.autoencoder_config = autoencoder_config or AUTOENCODER_CONFIG
        self.ensemble_config = ensemble_config or ENSEMBLE_CONFIG
        self.semi_supervised_config = semi_supervised_config or SEMI_SUPERVISED_CONFIG
        self.device = device or DEVICE
        
        # 特征处理器
        self.feature_scaler = RobustScaler()
        self.target_scaler = StandardScaler()
        self.poly_features = PolynomialFeatures(
            degree=self.semi_supervised_config['poly_degree'],
            interaction_only=self.semi_supervised_config['poly_interaction_only'],
            include_bias=self.semi_supervised_config['poly_include_bias']
        )
        
        # 模型组件
        self.autoencoder_trainer = None
        self.ensemble_estimator = None
        
        # 训练历史
        self.training_history = {
            'autoencoder_loss': [],
            'ensemble_performance': [],
            'pseudo_label_stats': []
        }
        
    def step1_feature_engineering_and_autoencoder(self, 
                                                 X_labeled: np.ndarray,
                                                 X_unlabeled: np.ndarray,
                                                 verbose: bool = True) -> None:
        """
        步骤1: 特征工程和自编码器训练
        
        Args:
            X_labeled: 标记数据特征
            X_unlabeled: 未标记数据特征
            verbose: 是否打印信息
        """
        if verbose:
            print("=== Step 1: Feature Engineering & VAE Training ===")
        
        # 特征工程：多项式特征
        X_labeled_poly = self.poly_features.fit_transform(X_labeled)
        X_unlabeled_poly = self.poly_features.transform(X_unlabeled)
        
        # 合并所有特征数据
        X_all = np.vstack([X_labeled_poly, X_unlabeled_poly])
        X_scaled = self.feature_scaler.fit_transform(X_all)
        
        # 初始化和训练自编码器
        input_dim = X_scaled.shape[1]
        self.autoencoder_trainer = VAETrainer(
            input_dim=input_dim,
            latent_dims=self.autoencoder_config['latent_dims'],
            dropout_rate=self.autoencoder_config['dropout_rate'],
            device=self.device,
            kl_weight=self.autoencoder_config.get('kl_weight', 1e-3)
        )
        
        # 训练自编码器
        history = self.autoencoder_trainer.train(
            X_train=X_scaled,
            epochs=self.autoencoder_config['epochs'],
            batch_size=self.autoencoder_config['batch_size'],
            learning_rate=self.autoencoder_config['learning_rate'],
            weight_decay=self.autoencoder_config['weight_decay'],
            scheduler_patience=self.autoencoder_config['scheduler_patience'],
            scheduler_factor=self.autoencoder_config['scheduler_factor'],
            verbose=verbose
        )
        
        self.training_history['autoencoder_loss'] = history
        
    def step2_train_ensemble_with_encoded_features(self,
                                                  X_labeled: np.ndarray,
                                                  y_labeled: np.ndarray,
                                                  verbose: bool = True) -> None:
        """
        步骤2: 使用编码特征训练集成模型
        
        Args:
            X_labeled: 标记数据特征
            y_labeled: 标记数据目标
            verbose: 是否打印信息
        """
        if verbose:
            print("=== Step 2: Training Ensemble with Encoded Features ===")
        
        # 特征预处理
        X_labeled_poly = self.poly_features.transform(X_labeled)
        X_scaled = self.feature_scaler.transform(X_labeled_poly)
        y_scaled = self.target_scaler.fit_transform(y_labeled.reshape(-1, 1)).ravel()
        
        # 获取编码特征
        encoded_features = self.autoencoder_trainer.encode(X_scaled)
        
        # 合并原始特征和编码特征
        X_combined = np.hstack([X_scaled, encoded_features])
        
        # 划分训练验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_combined, y_scaled, 
            test_size=self.semi_supervised_config['validation_split'], 
            random_state=SEED
        )
        
        # 初始化集成估计器
        self.ensemble_estimator = EnsembleUncertaintyEstimator(
            models_config=self.ensemble_config,
            n_cv_folds=self.semi_supervised_config['n_cv_folds'],
            use_gpu=(self.device == 'cuda')
        )
        
        # 训练集成模型
        self.ensemble_estimator.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=verbose)
        
        # 评估性能
        val_pred, val_uncertainty, _ = self.ensemble_estimator.predict_with_uncertainty(X_val)
        val_pred_original = self.target_scaler.inverse_transform(val_pred.reshape(-1, 1)).ravel()
        y_val_original = self.target_scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()
        
        val_rmse = np.sqrt(mean_squared_error(y_val_original, val_pred_original))
        val_r2 = r2_score(y_val_original, val_pred_original)
        
        performance = {
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'mean_uncertainty': np.mean(val_uncertainty)
        }
        
        self.training_history['ensemble_performance'].append(performance)
        
        if verbose:
            print(f"\nEnsemble training completed:")
            print(f"  - Validation RMSE: {val_rmse:.4f}")
            print(f"  - Validation R²: {val_r2:.4f}")
            print(f"  - Mean Uncertainty: {np.mean(val_uncertainty):.4f}")
    
    def step3_generate_high_quality_pseudo_labels(self,
                                                 X_unlabeled: np.ndarray,
                                                 pseudo_label_config: Optional[Dict] = None,
                                                 verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        步骤3: 生成高质量伪标签
        
        Args:
            X_unlabeled: 未标记数据特征
            pseudo_label_config: 伪标签配置
            verbose: 是否打印信息
            
        Returns:
            pseudo_features: 伪标签样本特征
            pseudo_targets: 伪标签目标值
            pseudo_uncertainties: 伪标签不确定性
        """
        if verbose:
            print("=== Step 3: Generating High-Quality Pseudo Labels ===")
        
        # 使用提供的配置或默认配置
        config = pseudo_label_config or PSEUDO_LABEL_CONFIG
        
        # 特征预处理
        X_unlabeled_poly = self.poly_features.transform(X_unlabeled)
        X_scaled = self.feature_scaler.transform(X_unlabeled_poly)
        
        # 获取编码特征和重建误差
        encoded_features, recon_errors = self.autoencoder_trainer.encode_and_get_errors(X_scaled)
        
        # 合并特征
        X_combined = np.hstack([X_scaled, encoded_features])
        
        # 集成预测
        predictions, uncertainties, individual_preds = self.ensemble_estimator.predict_with_uncertainty(X_combined)
        
        # 多重过滤策略
        filters = {}
        
        # 过滤器1: 不确定性过滤
        uncertainty_threshold = np.percentile(uncertainties, config['confidence_threshold'] * 100)
        filters['uncertainty'] = uncertainties <= uncertainty_threshold
        
        # 过滤器2: 重建误差过滤
        recon_threshold = np.percentile(recon_errors, config['reconstruction_threshold'] * 100)
        filters['reconstruction'] = recon_errors <= recon_threshold
        
        # 过滤器3: 集成一致性过滤
        pred_matrix = np.array(list(individual_preds.values())).T
        pred_std = np.std(pred_matrix, axis=1)
        pred_mean = np.mean(pred_matrix, axis=1)
        consistency_scores = 1 - (pred_std / (np.abs(pred_mean) + 1e-8))
        consistency_threshold = np.percentile(consistency_scores, config['ensemble_agreement_threshold'] * 100)
        filters['consistency'] = consistency_scores >= consistency_threshold
        
        # 过滤器4: 预测范围过滤（避免极端值）
        pred_q1, pred_q3 = np.percentile(predictions, [25, 75])
        iqr = pred_q3 - pred_q1
        iqr_multiplier = config.get('iqr_multiplier', 1.5)
        lower_bound = pred_q1 - iqr_multiplier * iqr
        upper_bound = pred_q3 + iqr_multiplier * iqr
        filters['range'] = (predictions >= lower_bound) & (predictions <= upper_bound)
        
        # 合并所有过滤器
        final_mask = np.ones(len(X_unlabeled), dtype=bool)
        for filter_name, mask in filters.items():
            final_mask &= mask
        
        # 获取伪标签
        pseudo_features = X_unlabeled[final_mask]
        pseudo_targets = self.target_scaler.inverse_transform(predictions[final_mask].reshape(-1, 1)).ravel()
        pseudo_uncertainties = uncertainties[final_mask]
        
        # 统计信息
        stats = {
            'total_unlabeled': len(X_unlabeled),
            'uncertainty_filtered': np.sum(filters['uncertainty']),
            'reconstruction_filtered': np.sum(filters['reconstruction']),
            'consistency_filtered': np.sum(filters['consistency']),
            'range_filtered': np.sum(filters['range']),
            'final_pseudo_labels': len(pseudo_features),
            'mean_uncertainty': np.mean(pseudo_uncertainties) if len(pseudo_uncertainties) > 0 else 0,
            'mean_consistency': np.mean(consistency_scores[final_mask]) if np.sum(final_mask) > 0 else 0,
            'coverage_ratio': len(pseudo_features) / len(X_unlabeled) if len(X_unlabeled) > 0 else 0
        }
        
        self.training_history['pseudo_label_stats'].append(stats)
        
        if verbose:
            print(f"Pseudo label generation completed:")
            print(f"  - Original unlabeled samples: {stats['total_unlabeled']}")
            print(f"  - After uncertainty filter: {stats['uncertainty_filtered']}")
            print(f"  - After reconstruction filter: {stats['reconstruction_filtered']}")
            print(f"  - After consistency filter: {stats['consistency_filtered']}")
            print(f"  - After range filter: {stats['range_filtered']}")
            print(f"  - Final pseudo labels: {stats['final_pseudo_labels']}")
            print(f"  - Coverage ratio: {stats['coverage_ratio']:.3f}")
            print(f"  - Mean uncertainty: {stats['mean_uncertainty']:.4f}")
        
        return pseudo_features, pseudo_targets, pseudo_uncertainties
    
    def step4_iterative_self_training(self,
                                     X_labeled: np.ndarray,
                                     y_labeled: np.ndarray,
                                     X_unlabeled: np.ndarray,
                                     self_training_config: Optional[Dict] = None,
                                     pseudo_label_config: Optional[Dict] = None,
                                     verbose: bool = True) -> None:
        """
        步骤4: 迭代自训练
        
        Args:
            X_labeled: 标记数据特征
            y_labeled: 标记数据目标
            X_unlabeled: 未标记数据特征
            self_training_config: 自训练配置
            pseudo_label_config: 伪标签配置
            verbose: 是否打印信息
        """
        if verbose:
            print("=== Step 4: Iterative Self-Training ===")
        
        # 使用提供的配置或默认配置
        st_config = self_training_config or SELF_TRAINING_CONFIG
        pl_config = pseudo_label_config or PSEUDO_LABEL_CONFIG
        
        current_X_labeled = X_labeled.copy()
        current_y_labeled = y_labeled.copy()
        current_X_unlabeled = X_unlabeled.copy()
        
        for iteration in range(st_config['n_iterations']):
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{st_config['n_iterations']} ---")
            
            # 生成伪标签
            pseudo_X, pseudo_y, pseudo_uncertainties = self.step3_generate_high_quality_pseudo_labels(
                current_X_unlabeled, pl_config, verbose=verbose
            )
            
            if len(pseudo_X) < pl_config['min_pseudo_labels']:
                if verbose:
                    print(f"Generated only {len(pseudo_X)} pseudo labels, which is below minimum threshold {pl_config['min_pseudo_labels']}. Stopping iterations.")
                break
            
            # 基于不确定性的样本权重
            sample_weights = 1.0 / (1.0 + pseudo_uncertainties)
            sample_weights = sample_weights / np.sum(sample_weights) * len(pseudo_X)
            
            # 扩展训练集
            expanded_X = np.vstack([current_X_labeled, pseudo_X])
            expanded_y = np.hstack([current_y_labeled, pseudo_y])
            
            # 重新训练集成模型
            if verbose:
                print("Retraining ensemble with expanded dataset...")
            self.step2_train_ensemble_with_encoded_features(expanded_X, expanded_y, verbose=verbose)
            
            # 更新当前标记数据集（选择性地添加高置信度伪标签）
            high_confidence_percentile = st_config.get('high_confidence_percentile', 50)
            high_confidence_mask = pseudo_uncertainties <= np.percentile(pseudo_uncertainties, high_confidence_percentile)
            
            if np.sum(high_confidence_mask) > 0:
                current_X_labeled = np.vstack([current_X_labeled, pseudo_X[high_confidence_mask]])
                current_y_labeled = np.hstack([current_y_labeled, pseudo_y[high_confidence_mask]])
            
            # 移除已使用的伪标签样本
            if len(current_X_unlabeled) > len(pseudo_X):
                remaining_indices = np.random.choice(
                    len(current_X_unlabeled),
                    size=len(current_X_unlabeled) - len(pseudo_X),
                    replace=False
                )
                current_X_unlabeled = current_X_unlabeled[remaining_indices]
            else:
                current_X_unlabeled = current_X_unlabeled[:len(current_X_unlabeled)//2]
            
            if verbose:
                print(f"Added {len(pseudo_X)} pseudo labels. Remaining unlabeled: {len(current_X_unlabeled)}")
                print(f"Current labeled dataset size: {len(current_y_labeled)}")
    
    def predict(self, X: np.ndarray, return_uncertainty: bool = False) -> Any:
        """
        预测新样本
        
        Args:
            X: 预测特征
            return_uncertainty: 是否返回不确定性
            
        Returns:
            predictions: 预测结果
            uncertainties: 不确定性（如果return_uncertainty=True）
            individual_preds: 各模型预测（如果return_uncertainty=True）
        """
        # 特征预处理
        X_poly = self.poly_features.transform(X)
        X_scaled = self.feature_scaler.transform(X_poly)
        
        # 获取编码特征
        encoded_features = self.autoencoder_trainer.encode(X_scaled)
        
        # 合并特征
        X_combined = np.hstack([X_scaled, encoded_features])
        
        # 集成预测
        predictions, uncertainties, individual_preds = self.ensemble_estimator.predict_with_uncertainty(X_combined)
        
        # 逆转换
        predictions = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
        
        if return_uncertainty:
            return predictions, uncertainties, individual_preds
        return predictions
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """
        评估模型性能
        
        Args:
            X_test: 测试特征
            y_test: 测试目标
            
        Returns:
            metrics: 评估指标
            predictions: 预测结果
            uncertainties: 不确定性
        """
        predictions, uncertainties, individual_preds = self.predict(X_test, return_uncertainty=True)
        
        # 计算各种指标
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        # 不确定性统计
        uncertainty_stats = {
            'mean_uncertainty': np.mean(uncertainties),
            'std_uncertainty': np.std(uncertainties),
            'uncertainty_coverage': np.mean(np.abs(y_test - predictions) <= uncertainties)
        }
        
        # 个体模型性能
        individual_metrics = {}
        for model_name, preds in individual_preds.items():
            preds_original = self.target_scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
            individual_metrics[model_name] = {
                'rmse': np.sqrt(mean_squared_error(y_test, preds_original)),
                'r2': r2_score(y_test, preds_original)
            }
        
        metrics = {
            'ensemble': {
                'MSE': mse,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            },
            'uncertainty': uncertainty_stats,
            'individual_models': individual_metrics
        }
        
        return metrics, predictions, uncertainties
    
    def save_model(self, filepath: str) -> None:
        """保存模型"""
        model_data = {
            'autoencoder_state_dict': self.autoencoder_trainer.model.state_dict(),
            'autoencoder_config': self.autoencoder_config,
            'ensemble_estimator': self.ensemble_estimator,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'poly_features': self.poly_features,
            'ensemble_config': self.ensemble_config,
            'semi_supervised_config': self.semi_supervised_config,
            'training_history': self.training_history,
            'device': self.device
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """加载模型"""
        model_data = joblib.load(filepath)
        
        # 加载配置
        self.autoencoder_config = model_data['autoencoder_config']
        self.ensemble_config = model_data['ensemble_config']
        self.semi_supervised_config = model_data['semi_supervised_config']
        self.device = model_data.get('device', self.device)
        
        # 加载特征处理器
        self.feature_scaler = model_data['feature_scaler']
        self.target_scaler = model_data['target_scaler']
        self.poly_features = model_data['poly_features']
        
        # 重建自编码器
        input_dim = self.poly_features.n_output_features_
        self.autoencoder_trainer = VAETrainer(
            input_dim=input_dim,
            latent_dims=self.autoencoder_config['latent_dims'],
            dropout_rate=self.autoencoder_config['dropout_rate'],
            device=self.device,
            kl_weight=self.autoencoder_config.get('kl_weight', 1e-3)
        )
        self.autoencoder_trainer.model.load_state_dict(model_data['autoencoder_state_dict'])
        
        # 加载集成估计器
        self.ensemble_estimator = model_data['ensemble_estimator']
        
        # 加载训练历史
        self.training_history = model_data['training_history']
        
        print(f"Model loaded from {filepath}")