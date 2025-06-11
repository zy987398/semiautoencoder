"""
半监督学习核心类
"""
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
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
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import entropy
# 导入高级伪标签生成器
from core.advanced_pseudo_labeling import AdvancedPseudoLabelGenerator

try:
    from .advanced_pseudo_labeling import AdvancedPseudoLabelGenerator
    ADVANCED_PSEUDO_LABELING_AVAILABLE = True
except ImportError:
    ADVANCED_PSEUDO_LABELING_AVAILABLE = False
    print("Warning: Advanced pseudo labeling module not found. Using standard method.")


class EnhancedSemiSupervisedEnsemble:
    """增强的半监督集成学习框架"""
    
    def __init__(self, 
                 autoencoder_config: Optional[Dict] = None,
                 ensemble_config: Optional[Dict] = None,
                 semi_supervised_config: Optional[Dict] = None,
                 device: Optional[str] = None,
                 use_advanced_pseudo_labeling: bool = True,  # 新增参数
                 use_improved_vae: bool = True):             # 新增参数
        """
        Args:
            autoencoder_config: 自编码器配置
            ensemble_config: 集成模型配置
            semi_supervised_config: 半监督学习配置
            device: 计算设备
            use_advanced_pseudo_labeling: 是否使用高级伪标签生成
            use_improved_vae: 是否使用改进的VAE架构
        """
        # 优先使用传入的配置文件，其次使用默认配置
        self.autoencoder_config = AUTOENCODER_CONFIG if autoencoder_config is None else autoencoder_config
        self.ensemble_config = ENSEMBLE_CONFIG if ensemble_config is None else ensemble_config
        self.semi_supervised_config = SEMI_SUPERVISED_CONFIG if semi_supervised_config is None else semi_supervised_config
        self.device = DEVICE if device is None else device
                
        # 新增：配置选项
        self.use_advanced_pseudo_labeling = use_advanced_pseudo_labeling and ADVANCED_PSEUDO_LABELING_AVAILABLE
        self.use_improved_vae = use_improved_vae
        
        # 特征处理器
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        if self.semi_supervised_config['poly_degree'] > 1:
            self.poly_features = PolynomialFeatures(
                degree=self.semi_supervised_config['poly_degree'],
                interaction_only=self.semi_supervised_config['poly_interaction_only'],
                include_bias=self.semi_supervised_config['poly_include_bias']
            )
        else:
            self.poly_features = None

        # 降维器用于编码特征
        self.pca = PCA(n_components=0.95)
        
        # 模型组件
        self.autoencoder_trainer = None
        self.ensemble_estimator = None
        
        # 新增：高级伪标签生成器
        if self.use_advanced_pseudo_labeling:
            try:
                self.pseudo_label_generator = AdvancedPseudoLabelGenerator(
                    base_config=PSEUDO_LABEL_CONFIG,
                    use_active_learning=self.semi_supervised_config.get('use_active_learning', True),
                    use_clustering=self.semi_supervised_config.get('use_clustering', True)
                )
            except Exception as e:
                print(f"Failed to initialize advanced pseudo label generator: {e}")
                self.pseudo_label_generator = None
                self.use_advanced_pseudo_labeling = False
        else:
            self.pseudo_label_generator = None
        
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
        
        修改：使用改进的VAE架构
        """
        if verbose:
            print("=== Step 1: Feature Engineering & VAE Training ===")
        
        # 特征工程：多项式特征（可选）
        if self.poly_features is not None:
            X_labeled_poly = self.poly_features.fit_transform(X_labeled)
            X_unlabeled_poly = self.poly_features.transform(X_unlabeled)
        else:
            X_labeled_poly = X_labeled
            X_unlabeled_poly = X_unlabeled
        
        # 合并所有特征数据
        X_all = np.vstack([X_labeled_poly, X_unlabeled_poly])
        X_scaled = self.feature_scaler.fit_transform(X_all)
        
        # 初始化和训练自编码器（使用改进版本）
        input_dim = X_scaled.shape[1]

        # 准备VAE配置
        vae_kwargs = {
            'input_dim': input_dim,
            'latent_dims': self.autoencoder_config['latent_dims'],
            'dropout_rate': self.autoencoder_config['dropout_rate'],
            'device': self.device,
            'kl_weight': self.autoencoder_config.get('kl_weight', 1e-3)
        }

        # 如果VAETrainer支持新参数，则添加
        try:
            # 尝试使用新参数
            self.autoencoder_trainer = VAETrainer(
                **vae_kwargs,
                use_improved_vae=self.use_improved_vae,
                use_attention=self.autoencoder_config.get('use_attention', True),
                use_residual=self.autoencoder_config.get('use_residual', True)
            )
        except TypeError:
            # 如果新参数不被支持，使用原始方式
            if verbose:
                print("Using standard VAE (improved version not available)")
            self.autoencoder_trainer = VAETrainer(**vae_kwargs)
        
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

        # 使用所有数据的编码特征来拟合PCA，用于后续降维
        encoded_all = self.autoencoder_trainer.encode(X_scaled)
        self.pca.fit(encoded_all)
        
    def step2_train_ensemble_with_encoded_features(self,
                                                  X_labeled: np.ndarray,
                                                  y_labeled: np.ndarray,
                                                  X_val: Optional[np.ndarray] = None,
                                                  y_val: Optional[np.ndarray] = None,
                                                  sample_weights: Optional[np.ndarray] = None,
                                                  verbose: bool = True) -> None:
        """
        步骤2: 使用编码特征训练集成模型
        
        Args:
            X_labeled: 标记数据特征
            y_labeled: 标记数据目标
            sample_weights: 样本权重（可选）
            verbose: 是否打印信息
        """
        if verbose:
            print("=== Step 2: Training Ensemble with Encoded Features ===")
        
        # 特征预处理
        if self.poly_features is not None:
            X_labeled_poly = self.poly_features.transform(X_labeled)
        else:
            X_labeled_poly = X_labeled
        X_scaled = self.feature_scaler.transform(X_labeled_poly)
        y_scaled = self.target_scaler.fit_transform(y_labeled.reshape(-1, 1)).ravel()

        # 获取编码特征
        encoded_features = self.autoencoder_trainer.encode(X_scaled)
        encoded_features = self.pca.transform(encoded_features)
        
        # 合并原始特征和编码特征
        X_combined = np.hstack([X_scaled, encoded_features])
        
        # 划分训练验证集
        if X_val is None or y_val is None:
            if sample_weights is not None:
                X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
                    X_combined, y_scaled, sample_weights,
                    test_size=self.semi_supervised_config['validation_split'],
                    random_state=SEED
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_combined, y_scaled,
                    test_size=self.semi_supervised_config['validation_split'],
                    random_state=SEED
                )
                sw_train = None
                sw_val = None
        else:
            # 外部提供验证集时，直接使用
            X_train, y_train = X_combined, y_scaled
            if self.poly_features is not None:
                X_val_proc = self.poly_features.transform(X_val)
            else:
                X_val_proc = X_val
            X_val_proc = self.feature_scaler.transform(X_val_proc)
            val_encoded = self.autoencoder_trainer.encode(X_val_proc)
            val_encoded = self.pca.transform(val_encoded)
            X_val = np.hstack([X_val_proc, val_encoded])
            y_val = self.target_scaler.transform(y_val.reshape(-1, 1)).ravel()
            sw_train = sample_weights if sample_weights is not None else None
            sw_val = None
        
        # 初始化集成估计器
        self.ensemble_estimator = EnsembleUncertaintyEstimator(
            models_config=self.ensemble_config,
            n_cv_folds=self.semi_supervised_config['n_cv_folds'],
            use_gpu=(self.device == 'cuda')
        )
        
        # 训练集成模型
        self.ensemble_estimator.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            verbose=verbose,
            sample_weight=sw_train
        )
        
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
                                                verbose: bool = True,
                                                iteration: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        步骤3: 生成高质量伪标签
        
        Args:
            X_unlabeled: 未标记数据特征
            pseudo_label_config: 伪标签配置
            verbose: 是否打印信息
            iteration: 当前迭代次数（用于自适应阈值）
            
        Returns:
            pseudo_features: 伪标签样本特征
            pseudo_targets: 伪标签目标值
            pseudo_uncertainties: 伪标签不确定性
        """
        if verbose:
            print("=== Step 3: Generating High-Quality Pseudo Labels ===")
        
        # 使用高级伪标签生成器（如果可用）
        if self.use_advanced_pseudo_labeling and self.pseudo_label_generator:
            try:
                # 定义特征处理函数
                def feature_processor(X):
                    if self.poly_features is not None:
                        X_poly = self.poly_features.transform(X)
                    else:
                        X_poly = X
                    return self.feature_scaler.transform(X_poly)
                
                # 使用高级生成器
                pseudo_features, pseudo_targets_scaled, pseudo_weights, raw_unc, stats = \
                    self.pseudo_label_generator.generate_pseudo_labels(
                        X_unlabeled=X_unlabeled,
                        ensemble_estimator=self.ensemble_estimator,
                        autoencoder_trainer=self.autoencoder_trainer,
                        feature_processor=feature_processor,
                        iteration=iteration
                    )
                
                # 逆转换目标值
                pseudo_targets = self.target_scaler.inverse_transform(
                    pseudo_targets_scaled.reshape(-1, 1)
                ).ravel()
                
                # 使用返回的不确定度
                pseudo_uncertainties = raw_unc
                
                # 添加统计信息
                self.training_history['pseudo_label_stats'].append(stats)
                
                if verbose:
                    print(f"Advanced pseudo label generation completed:")
                    print(f"  - Selected samples: {stats['selected_samples']}")
                    print(f"  - Selection rate: {stats['selection_rate']:.3f}")
                    print(f"  - Mean quality score: {stats['mean_quality_score']:.4f}")
                    print(f"  - Mean uncertainty: {stats['mean_uncertainty']:.4f}")
                
                return pseudo_features, pseudo_targets, pseudo_uncertainties
                
            except Exception as e:
                if verbose:
                    print(f"Advanced pseudo labeling failed: {e}")
                    print("Falling back to standard method...")
        
        # 使用原始方法（以下是您原始文件中的代码）
        config = pseudo_label_config or PSEUDO_LABEL_CONFIG
        
        # 特征预处理
        if self.poly_features is not None:
            X_unlabeled_poly = self.poly_features.transform(X_unlabeled)
        else:
            X_unlabeled_poly = X_unlabeled
        X_scaled = self.feature_scaler.transform(X_unlabeled_poly)

        # 获取编码特征和重建误差
        encoded_features, recon_errors = self.autoencoder_trainer.encode_and_get_errors(X_scaled)
        encoded_features = self.pca.transform(encoded_features)
        
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
        selected_indices = np.where(final_mask)[0]
        
        # 获取伪标签
        pseudo_features = X_unlabeled[selected_indices]
        pseudo_targets_raw = predictions[selected_indices]
        pseudo_uncertainties = uncertainties[selected_indices]

        # 仅选择最置信的 top-k 样本
        if len(pseudo_features) > 0:
            k = int(min(0.3 * len(pseudo_features), 100))
            if k > 0:
                idx = np.argsort(pseudo_uncertainties)[:k]
                pseudo_features = pseudo_features[idx]
                pseudo_targets_raw = pseudo_targets_raw[idx]
                pseudo_uncertainties = pseudo_uncertainties[idx]

        pseudo_targets = self.target_scaler.inverse_transform(pseudo_targets_raw.reshape(-1, 1)).ravel()
        
        # 统计信息
        stats = {
            'total_unlabeled': len(X_unlabeled),
            'uncertainty_filtered': np.sum(filters['uncertainty']),
            'reconstruction_filtered': np.sum(filters['reconstruction']),
            'consistency_filtered': np.sum(filters['consistency']),
            'range_filtered': np.sum(filters['range']),
            'final_pseudo_labels': len(pseudo_features),
            'mean_uncertainty': np.mean(pseudo_uncertainties) if len(pseudo_uncertainties) > 0 else 0,
            'mean_consistency': np.mean(consistency_scores[selected_indices]) if len(selected_indices) > 0 else 0,
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
                                     X_val: Optional[np.ndarray] = None,
                                     y_val: Optional[np.ndarray] = None,
                                     self_training_config: Optional[Dict] = None,
                                     pseudo_label_config: Optional[Dict] = None,
                                     verbose: bool = True) -> None:
        """
        步骤4: 迭代自训练
        
        修改：传递迭代次数给伪标签生成
        """
        if verbose:
            print("=== Step 4: Iterative Self-Training ===")
        
        # 使用提供的配置或默认配置
        st_config = self_training_config or SELF_TRAINING_CONFIG
        pl_config = pseudo_label_config or PSEUDO_LABEL_CONFIG
        
        current_X_labeled = X_labeled.copy()
        current_y_labeled = y_labeled.copy()
        current_sample_weights = np.ones(len(current_y_labeled))
        current_X_unlabeled = X_unlabeled.copy()
        
        for iteration in range(st_config['n_iterations']):
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{st_config['n_iterations']} ---")
            
            # 生成伪标签（传递迭代次数）
            pseudo_X, pseudo_y, pseudo_uncertainties = self.step3_generate_high_quality_pseudo_labels(
                current_X_unlabeled, pl_config, verbose=verbose, iteration=iteration  # 添加 iteration
            )
            
            if len(pseudo_X) < pl_config['min_pseudo_labels']:
                if verbose:
                    print(f"Generated only {len(pseudo_X)} pseudo labels, which is below minimum threshold {pl_config['min_pseudo_labels']}. Stopping iterations.")
                break
            
            # 基于不确定性的样本权重
            pseudo_weights = 0.5 * (1.0 / (1.0 + pseudo_uncertainties))
            pseudo_weights *= st_config.get('sample_weight_decay', 1.0) ** iteration

            # 扩展训练集
            expanded_X = np.vstack([current_X_labeled, pseudo_X])
            expanded_y = np.hstack([current_y_labeled, pseudo_y])
            combined_weights = np.hstack([current_sample_weights, pseudo_weights])
            
            # 重新训练集成模型
            if verbose:
                print("Retraining ensemble with expanded dataset...")
            self.step2_train_ensemble_with_encoded_features(
                expanded_X,
                expanded_y,
                X_val=X_val,
                y_val=y_val,
                sample_weights=combined_weights,
                verbose=verbose
            )
            
            # 更新当前标记数据集（选择性地添加高置信度伪标签）
            high_confidence_percentile = st_config.get('high_confidence_percentile', 50)
            high_confidence_mask = pseudo_uncertainties <= np.percentile(pseudo_uncertainties, high_confidence_percentile)
            
            if np.sum(high_confidence_mask) > 0:
                current_X_labeled = np.vstack([current_X_labeled, pseudo_X[high_confidence_mask]])
                current_y_labeled = np.hstack([current_y_labeled, pseudo_y[high_confidence_mask]])
                current_sample_weights = np.hstack([current_sample_weights, pseudo_weights[high_confidence_mask]])
            
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
        
        # 打印高级伪标签生成器的汇总统计（如果使用）
        if self.use_advanced_pseudo_labeling and self.pseudo_label_generator:
            summary = self.pseudo_label_generator.get_iteration_summary()
            if verbose and summary:
                print("\n=== Pseudo Label Generation Summary ===")
                print(f"Total iterations: {summary['n_iterations']}")
                print(f"Total pseudo labels generated: {summary['total_pseudo_labels']}")
                print(f"Mean selection rate: {summary['mean_selection_rate']:.3f}")
                print(f"Quality score trend: {summary['quality_trend']}")
                
    
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
        if self.poly_features is not None:
            X_poly = self.poly_features.transform(X)
        else:
            X_poly = X
        X_scaled = self.feature_scaler.transform(X_poly)

        # 获取编码特征
        encoded_features = self.autoencoder_trainer.encode(X_scaled)
        encoded_features = self.pca.transform(encoded_features)
        
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
            'device': self.device,
            'use_advanced_pseudo_labeling': self.use_advanced_pseudo_labeling,  # 新增
            'use_improved_vae': self.use_improved_vae,  # 新增
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
        self.use_advanced_pseudo_labeling = model_data.get('use_advanced_pseudo_labeling', False)
        self.use_improved_vae = model_data.get('use_improved_vae', False)
        # 如果使用高级伪标签生成，重新初始化生成器
        if self.use_advanced_pseudo_labeling and ADVANCED_PSEUDO_LABELING_AVAILABLE:
            try:
                self.pseudo_label_generator = AdvancedPseudoLabelGenerator(
                    base_config=PSEUDO_LABEL_CONFIG,
                    use_active_learning=self.semi_supervised_config.get('use_active_learning', True),
                    use_clustering=self.semi_supervised_config.get('use_clustering', True)
                )
            except:
                self.pseudo_label_generator = None
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
