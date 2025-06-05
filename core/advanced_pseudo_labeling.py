"""高级伪标签生成策略，包含自适应阈值和主动学习"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import entropy
import torch


class AdvancedPseudoLabelGenerator:
    """高级伪标签生成器"""
    
    def __init__(self, 
                 base_config: Dict,
                 use_active_learning: bool = True,
                 use_clustering: bool = True):
        """
        Args:
            base_config: 基础配置
            use_active_learning: 是否使用主动学习策略
            use_clustering: 是否使用聚类辅助
        """
        self.config = base_config
        self.use_active_learning = use_active_learning
        self.use_clustering = use_clustering
        self.iteration_stats = []
        
    def generate_pseudo_labels(self,
                             X_unlabeled: np.ndarray,
                             ensemble_estimator,
                             autoencoder_trainer,
                             feature_processor,
                             iteration: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        生成高质量伪标签
        
        Returns:
            pseudo_features: 伪标签样本特征
            pseudo_targets: 伪标签目标值
            pseudo_weights: 样本权重
            stats: 统计信息
        """
        # 特征预处理
        X_processed = feature_processor(X_unlabeled)
        encoded_features, recon_errors = autoencoder_trainer.encode_and_get_errors(X_processed)
        X_combined = np.hstack([X_processed, encoded_features])
        
        # 获取预测和不确定性
        predictions, uncertainties, individual_preds = ensemble_estimator.predict_with_uncertainty(X_combined)
        
        # 1. 自适应阈值调整
        adaptive_thresholds = self._compute_adaptive_thresholds(
            uncertainties, recon_errors, iteration
        )
        
        # 2. 多维度质量评分
        quality_scores = self._compute_quality_scores(
            predictions, uncertainties, individual_preds, 
            recon_errors, encoded_features
        )
        
        # 3. 聚类辅助选择
        if self.use_clustering:
            cluster_mask = self._cluster_based_selection(
                encoded_features, quality_scores
            )
        else:
            cluster_mask = np.ones(len(X_unlabeled), dtype=bool)
        
        # 4. 主动学习策略
        if self.use_active_learning:
            active_indices = self._active_learning_selection(
                uncertainties, quality_scores, encoded_features
            )
        else:
            active_indices = np.arange(len(X_unlabeled))
        
        # 5. 综合筛选
        final_mask = self._comprehensive_filtering(
            quality_scores, adaptive_thresholds, cluster_mask, active_indices
        )
        
        # 6. 计算样本权重
        sample_weights = self._compute_sample_weights(
            quality_scores[final_mask],
            uncertainties[final_mask],
            iteration
        )
        
        # 统计信息
        stats = self._collect_statistics(
            X_unlabeled, final_mask, quality_scores,
            uncertainties, predictions, sample_weights
        )
        
        self.iteration_stats.append(stats)
        
        return (X_unlabeled[final_mask], 
                predictions[final_mask], 
                sample_weights,
                stats)
    
    def _compute_adaptive_thresholds(self, 
                                   uncertainties: np.ndarray,
                                   recon_errors: np.ndarray,
                                   iteration: int) -> Dict[str, float]:
        """计算自适应阈值"""
        # 基于迭代次数动态调整
        decay_factor = 0.9 ** iteration
        
        # 使用IQR方法确定阈值
        q1_unc, q3_unc = np.percentile(uncertainties, [25, 75])
        iqr_unc = q3_unc - q1_unc
        
        q1_rec, q3_rec = np.percentile(recon_errors, [25, 75])
        iqr_rec = q3_rec - q1_rec
        
        thresholds = {
            'uncertainty': q1_unc + 1.5 * iqr_unc * decay_factor,
            'reconstruction': q1_rec + 1.5 * iqr_rec * decay_factor,
            'min_samples': max(100, int(self.config['min_pseudo_labels'] * decay_factor))
        }
        
        return thresholds
    
    def _compute_quality_scores(self,
                              predictions: np.ndarray,
                              uncertainties: np.ndarray,
                              individual_preds: Dict[str, np.ndarray],
                              recon_errors: np.ndarray,
                              encoded_features: np.ndarray) -> np.ndarray:
        """计算多维度质量评分"""
        n_samples = len(predictions)
        
        # 1. 不确定性评分（归一化）
        unc_scores = 1 - (uncertainties - uncertainties.min()) / (uncertainties.max() - uncertainties.min() + 1e-8)
        
        # 2. 重建误差评分
        rec_scores = 1 - (recon_errors - recon_errors.min()) / (recon_errors.max() - recon_errors.min() + 1e-8)
        
        # 3. 预测一致性评分
        pred_matrix = np.array(list(individual_preds.values())).T
        pred_std = np.std(pred_matrix, axis=1)
        consistency_scores = 1 - (pred_std - pred_std.min()) / (pred_std.max() - pred_std.min() + 1e-8)
        
        # 4. 预测熵评分（衡量预测分布的不确定性）
        pred_probs = np.abs(pred_matrix) / np.sum(np.abs(pred_matrix), axis=1, keepdims=True)
        pred_entropy = np.array([entropy(p) for p in pred_probs])
        entropy_scores = 1 - (pred_entropy - pred_entropy.min()) / (pred_entropy.max() - pred_entropy.min() + 1e-8)
        
        # 5. 局部密度评分（使用LOF）
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        lof_scores = lof.fit_predict(encoded_features)
        lof_scores = (lof_scores + 1) / 2  # 转换到[0, 1]
        
        # 综合评分（可调权重）
        weights = {
            'uncertainty': 0.3,
            'reconstruction': 0.2,
            'consistency': 0.2,
            'entropy': 0.15,
            'density': 0.15
        }
        
        quality_scores = (
            weights['uncertainty'] * unc_scores +
            weights['reconstruction'] * rec_scores +
            weights['consistency'] * consistency_scores +
            weights['entropy'] * entropy_scores +
            weights['density'] * lof_scores
        )
        
        return quality_scores
    
    def _cluster_based_selection(self,
                               encoded_features: np.ndarray,
                               quality_scores: np.ndarray) -> np.ndarray:
        """基于聚类的样本选择"""
        # 使用DBSCAN进行聚类
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(encoded_features)
        labels = clustering.labels_
        
        # 为每个聚类选择高质量样本
        selected_mask = np.zeros(len(encoded_features), dtype=bool)
        
        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # 噪声点
                continue
                
            cluster_mask = labels == cluster_id
            cluster_scores = quality_scores[cluster_mask]
            
            # 选择每个聚类中质量分数前50%的样本
            threshold = np.percentile(cluster_scores, 50)
            cluster_selected = cluster_scores >= threshold
            
            # 更新选择掩码
            cluster_indices = np.where(cluster_mask)[0]
            selected_mask[cluster_indices[cluster_selected]] = True
        
        return selected_mask
    
    def _active_learning_selection(self,
                                 uncertainties: np.ndarray,
                                 quality_scores: np.ndarray,
                                 encoded_features: np.ndarray) -> np.ndarray:
        """主动学习策略选择最有价值的样本"""
        n_samples = len(uncertainties)
        
        # 1. 不确定性采样：选择不确定性适中的样本
        unc_mean = np.mean(uncertainties)
        unc_std = np.std(uncertainties)
        unc_value = np.abs(uncertainties - unc_mean) / (unc_std + 1e-8)
        
        # 2. 多样性采样：确保样本多样性
        # 使用k-means++初始化策略选择多样化的样本
        diversity_indices = self._diversity_sampling(encoded_features, n_select=n_samples // 2)
        
        # 3. 综合价值评分
        value_scores = quality_scores * (1 - unc_value)  # 高质量但不确定性适中
        
        # 选择价值最高的样本
        n_select = max(self.config['min_pseudo_labels'], int(n_samples * 0.3))
        top_indices = np.argsort(value_scores)[-n_select:]
        
        # 确保包含多样性样本
        combined_indices = np.unique(np.concatenate([top_indices, diversity_indices[:n_select//4]]))
        
        return combined_indices
    
    def _diversity_sampling(self, features: np.ndarray, n_select: int) -> np.ndarray:
        """多样性采样"""
        if n_select >= len(features):
            return np.arange(len(features))
        
        # 使用k-means++策略
        indices = []
        # 随机选择第一个点
        indices.append(np.random.randint(len(features)))
        
        for _ in range(1, n_select):
            # 计算到已选择点的最小距离
            min_distances = np.full(len(features), np.inf)
            for idx in indices:
                distances = np.linalg.norm(features - features[idx], axis=1)
                min_distances = np.minimum(min_distances, distances)
            
            # 选择距离最远的点
            probabilities = min_distances / min_distances.sum()
            next_idx = np.random.choice(len(features), p=probabilities)
            indices.append(next_idx)
        
        return np.array(indices)
    
    def _comprehensive_filtering(self,
                               quality_scores: np.ndarray,
                               thresholds: Dict[str, float],
                               cluster_mask: np.ndarray,
                               active_indices: np.ndarray) -> np.ndarray:
        """综合过滤策略"""
        # 质量分数阈值
        quality_threshold = np.percentile(quality_scores, 100 - self.config['confidence_threshold'] * 100)
        quality_mask = quality_scores >= quality_threshold
        
        # 主动学习掩码
        active_mask = np.zeros(len(quality_scores), dtype=bool)
        active_mask[active_indices] = True
        
        # 综合所有过滤条件
        final_mask = quality_mask & cluster_mask & active_mask
        
        # 确保最小样本数
        if np.sum(final_mask) < thresholds['min_samples']:
            # 放宽条件，选择质量分数最高的样本
            top_indices = np.argsort(quality_scores)[-thresholds['min_samples']:]
            final_mask = np.zeros(len(quality_scores), dtype=bool)
            final_mask[top_indices] = True
        
        return final_mask
    
    def _compute_sample_weights(self,
                              quality_scores: np.ndarray,
                              uncertainties: np.ndarray,
                              iteration: int) -> np.ndarray:
        """计算样本权重"""
        # 基础权重：质量分数
        base_weights = quality_scores
        
        # 不确定性调整
        uncertainty_weights = 1.0 / (1.0 + uncertainties)
        
        # 迭代衰减
        iteration_decay = 0.95 ** iteration
        
        # 综合权重
        weights = base_weights * uncertainty_weights * iteration_decay
        
        # 归一化
        weights = weights / np.sum(weights) * len(weights)
        
        return weights
    
    def _collect_statistics(self,
                          X_unlabeled: np.ndarray,
                          final_mask: np.ndarray,
                          quality_scores: np.ndarray,
                          uncertainties: np.ndarray,
                          predictions: np.ndarray,
                          sample_weights: np.ndarray) -> Dict:
        """收集统计信息"""
        selected_indices = np.where(final_mask)[0]
        
        stats = {
            'total_unlabeled': len(X_unlabeled),
            'selected_samples': len(selected_indices),
            'selection_rate': len(selected_indices) / len(X_unlabeled),
            'mean_quality_score': np.mean(quality_scores[final_mask]),
            'std_quality_score': np.std(quality_scores[final_mask]),
            'mean_uncertainty': np.mean(uncertainties[final_mask]),
            'std_uncertainty': np.std(uncertainties[final_mask]),
            'mean_prediction': np.mean(predictions[final_mask]),
            'std_prediction': np.std(predictions[final_mask]),
            'mean_weight': np.mean(sample_weights),
            'std_weight': np.std(sample_weights),
            'quality_score_distribution': {
                'q25': np.percentile(quality_scores[final_mask], 25),
                'q50': np.percentile(quality_scores[final_mask], 50),
                'q75': np.percentile(quality_scores[final_mask], 75)
            }
        }
        
        return stats
    
    def get_iteration_summary(self) -> Dict:
        """获取所有迭代的汇总统计"""
        if not self.iteration_stats:
            return {}
        
        summary = {
            'n_iterations': len(self.iteration_stats),
            'total_pseudo_labels': sum(s['selected_samples'] for s in self.iteration_stats),
            'mean_selection_rate': np.mean([s['selection_rate'] for s in self.iteration_stats]),
            'quality_trend': [s['mean_quality_score'] for s in self.iteration_stats],
            'uncertainty_trend': [s['mean_uncertainty'] for s in self.iteration_stats]
        }
        
        return summary