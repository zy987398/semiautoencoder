"""
集成模型和不确定性估计
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from ngboost import NGBRegressor
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class EnsembleUncertaintyEstimator:
    """集成不确定性估计器，结合多个模型的预测和不确定性"""
    
    def __init__(self, models_config: Dict[str, Dict[str, Any]], 
                 n_cv_folds: int = 5,
                 use_gpu: bool = True):
        """
        Args:
            models_config: 模型配置字典
            n_cv_folds: 交叉验证折数
            use_gpu: 是否使用GPU（仅适用于支持GPU的模型）
        """
        self.models_config = models_config
        self.n_cv_folds = n_cv_folds
        self.use_gpu = use_gpu
        self.models = {}
        self.model_weights = {}
        self.feature_scaler = StandardScaler()
        self._is_fitted = False
        
        # 检查GPU支持
        self._check_gpu_support()
        
    def _check_gpu_support(self):
        """检查并配置GPU支持"""
        import torch
        
        self.gpu_available = torch.cuda.is_available() and self.use_gpu
        if self.gpu_available:
            print("GPU is available for ensemble models")
            
            # 检查XGBoost GPU支持
            try:
                import xgboost as xgb
                # XGBoost GPU支持需要特殊编译版本
                self.xgb_gpu_available = 'gpu_hist' in xgb.XGBRegressor()._get_param_names()
                if self.xgb_gpu_available:
                    print("✅ XGBoost GPU support detected")
            except:
                self.xgb_gpu_available = False
                
            # 检查LightGBM GPU支持
            try:
                import lightgbm as lgb
                # LightGBM GPU支持需要特殊编译版本
                self.lgb_gpu_available = True  # 需要测试
                print("✅ LightGBM GPU support available (requires GPU build)")
            except:
                self.lgb_gpu_available = False
                
            # CatBoost默认支持GPU
            self.catboost_gpu_available = True
            print("✅ CatBoost GPU support available")
        else:
            self.xgb_gpu_available = False
            self.lgb_gpu_available = False
            self.catboost_gpu_available = False
            if self.use_gpu:
                print("⚠️  GPU requested but not available, using CPU for ensemble models")
        
    def initialize_models(self):
        """初始化所有模型"""
        for model_name, config in self.models_config.items():
            config = config.copy()  # 避免修改原始配置
            
            if model_name == 'NGBoost':
                self.models[model_name] = NGBRegressor(**config)
            elif model_name == 'LightGBM':
                # LightGBM GPU配置
                if self.lgb_gpu_available and self.gpu_available:
                    config['device'] = 'gpu'
                    config['gpu_use_dp'] = True  # 使用双精度
                    print(f"LightGBM configured for GPU")
                self.models[model_name] = LGBMRegressor(**config)
            elif model_name == 'XGBoost':
                # XGBoost GPU配置
                if self.xgb_gpu_available and self.gpu_available:
                    config['tree_method'] = 'gpu_hist'
                    config['gpu_id'] = 0
                    config['predictor'] = 'gpu_predictor'
                    print(f"XGBoost configured for GPU")
                self.models[model_name] = XGBRegressor(**config)
            elif model_name == 'CatBoost':
                # CatBoost GPU配置
                if self.catboost_gpu_available and self.gpu_available:
                    # GPU 模式下不支持 rsm，先移除
                    if 'rsm' in config:
                        config.pop('rsm')
                    config['task_type'] = 'GPU'
                    config['devices'] = '0'
            elif model_name == 'RandomForest':
                # RandomForest不支持GPU，但可以通过n_jobs并行化
                config['n_jobs'] = -1  # 使用所有CPU核心
                self.models[model_name] = RandomForestRegressor(**config)
            else:
                raise ValueError(f"Unknown model type: {model_name}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            verbose: bool = True) -> 'EnsembleUncertaintyEstimator':
        """
        训练所有模型
        
        Args:
            X: 训练特征
            y: 训练目标
            eval_set: 验证集 (X_val, y_val)
            verbose: 是否打印训练信息
            
        Returns:
            self
        """
        # 初始化模型
        self.initialize_models()
        
        # 标准化特征
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # 使用交叉验证评估模型权重
        kf = KFold(n_splits=self.n_cv_folds, shuffle=True, random_state=42)
        model_scores = {}
        
        if verbose:
            print("Training ensemble models with cross-validation...")
        
        for model_name, model in self.models.items():
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # 克隆模型配置
                fold_config = self.models_config[model_name].copy()
                
                # 根据模型类型创建实例并配置GPU
                if model_name == 'NGBoost':
                    fold_model = NGBRegressor(**fold_config)
                elif model_name == 'LightGBM':
                    if self.lgb_gpu_available and self.gpu_available:
                        fold_config['device'] = 'gpu'
                        fold_config['gpu_use_dp'] = True
                    fold_model = LGBMRegressor(**fold_config)
                elif model_name == 'XGBoost':
                    if self.xgb_gpu_available and self.gpu_available:
                        fold_config['tree_method'] = 'gpu_hist'
                        fold_config['gpu_id'] = 0
                        fold_config['predictor'] = 'gpu_predictor'
                    fold_model = XGBRegressor(**fold_config)
                elif model_name == 'CatBoost':
                    if self.catboost_gpu_available and self.gpu_available:
                        # GPU 模式下不支持 rsm，先移除
                        if 'rsm' in fold_config:
                            fold_config.pop('rsm')
                        fold_config['task_type'] = 'GPU'
                        fold_config['devices'] = '0'
                    fold_model = CatBoostRegressor(**fold_config)
                elif model_name == 'RandomForest':
                    fold_config['n_jobs'] = -1
                    fold_model = RandomForestRegressor(**fold_config)
                
                # 训练模型
                fold_model.fit(X_train, y_train)
                
                # 预测和评估
                y_pred = fold_model.predict(X_val)
                cv_scores.append(r2_score(y_val, y_pred))
            
            model_scores[model_name] = np.mean(cv_scores)
            if verbose:
                print(f"{model_name} CV R² Score: {model_scores[model_name]:.4f}")
        
        # 计算模型权重（基于性能）
        total_score = sum(max(0, score) for score in model_scores.values())
        if total_score > 0:
            for model_name in model_scores:
                self.model_weights[model_name] = max(0, model_scores[model_name]) / total_score
        else:
            # 如果所有模型表现都很差，使用均等权重
            for model_name in model_scores:
                self.model_weights[model_name] = 1.0 / len(model_scores)
        
        if verbose:
            print("\nModel weights:")
            for model_name, weight in self.model_weights.items():
                print(f"  {model_name}: {weight:.3f}")
        
        # 在全数据集上重新训练
        if verbose:
            print("\nRetraining models on full dataset...")
        
        for model_name, model in self.models.items():
            model.fit(X_scaled, y)
        
        self._is_fitted = True
        return self
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        预测并估计不确定性
        
        Args:
            X: 预测特征
            
        Returns:
            predictions: 集成预测结果
            uncertainties: 不确定性估计
            individual_predictions: 各个模型的预测结果
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.feature_scaler.transform(X)
        
        individual_preds = {}
        uncertainties = {}
        
        for model_name, model in self.models.items():
            if model_name == 'NGBoost':
                # NGBoost 直接提供不确定性
                preds = model.predict(X_scaled)
                individual_preds[model_name] = preds
                # 尝试获取预测分布，否则将不确定性设为 0
                try:
                    pred_dists = model.pred_dist(X_scaled)
                    # 如果是“分布对象列表”，计算其 var；否则当作普通数组
                    if hasattr(pred_dists[0], 'var'):
                        uncert = np.array([np.sqrt(dist.var()) for dist in pred_dists])
                    else:
                        uncert = np.zeros_like(preds)
                except Exception:
                    uncert = np.zeros_like(preds)
                uncertainties[model_name] = uncert
            else:
                # 其他模型使用不同方法估计不确定性
                preds = model.predict(X_scaled)
                individual_preds[model_name] = preds
                
                if hasattr(model, 'estimators_'):  # RandomForest
                    # 使用决策树预测的标准差
                    tree_preds = np.array([tree.predict(X_scaled) for tree in model.estimators_])
                    uncert = np.std(tree_preds, axis=0)
                else:
                    # 对于其他模型，使用简单的经验不确定性估计
                    # 基于预测值与预测均值的偏差
                    pred_mean = np.mean(preds)
                    pred_std = np.std(preds) if np.std(preds) > 0 else 1.0
                    uncert = np.abs(preds - pred_mean) / pred_std * 0.1
                
                uncertainties[model_name] = uncert
        
        # 加权集成预测
        ensemble_pred = np.zeros(len(X))
        ensemble_uncertainty = np.zeros(len(X))
        
        for model_name in individual_preds:
            weight = self.model_weights[model_name]
            ensemble_pred += weight * individual_preds[model_name]
            ensemble_uncertainty += weight * uncertainties[model_name]
        
        return ensemble_pred, ensemble_uncertainty, individual_preds
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """简单预测接口"""
        predictions, _, _ = self.predict_with_uncertainty(X)
        return predictions
