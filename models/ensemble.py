"""
集成模型和不确定性估计
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from ngboost import NGBRegressor
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Any, Tuple, Optional
import torch
import warnings
warnings.filterwarnings('ignore')


class EnsembleUncertaintyEstimator:
    """集成不确定性估计器，结合多个模型的预测和不确定性"""
    
    def __init__(self, models_config: Dict[str, Dict[str, Any]],
                 n_cv_folds: int = 5,
                 use_gpu: bool = True,
                 bagging_estimators: int = 10):
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
        self.n_bags = bagging_estimators
        self.feature_scaler = StandardScaler()
        self._is_fitted = False
        
        # 检查GPU支持
        self._check_gpu_support()

    def _create_model(self, model_name: str, config: Dict[str, Any]):
        """根据名称和配置创建模型实例"""
        cfg = config.copy()
        if model_name == 'NGBoost':
            return NGBRegressor(**cfg)
        elif model_name == 'LightGBM':
            if self.lgb_gpu_available and self.gpu_available:
                cfg['device'] = 'gpu'
                cfg['gpu_use_dp'] = True
            return LGBMRegressor(**cfg)
        elif model_name == 'XGBoost':
            if self.xgb_gpu_available and self.gpu_available:
                cfg['tree_method'] = 'gpu_hist'
                cfg['gpu_id'] = 0
                cfg['predictor'] = 'gpu_predictor'
            return XGBRegressor(**cfg)
        elif model_name == 'CatBoost':
            if self.catboost_gpu_available and self.gpu_available:
                if 'rsm' in cfg:
                    cfg.pop('rsm')
                cfg['task_type'] = 'GPU'
                cfg['devices'] = '0'
            return CatBoostRegressor(**cfg)
        elif model_name == 'RandomForest':
            cfg['n_jobs'] = -1
            return RandomForestRegressor(**cfg)
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
    def _check_gpu_support(self):
        """检查并配置GPU支持"""
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
        """初始化模型列表"""
        self.models = {}
        for model_name in self.models_config.keys():
            self.models[model_name] = []
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            verbose: bool = True) -> 'EnsembleUncertaintyEstimator':
        """
        使用bagging训练所有模型

        Args:
            X: 训练特征
            y: 训练目标
            eval_set: 未使用，仅保持接口一致
            verbose: 是否打印训练信息

        Returns:
            self
        """
        # 初始化模型列表
        self.initialize_models()

        # 标准化特征
        X_scaled = self.feature_scaler.fit_transform(X)
        n_samples = X_scaled.shape[0]

        for model_name in self.models.keys():
            config = self.models_config[model_name]
            for b in range(self.n_bags):
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_bag = X_scaled[indices]
                y_bag = y[indices]
                bag_model = self._create_model(model_name, config)
                bag_model.fit(X_bag, y_bag)
                self.models[model_name].append(bag_model)
            if verbose:
                print(f"Trained {self.n_bags} {model_name} models")

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

        for model_name, bag_models in self.models.items():
            bag_predictions = np.array([m.predict(X_scaled) for m in bag_models])

            avg_pred = np.mean(bag_predictions, axis=0)
            std_pred = np.std(bag_predictions, axis=0)

            individual_preds[model_name] = avg_pred
            uncertainties[model_name] = std_pred

        if len(individual_preds) == 0:
            raise ValueError("No models were trained")

        all_preds = np.stack(list(individual_preds.values()), axis=0)
        all_uncerts = np.stack(list(uncertainties.values()), axis=0)

        ensemble_pred = np.mean(all_preds, axis=0)
        ensemble_uncertainty = np.mean(all_uncerts, axis=0)

        return ensemble_pred, ensemble_uncertainty, individual_preds
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """简单预测接口"""
        predictions, _, _ = self.predict_with_uncertainty(X)
        return predictions
