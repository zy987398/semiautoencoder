import torch

# 全局设置
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 固定随机种子，并设置 CUDNN 行为
torch.manual_seed(SEED)
if DEVICE == 'cuda':
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 自编码器配置（增强版）
AUTOENCODER_CONFIG = {
    'latent_dims': [64, 32, 16],
    'dropout_rate': 0.2,
    'epochs': 300,
    'batch_size': 1024,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'scheduler_patience': 20,
    'scheduler_factor': 0.5,
    'kl_weight': 1e-3,
    # 新增配置项
    'use_improved_vae': True,    # 使用改进的VAE架构
    'use_attention': True,        # 使用注意力机制
    'use_residual': True,         # 使用残差连接
    'vae_type': 'improved'        # 'standard', 'improved', 或 'temporal'
}

# 集成模型配置
ENSEMBLE_CONFIG = {
    'NGBoost': {
        'n_estimators': 800,
        'learning_rate': 0.01,
        'random_state': SEED
    },
    'LightGBM': {
        'n_estimators': 1500,
        'learning_rate': 0.03,
        'num_leaves': 63,
        'min_child_samples': 10,
        'reg_alpha': 5,
        'reg_lambda': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_gain_to_split': 0.1,
        'random_state': SEED,
        'verbosity': -1,
        'device_type': 'gpu' if DEVICE=='cuda' else 'cpu',
        'gpu_use_dp': True,
    },  
    'CatBoost': {
        'iterations': 300,
        'learning_rate': 0.05,
        'depth': 3,
        'l2_leaf_reg': 50,
        'bootstrap_type': 'MVS',
        'subsample': 0.5,
        'rsm': 0.5,
        'random_strength': 1.0,
        'random_state': SEED,
        'verbose': False,
        # GPU配置（如果可用）
        'task_type': 'GPU' if DEVICE == 'cuda' else 'CPU',
        'devices': '0' if DEVICE == 'cuda' else None
    },
    'XGBoost': {
        'n_estimators': 800,
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_child_weight': 10,
        'gamma': 0,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'reg_alpha': 5,
        'reg_lambda': 10,
        'random_state': SEED,
        'verbosity': 0,
        'tree_method': 'hist',
        'predictor': 'gpu_predictor' if DEVICE=='cuda' else 'cpu_predictor',
        'gpu_id': 0 if DEVICE=='cuda' else None,
    }
}

# 半监督学习配置（增强版）
SEMI_SUPERVISED_CONFIG = {
    'validation_split': 0.2,
    'n_cv_folds': 5,
    'poly_degree': 2,
    'poly_interaction_only': True,
    'poly_include_bias': False,
    # 新增配置项
    'use_advanced_pseudo_labeling': True,  # 使用高级伪标签生成
    'use_active_learning': True,           # 使用主动学习策略
    'use_clustering': True,                # 使用聚类辅助选择
}

# 伪标签生成配置（优化版）
PSEUDO_LABEL_CONFIG = {
    # 基础阈值配置
    'confidence_threshold': 0.3,           # 更严格的置信度阈值
    'reconstruction_threshold': 0.7,       # 重建误差阈值
    'ensemble_agreement_threshold': 0.95,  # 集成一致性阈值
    'min_pseudo_labels': 50,              # 最小伪标签数量
    'iqr_multiplier': 1.0,                # IQR乘数（用于异常值过滤）
    
    # 高级伪标签生成配置（新增）
    'quality_weights': {                   # 质量评分权重
        'uncertainty': 0.3,
        'reconstruction': 0.2,
        'consistency': 0.2,
        'entropy': 0.15,
        'density': 0.15
    },
    'adaptive_threshold_decay': 0.9,       # 自适应阈值衰减因子
    'clustering_eps': 0.5,                 # DBSCAN聚类参数
    'clustering_min_samples': 5,           # DBSCAN最小样本数
    'lof_n_neighbors': 20,                 # LOF邻居数
    'lof_contamination': 0.1,              # LOF异常比例
    'diversity_sampling_ratio': 0.25,      # 多样性采样比例
    'value_selection_ratio': 0.3,          # 价值选择比例
}

# 自训练配置（优化版）
SELF_TRAINING_CONFIG = {
    'n_iterations': 5,                     # 增加迭代次数
    'high_confidence_percentile': 30,      # 高置信度百分位（更严格）
    'sample_weight_decay': 0.95,           # 样本权重衰减因子
    'min_improvement': 0.001,              # 最小改进阈值（提前停止）
    'patience': 2,                         # 早停耐心值
}

# 性能优化配置（新增）
OPTIMIZATION_CONFIG = {
    'use_mixed_precision': True,           # 使用混合精度训练
    'use_gradient_checkpointing': True,    # 使用梯度检查点
    'dataloader_num_workers': 4,           # 数据加载器工作进程数
    'dataloader_pin_memory': True,         # 固定内存
    'dataloader_prefetch_factor': 2,       # 预取因子
    'parallel_backend': 'threading',       # 并行后端：'threading' 或 'multiprocessing'
    'n_jobs': -1,                         # 并行作业数（-1表示使用所有CPU）
}

# 评估配置（新增）
EVALUATION_CONFIG = {
    'compute_shap': True,                  # 计算SHAP值
    'n_shap_samples': 100,                 # SHAP分析样本数
    'compute_partial_dependence': True,     # 计算部分依赖
    'n_pd_features': 5,                    # 部分依赖特征数
    'noise_levels': [0.01, 0.05, 0.1],     # 鲁棒性测试噪声水平
    'missing_rates': [0.05, 0.1, 0.2],     # 缺失值测试比例
}