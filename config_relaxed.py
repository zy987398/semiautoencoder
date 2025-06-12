# 放宽的配置文件，专门用于解决伪标签生成失败的问题
import torch
import psutil

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 固定随机种子
torch.manual_seed(SEED)
if DEVICE == 'cuda':
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# VAE配置（简化）
AUTOENCODER_CONFIG = {
    'latent_dims': [16, 8],
    'dropout_rate': 0.2,
    'epochs': 100,
    'batch_size': 64,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'scheduler_patience': 20,
    'scheduler_factor': 0.5,
    'kl_weight': 1e-3,
    'use_improved_vae': False,  # 暂时禁用以避免兼容性问题
    'use_attention': False,
    'use_residual': False
}

# 集成模型配置（简化）
ENSEMBLE_CONFIG = {
    'LightGBM': {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'verbosity': -1,
        'force_col_wise': True,
        'device_type': 'cpu',  # 避免GPU兼容性问题
        'num_threads': 4,
    },
    'NGBoost': {
        'n_estimators': 500,
        'learning_rate': 0.01,
        'random_state': SEED
    }
}

# 半监督配置
SEMI_SUPERVISED_CONFIG = {
    'validation_split': 0.2,
    'n_cv_folds': 3,
    'poly_degree': 1,  # 不使用多项式特征！重要！
    'poly_interaction_only': False,
    'poly_include_bias': False,
    'use_advanced_pseudo_labeling': False,  # 禁用以避免特征维度问题
    'use_active_learning': False,
    'use_clustering': False
}

# 伪标签配置（非常宽松）
PSEUDO_LABEL_CONFIG = {
    # 基于百分位的阈值
    'confidence_threshold': 0.7,        # 选择不确定性最低的70%
    'reconstruction_threshold': 0.95,   # 选择重建误差最低的95%
    'ensemble_agreement_threshold': 0.5, # 选择一致性最高的50%
    'min_pseudo_labels': 10,           # 最少生成10个
    'iqr_multiplier': 3.0              # 非常宽松的范围
}

# 自训练配置
SELF_TRAINING_CONFIG = {
    'n_iterations': 3,
    'high_confidence_percentile': 50,
    'sample_weight_decay': 0.95,
    'min_improvement': 0.001,
    'patience': 2,
}

# 优化配置
OPTIMIZATION_CONFIG = {
    'chunk_size': 5000,
    'n_bags': 5,  # 使用5个bags（原来的默认值）
    'bag_subsample_ratio': 0.8,
    'dataloader_num_workers': 0,
    'n_jobs': 4,
}

# 内存配置
MEMORY_CONFIG = {
    'enable_gc_collection': True,
    'gc_collect_frequency': 5,
    'clear_gpu_cache': True,
    'log_memory_usage': True,
    'memory_limit_mb': 8192,
}

print(f"Relaxed configuration loaded")
print(f"Device: {DEVICE}")
print(f"Polynomial degree: {SEMI_SUPERVISED_CONFIG['poly_degree']} (no polynomial features)")
print(f"Min pseudo labels: {PSEUDO_LABEL_CONFIG['min_pseudo_labels']}")