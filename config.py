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

# 自编码器配置
AUTOENCODER_CONFIG = {
    'latent_dims': [64, 32, 16],
    'dropout_rate': 0.2,
    'epochs': 300,
    'batch_size': 1024,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'scheduler_patience': 20,
    'scheduler_factor': 0.5
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

# 半监督学习配置
SEMI_SUPERVISED_CONFIG = {
    'validation_split': 0.2,
    'n_cv_folds': 5,
    'poly_degree': 2,
    'poly_interaction_only': True,
    'poly_include_bias': False
}

# 伪标签生成配置
PSEUDO_LABEL_CONFIG = {
    'confidence_threshold': 0.9,
    'reconstruction_threshold': 0.9,
    'ensemble_agreement_threshold': 0.8,
    'min_pseudo_labels': 50,
    'iqr_multiplier': 1.5
}

# 自训练配置
SELF_TRAINING_CONFIG = {
    'n_iterations': 3,
    'high_confidence_percentile': 50
}