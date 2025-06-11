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

# 在现有 config.py 中进行以下修改：

# 2. VAE配置 - 适应小数据集
AUTOENCODER_CONFIG = {
    'latent_dims': [16, 8],      # 减少维度（原本是[64, 32, 16]）
    'dropout_rate': 0.3,         # 增加dropout
    'epochs': 150,               # 减少epochs
    'batch_size': 64,            # 减小批次
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'scheduler_patience': 20,
    'scheduler_factor': 0.5,
    'kl_weight': 1e-3,
    'use_improved_vae': False,   # 禁用高级功能
    'use_attention': False,
    'use_residual': False
}

# 3. 集成配置 - 只使用最高效的模型
ENSEMBLE_CONFIG = {
    'LightGBM': {
        'n_estimators': 300,     # 减少（原本是1500）
        'learning_rate': 0.05,
        'num_leaves': 31,        # 减少（原本是63）
        'min_child_samples': 10, # 适应小数据集
        'reg_alpha': 10,         # 增加正则化
        'reg_lambda': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'min_gain_to_split': 0.1,
        'random_state': SEED,
        'verbosity': -1,
        'force_col_wise': True,
        'num_threads': 4,        # 限制线程数
    },
    # 注释掉其他模型
    # 'XGBoost': {...},
    # 'CatBoost': {...},
    # 'NGBoost': {...},
}

# 4. 半监督配置 - 针对数据不平衡
SEMI_SUPERVISED_CONFIG = {
    'validation_split': 0.3,     # 增加验证集
    'n_cv_folds': 3,            # 减少折数
    'poly_degree': 1,           # 不使用多项式特征！
    'poly_interaction_only': True,
    'poly_include_bias': False,
    'use_advanced_pseudo_labeling': False,
    'use_active_learning': False,
    'use_clustering': False
}

# 5. 伪标签配置 - 更宽松以获得足够伪标签
PSEUDO_LABEL_CONFIG = {
    'confidence_threshold': 0.2,      # 放宽（原本是0.4）
    'reconstruction_threshold': 0.75,
    'ensemble_agreement_threshold': 0.85,  # 放宽（原本是0.95）
    'min_pseudo_labels': 10,          # 减少最小要求
    'iqr_multiplier': 2.0            # 放宽异常值过滤
}

# 6. 自训练配置 - 减少迭代
SELF_TRAINING_CONFIG = {
    'n_iterations': 2,               # 减少（原本是3-5）
    'high_confidence_percentile': 30 # 更严格选择
}

# 7. 新增：数据采样配置
DATA_SAMPLING_CONFIG = {
    'max_unlabeled_samples': 20000,  # 限制未标记数据量
    'unlabeled_sampling_strategy': 'random',  # 或 'diverse'
    'augment_labeled_data': True,    # 增强标记数据
    'augmentation_factor': 3,        # 增强倍数
    'batch_predict_size': 10000,     # 批量预测大小
}

# 8. 内存管理配置
MEMORY_CONFIG = {
    'enable_gc': True,               # 启用垃圾回收
    'gc_frequency': 10,              # 每10个批次GC一次
    'log_memory': True,              # 记录内存使用
    'memory_limit_mb': 3000,         # 内存限制3GB
}