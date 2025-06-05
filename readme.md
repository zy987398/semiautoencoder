# 半监督学习框架 (Semi-Supervised Learning Framework)

这是一个结合深度学习（变分自编码器）和集成学习的高级半监督学习框架，专门用于处理标记
数据有限的回归问题。
## 🚀 GPU加速支持

本框架完全支持GPU加速训练，可以显著提升训练速度。

### GPU环境检查

首先运行环境检查脚本：

```bash
python check_environment.py
```

这将检查：
- Python版本
- CUDA安装
- PyTorch GPU支持
- 所有依赖包
- 系统内存和GPU内存
- 性能测试

### GPU使用指南

1. **自动GPU检测**：框架会自动检测可用的GPU并使用
2. **手动指定设备**：
   ```python
   model = EnhancedSemiSupervisedEnsemble(device='cuda')  # 强制使用GPU
   model = EnhancedSemiSupervisedEnsemble(device='cpu')   # 强制使用CPU
   ```

3. **GPU优化训练示例**：
   ```bash
   python gpu_example.py
   ```

### GPU性能优化

框架包含多项GPU优化：
- **混合精度训练**：自动使用FP16加速
- **优化的批大小**：自动调整批大小以最大化GPU利用率
- **内存管理**：自动清理GPU内存
- **多GPU支持**：支持DataParallel多GPU训练

## 项目结构

```
semi_supervised_learning/
│
├── config.py              # 配置文件
├── models/               # 模型目录
│   ├── __init__.py
│   ├── vae.py            # 变分自编码器模型
│   └── ensemble.py       # 集成模型
├── core/                 # 核心功能
│   ├── __init__.py
│   └── semi_supervised.py # 半监督学习主类
├── utils/                # 工具函数
│   ├── __init__.py
│   ├── data_utils.py     # 数据处理工具
│   └── visualization.py  # 可视化工具
├── run_pipeline.py       # 主运行脚本
├── example_usage.py      # 使用示例
├── requirements.txt      # 依赖包列表
└── README.md            # 项目说明
```

## 主要特性

1. **深度特征学习**：使用变分自编码器从标记和未标记数据中学习有效的特征表示
2. **集成学习**：结合多个强大的机器学习模型（LightGBM、XGBoost、CatBoost、NGBoost）
3. **不确定性估计**：提供预测不确定性的量化估计
4. **高质量伪标签**：通过多重过滤机制确保伪标签的质量
5. **迭代自训练**：逐步扩展训练集，提高模型性能

## 安装

### 1. 克隆或下载项目

```bash
git clone <repository-url>
cd semi_supervised_learning
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. （可选）安装GPU版本的PyTorch

如果您有NVIDIA GPU，请访问 [PyTorch官网](https://pytorch.org/get-started/locally/) 选择合适的CUDA版本安装。

例如，对于CUDA 11.8：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 5. 验证安装

```bash
python check_environment.py
```

## 快速开始

### 1. 准备数据

准备两个CSV文件：
- `labeled.csv`：包含特征列和目标列（如 'crack_length'）
- `unlabeled.csv`：只包含特征列

### 2. 运行完整流程

```bash
python run_pipeline.py --labeled_file labeled.csv --unlabeled_file unlabeled.csv --target_column crack_length
```

### 3. 使用Python代码

```python
from core.semi_supervised import EnhancedSemiSupervisedEnsemble
from utils.data_utils import load_data
from run_pipeline import run_full_pipeline

# 加载数据
X_labeled, y_labeled, X_unlabeled = load_data(
    'labeled.csv',
    'unlabeled.csv',
    target_column='crack_length'
)

# 运行完整流程
model = run_full_pipeline(
    X_labeled, y_labeled, X_unlabeled,
    verbose=True
)

# 预测新数据
predictions = model.predict(new_data)

# 保存模型
model.save_model('my_model.pkl')
```

## 详细使用指南

### 1. 自定义配置

```python
# 自定义VAE配置
autoencoder_config = {
    'latent_dims': [128, 64, 32],
    'dropout_rate': 0.3,
    'epochs': 500,
    'batch_size': 128,
    'learning_rate': 5e-4
}

# 自定义集成模型配置
ensemble_config = {
    'LightGBM': {
        'n_estimators': 2000,
        'learning_rate': 0.02,
        'num_leaves': 31
    },
    'XGBoost': {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 6
    }
}

# 创建模型
model = EnhancedSemiSupervisedEnsemble(
    autoencoder_config=autoencoder_config,
    ensemble_config=ensemble_config
)
```

### 2. 分步执行

```python
# Step 1: 特征工程和VAE训练
model.step1_feature_engineering_and_autoencoder(X_labeled, X_unlabeled)

# Step 2: 训练集成模型
model.step2_train_ensemble_with_encoded_features(X_labeled, y_labeled)

# Step 3: 生成伪标签
pseudo_X, pseudo_y, pseudo_uncertainties = model.step3_generate_high_quality_pseudo_labels(
    X_unlabeled
)

# Step 4: 迭代自训练
model.step4_iterative_self_training(
    X_labeled, y_labeled, X_unlabeled,
    n_iterations=3
)
```

### 3. 不确定性分析

```python
# 获取预测和不确定性
predictions, uncertainties, individual_preds = model.predict(
    X_test, 
    return_uncertainty=True
)

# 可视化不确定性
from utils.visualization import plot_uncertainty_analysis
plot_uncertainty_analysis(y_test, predictions, uncertainties)
```

### 4. 模型评估

```python
# 评估模型性能
metrics, predictions, uncertainties = model.evaluate(X_test, y_test)

print(f"RMSE: {metrics['ensemble']['RMSE']:.4f}")
print(f"R²: {metrics['ensemble']['R2']:.4f}")
print(f"Mean Uncertainty: {metrics['uncertainty']['mean_uncertainty']:.4f}")
```

## 命令行参数

运行 `python run_pipeline.py --help` 查看所有可用参数：

- `--labeled_file`: 标记数据文件路径
- `--unlabeled_file`: 未标记数据文件路径
- `--target_column`: 目标列名称
- `--test_size`: 测试集比例
- `--model_save_path`: 模型保存路径
- `--results_dir`: 结果保存目录
- `--device`: 使用的设备（cuda/cpu）
- `--verbose`: 打印详细信息

## 输出结果

运行完成后，会在指定的结果目录（默认为 `results/`）生成：

1. **模型文件**：`semi_supervised_model.pkl`
2. **评估指标**：`test_metrics.json`
3. **预测结果**：`test_predictions.csv`
4. **可视化图表**：
   - `training_history.png`：训练历史
   - `predictions.png`：预测对比图
   - `residuals.png`：残差分析
   - `model_comparison.png`：模型性能对比
   - `uncertainty_analysis.png`：不确定性分析

## 关键参数说明

### 伪标签过滤参数

- `confidence_threshold`：不确定性过滤阈值（默认0.85）
- `reconstruction_threshold`：重建误差过滤阈值（默认0.9）
- `ensemble_agreement_threshold`：集成一致性过滤阈值（默认0.8）
- `min_pseudo_labels`：每轮最少伪标签数（默认50）

### 自训练参数

- `n_iterations`：自训练迭代次数（默认3）
- `high_confidence_percentile`：高置信度百分位（默认50）

## 注意事项

1. **数据质量**：确保输入数据没有缺失值和异常值
2. **计算资源**：深度学习部分可以使用GPU加速
3. **内存使用**：大数据集可能需要较多内存
4. **模型选择**：可以根据具体任务调整使用的集成模型

## 故障排除

1. **内存不足**：减少批量大小或使用更少的集成模型
2. **训练时间过长**：减少VAE训练轮数或集成模型数量
3. **伪标签过少**：降低过滤阈值或检查数据质量
4. **预测性能差**：增加标记数据或调整模型参数

## 扩展和定制

框架设计灵活，易于扩展：

1. **添加新模型**：在 `ensemble.py` 中的 `initialize_models` 方法添加
2. **自定义过滤器**：在 `step3_generate_high_quality_pseudo_labels` 中添加
3. **新的特征工程**：修改 `step1_feature_engineering_and_autoencoder`
4. **自定义可视化**：在 `visualization.py` 中添加新函数

## 参考文献

本框架基于以下技术：
- 变分自编码器特征学习
- 集成学习
- 半监督学习
- 不确定性估计

## 许可证

[添加您的许可证信息]

## 联系方式

[添加您的联系信息]