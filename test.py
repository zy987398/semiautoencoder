"""
测试脚本：验证项目更新是否成功
"""
import numpy as np
import sys
import traceback
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def test_improved_vae():
    """测试改进的VAE架构"""
    print("Testing Improved VAE Architecture...")
    try:
        from models.vae import VAETrainer, ImprovedVariationalAutoEncoder
        
        # 创建测试数据
        X_test = np.random.randn(100, 20).astype(np.float32)
        
        # 测试改进的VAE
        trainer = VAETrainer(
            input_dim=20,
            latent_dims=[16, 8],
            use_improved_vae=True,
            use_attention=True,
            use_residual=True
        )
        
        # 简单训练测试
        history = trainer.train(X_test, epochs=5, verbose=False)
        
        # 测试编码
        encoded = trainer.encode(X_test)
        assert encoded.shape == (100, 8), f"Expected shape (100, 8), got {encoded.shape}"
        
        print("✓ Improved VAE test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Improved VAE test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_advanced_pseudo_labeling():
    """测试高级伪标签生成器"""
    print("\nTesting Advanced Pseudo Label Generator...")
    try:
        from core.advanced_pseudo_labeling import AdvancedPseudoLabelGenerator
        from models.vae import VAETrainer
        from models.ensemble import EnsembleUncertaintyEstimator
        import config
        
        # 创建测试数据
        X_unlabeled = np.random.randn(200, 10).astype(np.float32)
        
        # 创建模拟的组件
        # VAE训练器
        vae_trainer = VAETrainer(
            input_dim=10,
            latent_dims=[8, 4],
            use_improved_vae=False  # 使用标准版本以加快测试
        )
        vae_trainer.train(X_unlabeled, epochs=5, verbose=False)
        
        # 集成估计器（简化版）
        class MockEnsembleEstimator:
            def predict_with_uncertainty(self, X):
                n = len(X)
                predictions = np.random.randn(n)
                uncertainties = np.abs(np.random.randn(n)) * 0.1
                individual_preds = {
                    'model1': predictions + np.random.randn(n) * 0.1,
                    'model2': predictions + np.random.randn(n) * 0.1
                }
                return predictions, uncertainties, individual_preds
        
        ensemble_estimator = MockEnsembleEstimator()
        
        # 特征处理器
        def feature_processor(X):
            return X  # 简化处理
        
        # 创建伪标签生成器
        generator = AdvancedPseudoLabelGenerator(
            base_config=config.PSEUDO_LABEL_CONFIG,
            use_active_learning=True,
            use_clustering=True
        )
        
        # 生成伪标签
        pseudo_features, pseudo_targets, pseudo_weights, stats = generator.generate_pseudo_labels(
            X_unlabeled=X_unlabeled,
            ensemble_estimator=ensemble_estimator,
            autoencoder_trainer=vae_trainer,
            feature_processor=feature_processor,
            iteration=0
        )
        
        # 验证结果
        assert len(pseudo_features) > 0, "No pseudo labels generated"
        assert len(pseudo_features) == len(pseudo_targets), "Features and targets length mismatch"
        assert 'selected_samples' in stats, "Missing statistics"
        
        print(f"✓ Advanced Pseudo Labeling test passed!")
        print(f"  - Generated {len(pseudo_features)} pseudo labels from {len(X_unlabeled)} samples")
        print(f"  - Selection rate: {stats['selection_rate']:.3f}")
        return True
        
    except Exception as e:
        print(f"✗ Advanced Pseudo Labeling test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_integrated_pipeline():
    """测试集成管道"""
    print("\nTesting Integrated Pipeline...")
    try:
        from core.semi_supervised import EnhancedSemiSupervisedEnsemble
        import config
        
        # 创建小规模测试数据
        n_labeled = 50
        n_unlabeled = 100
        n_features = 5
        
        X_labeled = np.random.randn(n_labeled, n_features).astype(np.float32)
        y_labeled = np.random.randn(n_labeled).astype(np.float32)
        X_unlabeled = np.random.randn(n_unlabeled, n_features).astype(np.float32)
        
        # 修改配置以加快测试
        test_config = config.AUTOENCODER_CONFIG.copy()
        test_config['epochs'] = 5
        test_config['use_improved_vae'] = True
        
        # 创建模型
        model = EnhancedSemiSupervisedEnsemble(
            autoencoder_config=test_config,
            use_advanced_pseudo_labeling=True,
            use_improved_vae=True
        )
        
        # 测试步骤1
        model.step1_feature_engineering_and_autoencoder(
            X_labeled, X_unlabeled, verbose=False
        )
        
        print("✓ Integrated Pipeline test passed!")
        print("  - VAE training completed")
        print("  - Advanced pseudo labeling enabled")
        return True
        
    except Exception as e:
        print(f"✗ Integrated Pipeline test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_config_updates():
    """测试配置更新"""
    print("\nTesting Configuration Updates...")
    try:
        import config
        
        # 检查新配置项
        required_configs = [
            ('AUTOENCODER_CONFIG', ['use_improved_vae', 'use_attention', 'use_residual']),
            ('SEMI_SUPERVISED_CONFIG', ['use_advanced_pseudo_labeling', 'use_active_learning']),
            ('PSEUDO_LABEL_CONFIG', ['quality_weights', 'adaptive_threshold_decay']),
            ('OPTIMIZATION_CONFIG', ['use_mixed_precision', 'parallel_backend']),
            ('EVALUATION_CONFIG', ['compute_shap', 'noise_levels'])
        ]
        
        all_passed = True
        for config_name, required_keys in required_configs:
            if hasattr(config, config_name):
                config_dict = getattr(config, config_name)
                for key in required_keys:
                    if key not in config_dict:
                        print(f"  ✗ Missing key '{key}' in {config_name}")
                        all_passed = False
            else:
                print(f"  ✗ Missing configuration: {config_name}")
                all_passed = False
        
        if all_passed:
            print("✓ Configuration test passed!")
        return all_passed
        
    except Exception as e:
        print(f"✗ Configuration test failed: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("="*60)
    print("Running Update Verification Tests")
    print("="*60)
    
    tests = [
        test_config_updates,
        test_improved_vae,
        test_advanced_pseudo_labeling,
        test_integrated_pipeline
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    # 总结
    print("="*60)
    print("Test Summary:")
    print(f"  - Total tests: {len(tests)}")
    print(f"  - Passed: {sum(results)}")
    print(f"  - Failed: {len(tests) - sum(results)}")
    
    if all(results):
        print("\n✅ All tests passed! The update was successful.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()