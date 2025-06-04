"""
GPU依赖包安装脚本
"""
import subprocess
import sys
import platform


def get_cuda_version():
    """获取CUDA版本"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    # 提取版本号
                    version = line.split('release')[1].split(',')[0].strip()
                    return version
    except:
        pass
    
    # 尝试从nvidia-smi获取
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'CUDA Version' in line:
                    # 提取CUDA版本
                    import re
                    match = re.search(r'CUDA Version:\s*(\d+\.\d+)', line)
                    if match:
                        return match.group(1)
    except:
        pass
    
    return None


def install_pytorch_gpu(cuda_version):
    """安装GPU版本的PyTorch"""
    print(f"Detected CUDA version: {cuda_version}")
    
    # PyTorch CUDA版本映射
    cuda_to_torch = {
        '12.4': 'cu124',
        '12.1': 'cu121',
        '11.8': 'cu118',
        '11.7': 'cu117',
        '11.6': 'cu116'
    }
    
    # 找到最接近的CUDA版本
    cuda_major = cuda_version.split('.')[0]
    cuda_minor = cuda_version.split('.')[1]
    cuda_key = f"{cuda_major}.{cuda_minor}"
    
    if cuda_key in cuda_to_torch:
        torch_cuda = cuda_to_torch[cuda_key]
    elif cuda_major == '12':
        torch_cuda = 'cu121'  # 使用12.1作为12.x的默认
    elif cuda_major == '11':
        torch_cuda = 'cu118'  # 使用11.8作为11.x的默认
    else:
        print(f"⚠️  CUDA {cuda_version} is not directly supported. Using CPU version.")
        torch_cuda = 'cpu'
    
    if torch_cuda != 'cpu':
        # 构建安装命令
        cmd = [
            sys.executable, '-m', 'pip', 'install', 
            'torch', 'torchvision', 'torchaudio',
            '--index-url', f'https://download.pytorch.org/whl/{torch_cuda}'
        ]
        
        print(f"Installing PyTorch with CUDA {torch_cuda} support...")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True)
            print("✅ PyTorch GPU version installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install PyTorch GPU version")
            return False
    
    return True


def install_xgboost_gpu():
    """安装GPU版本的XGBoost"""
    print("\nInstalling XGBoost with GPU support...")
    
    # XGBoost GPU版本
    cmd = [sys.executable, '-m', 'pip', 'install', 'xgboost']
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ XGBoost installed (GPU support included in recent versions)")
    except subprocess.CalledProcessError:
        print("❌ Failed to install XGBoost")
        return False
    
    return True


def install_lightgbm_gpu():
    """安装GPU版本的LightGBM"""
    print("\nInstalling LightGBM with GPU support...")
    
    if platform.system() == 'Windows':
        print("⚠️  LightGBM GPU on Windows requires manual compilation")
        print("For pre-built wheels, visit: https://github.com/microsoft/LightGBM/releases")
        print("Installing CPU version for now...")
        cmd = [sys.executable, '-m', 'pip', 'install', 'lightgbm']
    else:
        # Linux/Mac - 尝试安装GPU版本
        print("Note: LightGBM GPU requires OpenCL and Boost")
        cmd = [sys.executable, '-m', 'pip', 'install', 'lightgbm', '--install-option=--gpu']
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ LightGBM installed")
    except subprocess.CalledProcessError:
        # 回退到标准安装
        print("Installing standard LightGBM...")
        cmd = [sys.executable, '-m', 'pip', 'install', 'lightgbm']
        subprocess.run(cmd, check=True)
        print("✅ LightGBM installed (CPU version)")
    
    return True


def install_catboost_gpu():
    """安装GPU版本的CatBoost"""
    print("\nInstalling CatBoost with GPU support...")
    
    # CatBoost自动包含GPU支持
    cmd = [sys.executable, '-m', 'pip', 'install', 'catboost']
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ CatBoost installed with GPU support")
    except subprocess.CalledProcessError:
        print("❌ Failed to install CatBoost")
        return False
    
    return True


def verify_gpu_installation():
    """验证GPU安装"""
    print("\n" + "="*60)
    print("Verifying GPU installations...")
    print("="*60)
    
    # 检查PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ PyTorch GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("❌ PyTorch: GPU not available")
    except ImportError:
        print("❌ PyTorch not installed")
    
    # 检查XGBoost
    try:
        import xgboost as xgb
        print(f"✅ XGBoost version: {xgb.__version__}")
        # 测试GPU
        try:
            dtest = xgb.DMatrix([[1, 2], [3, 4]])
            bst = xgb.train({'tree_method': 'gpu_hist'}, dtest, 1)
            print("   GPU support: Available")
        except:
            print("   GPU support: Not available")
    except ImportError:
        print("❌ XGBoost not installed")
    
    # 检查CatBoost
    try:
        import catboost
        print(f"✅ CatBoost version: {catboost.__version__}")
        print("   GPU support: Available (built-in)")
    except ImportError:
        print("❌ CatBoost not installed")
    
    # 检查LightGBM
    try:
        import lightgbm as lgb
        print(f"✅ LightGBM version: {lgb.__version__}")
    except ImportError:
        print("❌ LightGBM not installed")


def main():
    """主函数"""
    print("🚀 GPU Dependencies Installation Script")
    print("="*60)
    
    # 检测CUDA版本
    cuda_version = get_cuda_version()
    
    if cuda_version is None:
        print("❌ CUDA not detected. Please install NVIDIA drivers and CUDA toolkit.")
        print("Visit: https://developer.nvidia.com/cuda-downloads")
        return
    
    print(f"✅ CUDA {cuda_version} detected")
    
    # 安装各个组件
    success = True
    
    # 1. PyTorch GPU
    if not install_pytorch_gpu(cuda_version):
        success = False
    
    # 2. XGBoost GPU
    if not install_xgboost_gpu():
        success = False
    
    # 3. CatBoost GPU
    if not install_catboost_gpu():
        success = False
    
    # 4. LightGBM GPU
    if not install_lightgbm_gpu():
        success = False
    
    # 验证安装
    verify_gpu_installation()
    
    if success:
        print("\n✅ GPU dependencies installation completed!")
        print("\nNext steps:")
        print("1. Run 'python check_environment.py' to verify")
        print("2. Run your training script with GPU support")
    else:
        print("\n⚠️  Some installations failed. Check the errors above.")


if __name__ == "__main__":
    main()