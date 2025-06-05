"""
环境检查脚本 - 验证GPU和依赖包设置
"""
import sys
import subprocess
import pkg_resources
import psutil

def check_python_version():
    """检查Python版本"""
    print("=" * 60)
    print("Python Version Check")
    print("=" * 60)
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("⚠️  WARNING: Python 3.7+ is recommended")
    else:
        print("✅ Python version is compatible")


def check_cuda_installation():
    """检查CUDA安装"""
    print("\n" + "=" * 60)
    print("CUDA Installation Check")
    print("=" * 60)
    
    try:
        # 检查nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA driver is installed")
            print("\nGPU Information:")
            # 提取GPU信息
            lines = result.stdout.split('\n')
            for line in lines:
                if 'NVIDIA' in line and 'CUDA' in line:
                    print(line.strip())
        else:
            print("❌ nvidia-smi not found. GPU driver may not be installed.")
    except FileNotFoundError:
        print("❌ nvidia-smi command not found. NVIDIA driver is not installed.")
    
    # 检查CUDA版本
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("\n✅ CUDA compiler is installed")
            # 提取CUDA版本
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    print(f"CUDA version: {line.strip()}")
        else:
            print("\n⚠️  CUDA compiler (nvcc) not found")
    except FileNotFoundError:
        print("\n⚠️  CUDA compiler (nvcc) not found")


def check_pytorch_gpu():
    """检查PyTorch GPU支持"""
    print("\n" + "=" * 60)
    print("PyTorch GPU Support Check")
    print("=" * 60)
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        # 检查CUDA可用性
        if torch.cuda.is_available():
            print(f"✅ CUDA is available in PyTorch")
            print(f"✅ CUDA version (PyTorch): {torch.version.cuda}")
            print(f"✅ Number of GPUs: {torch.cuda.device_count()}")
            
            # 显示所有GPU信息
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  - Compute capability: {props.major}.{props.minor}")
                print(f"  - Total memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"  - Multiprocessors: {props.multi_processor_count}")
                
            # 测试GPU操作
            print("\n🔧 Testing GPU operations...")
            try:
                # 创建张量并移到GPU
                x = torch.randn(1000, 1000).cuda()
                y = torch.randn(1000, 1000).cuda()
                z = torch.matmul(x, y)
                torch.cuda.synchronize()
                print("✅ GPU tensor operations work correctly")
                
                # 清理
                del x, y, z
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"❌ GPU operation failed: {e}")
        else:
            print("❌ CUDA is NOT available in PyTorch")
            print("\nPossible reasons:")
            print("1. No NVIDIA GPU installed")
            print("2. NVIDIA driver not installed")
            print("3. PyTorch installed without CUDA support")
            print("\nTo install PyTorch with CUDA support:")
            print("Visit: https://pytorch.org/get-started/locally/")
            
    except ImportError:
        print("❌ PyTorch is not installed")


def check_dependencies():
    """检查所有依赖包"""
    print("\n" + "=" * 60)
    print("Dependencies Check")
    print("=" * 60)
    
    required_packages = [
        'torch', 'torchvision', 'numpy', 'pandas', 'scikit-learn',
        'lightgbm', 'xgboost', 'catboost', 'ngboost',
        'matplotlib', 'seaborn', 'joblib'
    ]
    
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    
    all_installed = True
    for package in required_packages:
        if package in installed_packages:
            version = pkg_resources.get_distribution(package).version
            print(f"✅ {package:15} {version}")
        else:
            print(f"❌ {package:15} NOT INSTALLED")
            all_installed = False
    
    if all_installed:
        print("\n✅ All required packages are installed")
    else:
        print("\n❌ Some packages are missing. Run: pip install -r requirements.txt")


def check_memory():
    """检查系统内存"""
    print("\n" + "=" * 60)
    print("System Memory Check")
    print("=" * 60)
    
    try:

        
        # RAM信息
        ram = psutil.virtual_memory()
        print(f"Total RAM: {ram.total / 1024**3:.2f} GB")
        print(f"Available RAM: {ram.available / 1024**3:.2f} GB")
        print(f"Used RAM: {ram.used / 1024**3:.2f} GB ({ram.percent}%)")
        
        # GPU内存信息
        try:
            import torch
            if torch.cuda.is_available():
                print("\nGPU Memory:")
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    total = props.total_memory / 1024**3
                    
                    print(f"GPU {i}:")
                    print(f"  - Total: {total:.2f} GB")
                    print(f"  - Allocated: {allocated:.2f} GB")
                    print(f"  - Reserved: {reserved:.2f} GB")
                    print(f"  - Free: {total - reserved:.2f} GB")
        except:
            pass
            
    except ImportError:
        print("⚠️  psutil not installed. Cannot check system memory.")
        print("Install with: pip install psutil")


def performance_test():
    """简单的性能测试"""
    print("\n" + "=" * 60)
    print("Performance Test")
    print("=" * 60)
    
    try:
        import torch
        import time
        
        # 矩阵大小
        sizes = [1000, 2000, 4000]
        
        for size in sizes:
            print(f"\nMatrix multiplication ({size}x{size}):")
            
            # CPU测试
            x_cpu = torch.randn(size, size)
            y_cpu = torch.randn(size, size)
            
            start = time.time()
            z_cpu = torch.matmul(x_cpu, y_cpu)
            cpu_time = time.time() - start
            print(f"  CPU time: {cpu_time:.4f} seconds")
            
            # GPU测试（如果可用）
            if torch.cuda.is_available():
                x_gpu = x_cpu.cuda()
                y_gpu = y_cpu.cuda()
                
                # 预热
                _ = torch.matmul(x_gpu, y_gpu)
                torch.cuda.synchronize()
                
                start = time.time()
                z_gpu = torch.matmul(x_gpu, y_gpu)
                torch.cuda.synchronize()
                gpu_time = time.time() - start
                
                print(f"  GPU time: {gpu_time:.4f} seconds")
                print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
                
                # 清理
                del x_gpu, y_gpu, z_gpu
                torch.cuda.empty_cache()
            
            del x_cpu, y_cpu, z_cpu
            
    except Exception as e:
        print(f"Performance test failed: {e}")


def main():
    """主函数"""
    print("🔍 Semi-Supervised Learning Framework - Environment Check")
    print("=" * 60)
    
    # 运行所有检查
    check_python_version()
    check_cuda_installation()
    check_pytorch_gpu()
    check_dependencies()
    check_memory()
    performance_test()
    
    # 总结
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ GPU is available and configured correctly!")
            print("🚀 You can use GPU acceleration for training.")
        else:
            print("⚠️  GPU is not available.")
            print("💡 Training will use CPU (slower but still functional).")
    except:
        print("❌ PyTorch is not installed correctly.")
    
    print("\n📝 To use GPU acceleration:")
    print("1. Ensure you have an NVIDIA GPU")
    print("2. Install NVIDIA drivers")
    print("3. Install PyTorch with CUDA support")
    print("4. Run your training script")


if __name__ == "__main__":
    main()
