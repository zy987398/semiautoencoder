"""
自编码器模型定义
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Tuple, Optional


class DeepAutoEncoder(nn.Module):
    """深度自编码器"""
    
    def __init__(self, input_dim: int, latent_dims: List[int], dropout_rate: float = 0.2):
        super().__init__()
        
        # 构建编码器
        encoder_layers = []
        prev_dim = input_dim
        for latent_dim in latent_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = latent_dim
        
        # 移除最后一个dropout层
        self.encoder = nn.Sequential(*encoder_layers[:-1])
        
        # 构建解码器
        decoder_layers = []
        reversed_dims = latent_dims[::-1] + [input_dim]
        for i in range(len(reversed_dims) - 1):
            decoder_layers.extend([
                nn.Linear(reversed_dims[i], reversed_dims[i + 1]),
                nn.BatchNorm1d(reversed_dims[i + 1]) if i < len(reversed_dims) - 2 else nn.Identity(),
                nn.ReLU() if i < len(reversed_dims) - 2 else nn.Identity(),
                nn.Dropout(dropout_rate) if i < len(reversed_dims) - 2 else nn.Identity()
            ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码输入数据"""
        return self.encoder(x)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        z = self.encode(x)
        return self.decoder(z), z


class AutoEncoderTrainer:
    """自编码器训练器"""
    
    def __init__(self, 
                 input_dim: int,
                 latent_dims: List[int],
                 dropout_rate: float = 0.2,
                 device: Optional[str] = None):
        
        # 自动选择设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"AutoEncoder using device: {self.device}")
        
        self.model = DeepAutoEncoder(input_dim, latent_dims, dropout_rate).to(self.device)
        self.training_history = []
        
    def train(self,
              X_train: np.ndarray,
              epochs: int = 300,
              batch_size: int = 64,
              learning_rate: float = 1e-3,
              weight_decay: float = 1e-5,
              scheduler_patience: int = 20,
              scheduler_factor: float = 0.5,
              verbose: bool = True,
              use_amp: bool = True) -> List[float]:
        """
        训练自编码器
        
        Args:
            X_train: 训练数据
            epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            weight_decay: 权重衰减
            scheduler_patience: 学习率调度器耐心值
            scheduler_factor: 学习率调度器衰减因子
            verbose: 是否打印训练信息
            use_amp: 是否使用混合精度训练（GPU时自动启用）
            
        Returns:
            训练损失历史
        """
        # 设置优化器和调度器
        optimizer = optim.Adam(self.model.parameters(), 
                             lr=learning_rate, 
                             weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                        patience=scheduler_patience, 
                                                        factor=scheduler_factor)
        criterion = nn.MSELoss()
        
        # 混合精度训练（如果使用GPU）
        scaler = None
        if use_amp and self.device.type == 'cuda':
            scaler = torch.amp.GradScaler(self.device.type)
            if verbose:
                print("Using mixed precision training")
        
        # 创建数据加载器
        dataset = TensorDataset(torch.FloatTensor(X_train))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                              pin_memory=(self.device.type == 'cuda'),
                              num_workers=2 if self.device.type == 'cuda' else 0)
        
        # 训练循环
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_x, in dataloader:

                batch_x = batch_x.to(self.device)
                
                optimizer.zero_grad()
                
                if scaler is not None:
                    # 混合精度训练
                    with torch.amp.autocast(self.device.type):
                        x_recon, _ = self.model(batch_x)
                        loss = criterion(x_recon, batch_x)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 标准训练
                    x_recon, _ = self.model(batch_x)
                    loss = criterion(x_recon, batch_x)
                    loss.backward()
                    optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            self.training_history.append(avg_loss)
            scheduler.step(avg_loss)
            
            if verbose and epoch % 50 == 0:
                print(f"Epoch {epoch}, Reconstruction Loss: {avg_loss:.6f}")
                if self.device.type == 'cuda':
                    print(f"GPU Memory: {torch.cuda.memory_allocated(self.device) / 1024**2:.2f} MB")
        
        if verbose:
            print(f"AutoEncoder training completed. Final loss: {avg_loss:.6f}")
            
        return self.training_history
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """编码数据"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            encoded = self.model.encode(X_tensor).cpu().numpy()
        return encoded
    
    def get_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """计算重建误差"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstructions, _ = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructions)**2, dim=1).cpu().numpy()
        return errors
    
    def encode_and_get_errors(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """同时获取编码特征和重建误差"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            encoded = self.model.encode(X_tensor)
            reconstructions, _ = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructions)**2, dim=1).cpu().numpy()
            encoded = encoded.cpu().numpy()
        return encoded, errors
