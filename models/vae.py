"""Variational AutoEncoder model and trainer - Enhanced Version"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Tuple, Optional


# ===== 原始VAE架构（保持兼容性） =====
class VariationalAutoEncoder(nn.Module):
    """Deep Variational AutoEncoder"""

    def __init__(self, input_dim: int, latent_dims: List[int], dropout_rate: float = 0.2):
        super().__init__()

        assert len(latent_dims) >= 1, "latent_dims should contain at least one dimension"

        encoder_layers = []
        prev_dim = input_dim
        for dim in latent_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim

        # remove last dropout
        self.encoder = nn.Sequential(*encoder_layers[:-1])
        self.fc_mu = nn.Linear(latent_dims[-1], latent_dims[-1])
        self.fc_logvar = nn.Linear(latent_dims[-1], latent_dims[-1])

        decoder_layers = []
        reversed_dims = latent_dims[::-1] + [input_dim]
        for i in range(len(reversed_dims) - 1):
            decoder_layers.extend([
                nn.Linear(reversed_dims[i], reversed_dims[i + 1]),
                nn.BatchNorm1d(reversed_dims[i + 1]) if i < len(reversed_dims) - 2 else nn.Identity(),
                nn.ReLU() if i < len(reversed_dims) - 2 else nn.Identity(),
                nn.Dropout(dropout_rate) if i < len(reversed_dims) - 2 else nn.Identity(),
            ])
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z


# ===== 新增：改进的VAE架构 =====
class AttentionBlock(nn.Module):
    """自注意力块"""
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 添加序列维度
        x = x.unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        return x.squeeze(1)


class ImprovedVariationalAutoEncoder(nn.Module):
    """改进的VAE，包含注意力机制和残差连接"""
    
    def __init__(self, 
                 input_dim: int, 
                 latent_dims: List[int], 
                 dropout_rate: float = 0.2,
                 use_attention: bool = True,
                 use_residual: bool = True):
        super().__init__()
        
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # 编码器
        encoder_layers = []
        residual_dims = []
        prev_dim = input_dim
        
        for i, dim in enumerate(latent_dims):
            # 主要层
            encoder_layers.append(nn.Sequential(
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.GELU(),  # 使用GELU代替ReLU
                nn.Dropout(dropout_rate)
            ))
            
            # 残差连接（如果维度匹配）
            if self.use_residual and i > 0 and prev_dim == dim:
                residual_dims.append(i)
            
            # 注意力层（在中间层添加）
            if self.use_attention and i == len(latent_dims) // 2:
                encoder_layers.append(AttentionBlock(dim))
            
            prev_dim = dim
        
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.residual_dims = residual_dims
        
        # VAE的均值和方差层
        self.fc_mu = nn.Linear(latent_dims[-1], latent_dims[-1])
        self.fc_logvar = nn.Linear(latent_dims[-1], latent_dims[-1])
        
        # 解码器
        decoder_layers = []
        reversed_dims = latent_dims[::-1] + [input_dim]
        
        for i in range(len(reversed_dims) - 1):
            decoder_layers.append(nn.Sequential(
                nn.Linear(reversed_dims[i], reversed_dims[i + 1]),
                nn.BatchNorm1d(reversed_dims[i + 1]) if i < len(reversed_dims) - 2 else nn.Identity(),
                nn.GELU() if i < len(reversed_dims) - 2 else nn.Identity(),
                nn.Dropout(dropout_rate) if i < len(reversed_dims) - 2 else nn.Identity(),
            ))
        
        self.decoder_layers = nn.ModuleList(decoder_layers)
        
        # 输出层不再限制到 [-1,1]
        self.output_activation = nn.Identity()
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x
        residuals = []
        
        for i, layer in enumerate(self.encoder_layers):
            if self.use_residual and i in self.residual_dims:
                residuals.append(h)
            
            h = layer(h)
            
            if self.use_residual and len(residuals) > 0 and i in self.residual_dims:
                h = h + residuals.pop()
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        for layer in self.decoder_layers:
            h = layer(h)
        return self.output_activation(h)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


# ===== 更新VAETrainer以支持新架构 =====
class VAETrainer:
    """Trainer for Variational AutoEncoder - Enhanced Version"""

    def __init__(self,
                 input_dim: int,
                 latent_dims: List[int],
                 dropout_rate: float = 0.2,
                 device: Optional[str] = None,
                 kl_weight: float = 1e-3,
                 use_improved_vae: bool = False,  # 新增参数
                 use_attention: bool = True,       # 新增参数
                 use_residual: bool = True):       # 新增参数

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"VAE using device: {self.device}")

        # 根据配置选择VAE架构
        if use_improved_vae:
            print("Using Improved VAE with attention and residual connections")
            self.model = ImprovedVariationalAutoEncoder(
                input_dim, latent_dims, dropout_rate,
                use_attention=use_attention,
                use_residual=use_residual
            ).to(self.device)
        else:
            self.model = VariationalAutoEncoder(
                input_dim, latent_dims, dropout_rate
            ).to(self.device)

        self.kl_weight = kl_weight
        self.training_history: List[float] = []

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
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience, factor=scheduler_factor)
        criterion = nn.MSELoss()

        scaler = None
        if use_amp and self.device.type == 'cuda':
            scaler = torch.amp.GradScaler(self.device.type)
            if verbose:
                print("Using mixed precision training")

        dataset = TensorDataset(torch.FloatTensor(X_train))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                pin_memory=(self.device.type == 'cuda'),
                                num_workers=2 if self.device.type == 'cuda' else 0)

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            # KL annealing: gradually increase KL weight after 20 epochs
            if epoch < 20:
                kl_w = 0.0
            else:
                progress = (epoch - 20) / max(1, (epochs - 20))
                kl_w = self.kl_weight * progress
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()

                if scaler is not None:
                    with torch.amp.autocast(self.device.type):
                        recon, mu, logvar, _ = self.model(batch_x)
                        recon_loss = criterion(recon, batch_x)
                        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                        loss = recon_loss + kl_w * kl_loss
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    recon, mu, logvar, _ = self.model(batch_x)
                    recon_loss = criterion(recon, batch_x)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + kl_w * kl_loss
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            self.training_history.append(avg_loss)
            scheduler.step(avg_loss)

            if verbose and epoch % 50 == 0:
                print(f"Epoch {epoch}, VAE Loss: {avg_loss:.6f}")
                if self.device.type == 'cuda':
                    mem = torch.cuda.memory_allocated(self.device) / 1024**2
                    print(f"GPU Memory: {mem:.2f} MB")

        if verbose:
            print(f"VAE training completed. Final loss: {avg_loss:.6f}")

        return self.training_history

    def encode(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            mu, _ = self.model.encode(X_tensor)
            encoded = mu.cpu().numpy()
        return encoded

    def get_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            recon, _, _, _ = self.model(X_tensor)
            errors = torch.mean((X_tensor - recon)**2, dim=1).cpu().numpy()
        return errors

    def encode_and_get_errors(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            mu, _ = self.model.encode(X_tensor)
            recon, _, _, _ = self.model(X_tensor)
            errors = torch.mean((X_tensor - recon)**2, dim=1).cpu().numpy()
            encoded = mu.cpu().numpy()
        return encoded, errors