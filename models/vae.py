"""Variational AutoEncoder model and trainer"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Tuple, Optional


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


class VAETrainer:
    """Trainer for Variational AutoEncoder"""

    def __init__(self,
                 input_dim: int,
                 latent_dims: List[int],
                 dropout_rate: float = 0.2,
                 device: Optional[str] = None,
                 kl_weight: float = 1e-3):

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"VAE using device: {self.device}")

        self.model = VariationalAutoEncoder(input_dim, latent_dims, dropout_rate).to(self.device)
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
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()

                if scaler is not None:
                    with torch.amp.autocast(self.device.type):
                        recon, mu, logvar, _ = self.model(batch_x)
                        recon_loss = criterion(recon, batch_x)
                        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                        loss = recon_loss + self.kl_weight * kl_loss
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    recon, mu, logvar, _ = self.model(batch_x)
                    recon_loss = criterion(recon, batch_x)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + self.kl_weight * kl_loss
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
