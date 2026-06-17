import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Use non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import STGVAE

class TrajectoryDataset(Dataset):
    def __init__(self, data_path, seq_len=30, stride=15):
        """
        Loads spring simulation data and splits it into sliding window sequences.
        Normalizes features to zero mean and unit variance per feature dimension.
        """
        npz_data = np.load(data_path)
        # raw_data shape: (num_simulations, num_steps, num_particles, 6)
        raw_data = npz_data['data']
        self.edge_index = torch.tensor(npz_data['edge_index'], dtype=torch.long)
        
        self.sequences = []
        num_sims, num_steps, num_parts, features = raw_data.shape
        
        for sim in range(num_sims):
            sim_traj = raw_data[sim]  # (num_steps, num_particles, 6)
            for start in range(0, num_steps - seq_len + 1, stride):
                end = start + seq_len
                seq = sim_traj[start:end]  # (seq_len, num_particles, 6)
                self.sequences.append(seq)
                
        self.sequences = torch.tensor(np.array(self.sequences), dtype=torch.float32)
        
        # Compute per-feature normalization stats across all dimensions
        self.mean = self.sequences.mean(dim=(0, 1, 2))  # (6,) - position/velocity means
        self.std = self.sequences.std(dim=(0, 1, 2))    # (6,) - position/velocity stds
        
        # Normalize to zero mean, unit variance
        self.sequences = (self.sequences - self.mean) / (self.std + 1e-8)
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx]

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Training on device: {device}")
    
    os.makedirs("backend/data", exist_ok=True)
    os.makedirs("backend/weights", exist_ok=True)
    
    # 1. Load dataset
    data_path = "backend/data/spring_data.npz"
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Generating data first...")
        from data_generator import generate_spring_system_data
        data, edge_index = generate_spring_system_data(num_simulations=100)
        np.savez(data_path, data=data, edge_index=edge_index)
        
    dataset = TrajectoryDataset(data_path, seq_len=args.seq_len, stride=args.stride)
    edge_index = dataset.edge_index.to(device)
    
    # Train / Val Split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Dataset size: {len(dataset)} sequences (Train: {train_size}, Val: {val_size})")
    
    # Get metadata from first batch
    sample_seq = dataset[0]
    num_nodes = sample_seq.size(1)
    in_features = sample_seq.size(2)
    
    # 2. Instantiate Model
    model = STGVAE(
        num_nodes=num_nodes,
        in_features=in_features,
        spatial_dim=args.spatial_dim,
        temporal_dim=args.temporal_dim,
        latent_dim=args.latent_dim
    ).to(device)
    
    # Use AdamW for improved generalization as recommended
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    train_losses = []
    val_losses = []
    val_recon_losses = []
    val_kl_losses = []
    
    # 3. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        
        # Three-phase training:
        # Phase 1 (epochs 1-10): Deterministic autoencoder - build good reconstructions
        # Phase 2 (epochs 11-30): Gradually anneal noise 0→1 to prevent posterior collapse
        # Phase 3 (epochs 31+): Full VAE with gentle KL pressure
        if epoch <= 10:
            current_beta = 0.0
            noise_scale = 0.0
        elif epoch <= args.anneal_epochs:
            progress = (epoch - 10) / (args.anneal_epochs - 10)
            current_beta = args.beta * progress
            noise_scale = min(1.0, (epoch - 10) / 20)
        else:
            current_beta = args.beta
            noise_scale = 1.0
            
        for batch in train_loader:
            batch = batch.to(device) # (B, T, N, F)
            
            optimizer.zero_grad()
            recon, mu, logvar = model(batch, edge_index, noise_scale=noise_scale)
            
            # Reconstruction Loss (MSE)
            recon_loss = F.mse_loss(recon, batch)
            
            # KL Divergence
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
            
            # Total Loss (beta-VAE Loss with KL Annealing)
            loss = recon_loss + current_beta * kl_loss
            loss.backward()
            
            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
            
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Step the learning rate scheduler
        scheduler.step()
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        epoch_val_recon = 0
        epoch_val_kl = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, mu, logvar = model(batch, edge_index, noise_scale=noise_scale)
                
                recon_loss = F.mse_loss(recon, batch)
                kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
                loss = recon_loss + current_beta * kl_loss
                
                epoch_val_loss += loss.item() * batch.size(0)
                epoch_val_recon += recon_loss.item() * batch.size(0)
                epoch_val_kl += kl_loss.item() * batch.size(0)
                
        epoch_val_loss /= len(val_loader.dataset)
        epoch_val_recon /= len(val_loader.dataset)
        epoch_val_kl /= len(val_loader.dataset)
        
        val_losses.append(epoch_val_loss)
        val_recon_losses.append(epoch_val_recon)
        val_kl_losses.append(epoch_val_kl)
        
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch {epoch}/{args.epochs} (Beta: {current_beta:.4f}) - Loss: {epoch_loss:.5f} | Val Loss: {epoch_val_loss:.5f} (Recon: {epoch_val_recon:.5f}, KL: {epoch_val_kl:.5f})")
            
    # Save model weights
    torch.save(model.state_dict(), "backend/weights/st_gvae.pt")
    
    # Save normalization stats for inference
    np.savez("backend/weights/normalization_stats.npz", 
             mean=dataset.mean.numpy(), 
             std=dataset.std.numpy())
    print("Model weights saved to backend/weights/st_gvae.pt")
    print(f"Normalization stats saved (mean: {dataset.mean.numpy()}, std: {dataset.std.numpy()})")
    
    # 4. Generate Figures (Simple, Minimalist styling)
    generate_figures(train_losses, val_losses, val_recon_losses, val_kl_losses, model, val_loader, edge_index, device)
    
def generate_figures(train_losses, val_losses, val_recon, val_kl, model, val_loader, edge_index, device):
    """
    Creates and saves minimalist charts for the paper/UI.
    """
    # 1. Training Curve Plot
    plt.figure(figsize=(8, 4), dpi=150)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Customize standard minimalist styling
    plt.rcParams['font.sans-serif'] = 'Helvetica'
    plt.rcParams['axes.edgecolor'] = '#e5e7eb'
    plt.rcParams['axes.linewidth'] = 0.8
    
    plt.plot(train_losses, label='Train Loss', color='#2563eb', linewidth=1.5)
    plt.plot(val_losses, label='Val Loss', color='#dc2626', linewidth=1.5)
    plt.title('ST-GVAE Training Convergence', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Loss Value', fontsize=10)
    plt.legend(frameon=True, facecolor='#ffffff', edgecolor='#e5e7eb')
    plt.tight_layout()
    plt.savefig('backend/data/loss_curve.png', bbox_inches='tight')
    plt.close()
    
    # 2. Latent Space PCA Projection Plot
    model.eval()
    all_mus = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            mu, _ = model.encode(batch, edge_index)
            all_mus.append(mu.cpu().numpy())
    all_mus = np.concatenate(all_mus, axis=0)  # Shape: (M, Z)
    
    # Run PCA to project to 2D
    # Centering
    mean_mu = np.mean(all_mus, axis=0)
    centered_mus = all_mus - mean_mu
    # Covariance and Eigendecomposition
    cov = np.cov(centered_mus.T)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    # Get top 2 principal components
    idx = np.argsort(eig_vals)[::-1]
    top_vecs = eig_vecs[:, idx[:2]]
    projected = centered_mus @ top_vecs  # (M, 2)
    
    # Save projection matrix and mean for local deployment in swarm explorer
    np.savez('backend/data/pca_projection.npz', mean=mean_mu, components=top_vecs)
    
    plt.figure(figsize=(6, 5), dpi=150)
    plt.scatter(projected[:, 0], projected[:, 1], c='#2563eb', alpha=0.5, s=15, edgecolors='none')
    plt.title('Latent Space PCA Projection', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('PC 1', fontsize=10)
    plt.ylabel('PC 2', fontsize=10)
    plt.tight_layout()
    plt.savefig('backend/data/latent_space.png', bbox_inches='tight')
    plt.close()
    print("Figures saved successfully to backend/data/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--stride', type=int, default=15)
    parser.add_argument('--spatial_dim', type=int, default=32)
    parser.add_argument('--temporal_dim', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--beta', type=float, default=0.0005)
    parser.add_argument('--anneal_epochs', type=int, default=50)
    args = parser.parse_args()
    
    train_model(args)
