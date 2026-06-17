import numpy as np
import torch
import torch.nn.functional as F
from model import STGVAE
from train import TrajectoryDataset

def debug_gradients():
    # 1. Load data and print stats
    data_path = "backend/data/spring_data.npz"
    npz_data = np.load(data_path)
    raw_data = npz_data['data']
    print("Dataset Stats:")
    print("  Shape:", raw_data.shape)
    print("  Mean:", np.mean(raw_data))
    print("  Std:", np.std(raw_data))
    print("  Min:", np.min(raw_data))
    print("  Max:", np.max(raw_data))
    
    # Load dataset
    dataset = TrajectoryDataset(data_path, seq_len=30, stride=15)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=4)))
    edge_index = dataset.edge_index
    
    # 2. Test on CPU vs MPS
    devices = ['cpu']
    if torch.backends.mps.is_available():
        devices.append('mps')
        
    for device_name in devices:
        device = torch.device(device_name)
        print(f"\n--- Testing on {device_name.upper()} ---")
        
        # Instantiate model
        model = STGVAE(num_nodes=5, in_features=6, latent_dim=8).to(device)
        x = batch.to(device)
        edges = edge_index.to(device)
        
        # Forward pass
        recon, mu, logvar = model(x, edges)
        
        # Loss
        recon_loss = F.mse_loss(recon, x)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
        loss = recon_loss + 0.05 * kl_loss
        
        print(f"  Forward Pass Success. Loss: {loss.item():.5f} (Recon: {recon_loss.item():.5f}, KL: {kl_loss.item():.5f})")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        grad_norms = []
        zero_grad_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                norm = torch.norm(param.grad).item()
                grad_norms.append((name, norm))
                if norm == 0.0:
                    zero_grad_count += 1
            else:
                grad_norms.append((name, None))
                
        print(f"  Gradients checked: {len(grad_norms)} parameters.")
        print(f"  Zero gradient count: {zero_grad_count}")
        print("  Sample Gradients (top 5 by norm):")
        valid_grads = [(n, v) for n, v in grad_norms if v is not None]
        valid_grads.sort(key=lambda x: x[1], reverse=True)
        for name, norm in valid_grads[:5]:
            print(f"    {name}: {norm:.6f}")

if __name__ == "__main__":
    debug_gradients()
