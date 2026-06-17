import numpy as np
import torch
import torch.nn.functional as F
from model import STGVAE
from train import TrajectoryDataset

def test_reconstruction():
    device = torch.device('cpu')
    data_path = "backend/data/spring_data.npz"
    dataset = TrajectoryDataset(data_path, seq_len=30, stride=15)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    
    num_nodes = dataset[0].size(1)
    in_features = dataset[0].size(2)
    edge_index = dataset.edge_index.to(device)
    
    model = STGVAE(num_nodes=num_nodes, in_features=in_features, spatial_dim=32, temporal_dim=64, latent_dim=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01) # Use higher LR to check if it moves
    
    print("Initial target sample (Batch 0, Step 0, Node 0):")
    first_batch = next(iter(loader))
    print(first_batch[0, 0, 0].numpy())
    
    # We will limit the training to 10 batches to debug fast
    debug_batches = []
    for i, batch in enumerate(loader):
        if i >= 10:
            break
        debug_batches.append(batch)
        
    for epoch in range(1, 11):
        model.train()
        epoch_loss = 0
        for step, batch in enumerate(debug_batches):
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch, edge_index)
            loss = F.mse_loss(recon, batch)
            loss.backward()
            
            # Print gradient norms for debugging
            grad_norms = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norms[name] = param.grad.norm().item()
                    
            optimizer.step()
            epoch_loss += loss.item()
            
            if epoch in [1, 2, 5, 10] and step == 0:
                print(f"Epoch {epoch}, Step {step} - Loss: {loss.item():.6f}")
                # Print some representative gradient norms
                print(f"  Gradients: enc_gat: {grad_norms.get('enc_gat.lin.weight', 0.0):.6f}, enc_gru: {grad_norms.get('enc_gru.weight_ih_l0', 0.0):.6f}, fc_dec_init: {grad_norms.get('fc_dec_init.weight', 0.0):.6f}, node_emb: {grad_norms.get('node_emb', 0.0):.6f}")
                
        # Eval print
        model.eval()
        with torch.no_grad():
            recon, _, _ = model(first_batch, edge_index)
            if epoch in [1, 2, 5, 10]:
                print(f"Epoch {epoch} Eval - Target:    {first_batch[0, 0, 0].numpy()}")
                print(f"Epoch {epoch} Eval - Predicted: {recon[0, 0, 0].numpy()}")

if __name__ == "__main__":
    test_reconstruction()
