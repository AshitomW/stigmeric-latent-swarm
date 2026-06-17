import os
import threading
import numpy as np
import torch
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import STGVAE
from swarm import LatentSwarmExplorer
from data_generator import generate_spring_system_data

app = FastAPI(title="Stigmergic Latent Swarms")

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
training_state = {
    "status": "idle",
    "epoch": 0,
    "max_epochs": 0,
    "loss": 0.0,
    "val_loss": 0.0,
    "message": ""
}

# Explorer instance
explorer = None
model = None
edge_index = None
pca_mean = None
pca_components = None

class ExploreConfig(BaseModel):
    objective: str = "spread"
    num_agents: int = 15
    w_inertia: float = 0.7
    c_stigmergy: float = 0.3
    c_mutual: float = 0.3
    c_goal: float = 0.5
    dt: float = 0.1

def run_training_in_background(epochs: int):
    global training_state, model, edge_index
    try:
        training_state["status"] = "running"
        training_state["epoch"] = 0
        training_state["max_epochs"] = epochs
        training_state["message"] = "Initializing datasets..."
        
        # 1. Generate data if not exists
        data_path = "backend/data/spring_data.npz"
        if not os.path.exists(data_path):
            training_state["message"] = "Generating synthetic data..."
            data, edge_index = generate_spring_system_data(num_simulations=100)
            os.makedirs("backend/data", exist_ok=True)
            np.savez(data_path, data=data, edge_index=edge_index)
            
        # 2. Set up training parameters
        from train import TrajectoryDataset, generate_figures
        import torch.nn.functional as F
        
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        dataset = TrajectoryDataset(data_path, seq_len=30, stride=15)
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        num_nodes = dataset[0].size(1)
        in_features = dataset[0].size(2)
        
        model = STGVAE(num_nodes=num_nodes, in_features=in_features, spatial_dim=32, temporal_dim=64, latent_dim=256).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        
        train_losses = []
        val_losses = []
        val_recon = []
        val_kl = []
        
        edge_index_dev = dataset.edge_index.to(device)
        
        beta_target = 0.0005
        anneal_epochs = 50
        
        for epoch in range(1, epochs + 1):
            training_state["epoch"] = epoch
            
            # Three-phase training:
            # Phase 1 (epochs 1-10): Deterministic autoencoder
            # Phase 2 (epochs 11-30): Gradually anneal noise 0→1
            # Phase 3 (epochs 31+): Full VAE with gentle KL pressure
            if epoch <= 10:
                current_beta = 0.0
                noise_scale = 0.0
            elif epoch <= anneal_epochs:
                progress = (epoch - 10) / (anneal_epochs - 10)
                current_beta = beta_target * progress
                noise_scale = min(1.0, (epoch - 10) / 20)
            else:
                current_beta = beta_target
                noise_scale = 1.0
                
            training_state["message"] = f"Training epoch {epoch}/{epochs} (Beta: {current_beta:.6f})..."
            
            model.train()
            epoch_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                recon, mu, logvar = model(batch, edge_index_dev, noise_scale=noise_scale)
                recon_loss = F.mse_loss(recon, batch)
                kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
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
                    recon, mu, logvar = model(batch, edge_index_dev, noise_scale=noise_scale)
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
            val_recon.append(epoch_val_recon)
            val_kl.append(epoch_val_kl)
            
            training_state["loss"] = epoch_loss
            training_state["val_loss"] = epoch_val_loss
            
        # Save model and normalization stats
        os.makedirs("backend/weights", exist_ok=True)
        torch.save(model.state_dict(), "backend/weights/st_gvae.pt")
        np.savez("backend/weights/normalization_stats.npz",
                 mean=dataset.mean.numpy(),
                 std=dataset.std.numpy())
        generate_figures(train_losses, val_losses, val_recon, val_kl, model, val_loader, edge_index_dev, device)
        
        # Reload PCA data
        load_pca_data()
        
        training_state["status"] = "completed"
        training_state["message"] = "Model trained successfully!"
    except Exception as e:
        training_state["status"] = "error"
        training_state["message"] = f"Training failed: {str(e)}"
        print(f"Error in background training: {e}")

def load_pca_data():
    global pca_mean, pca_components
    pca_path = "backend/data/pca_projection.npz"
    if os.path.exists(pca_path):
        pca_data = np.load(pca_path)
        pca_mean = torch.tensor(pca_data['mean'], dtype=torch.float32)
        pca_components = torch.tensor(pca_data['components'], dtype=torch.float32)

def init_model_if_needed():
    global model, edge_index
    if model is not None:
        model = model.to(torch.device('cpu'))
        if edge_index is not None:
            edge_index = edge_index.to(torch.device('cpu'))
        return
        
    device = torch.device('cpu')
    weights_path = "backend/weights/st_gvae.pt"
    data_path = "backend/data/spring_data.npz"
    
    if os.path.exists(weights_path) and os.path.exists(data_path):
        npz_data = np.load(data_path)
        raw_data = npz_data['data']
        edge_index = torch.tensor(npz_data['edge_index'], dtype=torch.long)
        
        num_nodes = raw_data.shape[2]
        in_features = raw_data.shape[3]
        
        model = STGVAE(num_nodes=num_nodes, in_features=in_features, spatial_dim=32, temporal_dim=64, latent_dim=256).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        
        # Load normalization stats
        norm_path = "backend/weights/normalization_stats.npz"
        if os.path.exists(norm_path):
            norm_data = np.load(norm_path)
            model.register_buffer('data_mean', torch.tensor(norm_data['mean'], dtype=torch.float32, device=device))
            model.register_buffer('data_std', torch.tensor(norm_data['std'], dtype=torch.float32, device=device))
        
        load_pca_data()

@app.get("/api/status")
def get_status():
    data_path = "backend/data/spring_data.npz"
    weights_path = "backend/weights/st_gvae.pt"
    return {
        "data_exists": os.path.exists(data_path),
        "model_exists": os.path.exists(weights_path),
        "training": training_state
    }

@app.post("/api/train")
def train_model_endpoint(background_tasks: BackgroundTasks, epochs: int = 30):
    global training_state
    if training_state["status"] == "running":
        raise HTTPException(status_code=400, detail="Training is already running.")
        
    training_state["status"] = "running"
    background_tasks.add_task(run_training_in_background, epochs)
    return {"status": "started"}

@app.get("/api/trajectories")
def get_trajectories(num_samples: int = 5):
    data_path = "backend/data/spring_data.npz"
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Data not generated yet. Please train the model to generate data.")
    
    npz_data = np.load(data_path)
    data = npz_data['data']  # (sims, steps, nodes, 6)
    
    # Return first few trajectories, positions only (first 3 channels: x, y, z)
    # Downsample time steps to 200 for easier frontend loading/rendering
    downsample_factor = 5
    sampled_data = data[:num_samples, ::downsample_factor, :, :3].tolist()
    
    return {
        "trajectories": sampled_data,
        "edge_index": npz_data['edge_index'].tolist()
    }

@app.post("/api/explore/start")
def explore_start(config: ExploreConfig):
    global explorer, model, edge_index
    init_model_if_needed()
    if model is None:
        raise HTTPException(status_code=400, detail="Model weights not found. Please train the model first.")
        
    device = torch.device('cpu')
    
    # Pass normalization stats from model for denormalizing decoded outputs
    data_mean = getattr(model, 'data_mean', None)
    data_std = getattr(model, 'data_std', None)
    
    # Instantiate explorer in latent space
    explorer = LatentSwarmExplorer(
        model=model,
        edge_index=edge_index.to(device),
        num_agents=config.num_agents,
        latent_dim=model.latent_dim,
        seq_len=30,
        device=device,
        data_mean=data_mean,
        data_std=data_std
    )
    return {"status": "initialized", "num_agents": config.num_agents, "objective": config.objective}

@app.post("/api/explore/step")
def explore_step(config: ExploreConfig):
    global explorer, pca_mean, pca_components
    if explorer is None:
        # Auto-initialize
        explore_start(config)
        
    # Perform one swarm simulation step
    # traj shape: (num_agents, T, N, 6)
    # z shape: (num_agents, 8)
    # fitness shape: (num_agents,)
    traj, z, fitness = explorer.step(
        objective=config.objective,
        w_inertia=config.w_inertia,
        c_stigmergy=config.c_stigmergy,
        c_mutual=config.c_mutual,
        c_goal=config.c_goal,
        dt=config.dt
    )
    
    # Project latent positions to 2D for PCA plot
    z_2d = np.zeros((config.num_agents, 2))
    if pca_mean is not None and pca_components is not None:
        z_tensor = torch.tensor(z, dtype=torch.float32)
        z_centered = z_tensor - pca_mean
        z_proj = z_centered @ pca_components
        z_2d = z_proj.numpy()
    else:
        # Fallback to simple slice if PCA data not found
        z_2d = z[:, :2]
        
    # Return physical positions (x, y, z) and velocities for representation
    # Shape of return trajectories: (num_agents, T, N, 3) (positions only)
    return {
        "trajectories": traj[:, :, :, :3].tolist(),
        "latent_positions": z_2d.tolist(),
        "fitness": fitness.tolist()
    }

@app.post("/api/explore/reset")
def explore_reset():
    global explorer
    explorer = None
    return {"status": "reset"}

# Mount frontend files
# Note: we will mount static files at '/' so that the client-side code is served from port 8000
frontend_dir = os.path.abspath("frontend")
os.makedirs(frontend_dir, exist_ok=True)
os.makedirs("backend/data", exist_ok=True)

# Mount generated backend figures for UI serving
app.mount("/backend/data", StaticFiles(directory="backend/data"), name="backend_data")
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    # Auto-load PCA mapping if it exists
    load_pca_data()
    # Start server
    uvicorn.run(app, host="127.0.0.1", port=8000)
