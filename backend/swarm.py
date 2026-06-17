import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentSwarmExplorer:
    def __init__(
        self,
        model,
        edge_index,
        num_agents=20,
        latent_dim=8,
        seq_len=30,
        device='cpu',
        data_mean=None,
        data_std=None
    ):
        self.model = model
        self.edge_index = edge_index
        self.num_agents = num_agents
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.device = device
        self.data_mean = data_mean
        self.data_std = data_std
        
        # Initialize agent positions randomly inside [-2, 2]^Z
        self.positions = torch.randn(num_agents, latent_dim, device=device) * 1.5
        self.velocities = torch.randn(num_agents, latent_dim, device=device) * 0.1
        
        # Pheromone trail: list of past positions
        self.pheromone_history = []
        
    def calculate_fitness(self, recon, objective="spread"):
        """
        Calculates fitness for the generated trajectories.
        recon shape: (num_agents, T, N, 6)
        Features: [x,y,z, vx,vy,vz]
        """
        # Node positions: (num_agents, T, N, 3)
        pos = recon[:, :, :, :3]
        # Node velocities: (num_agents, T, N, 3)
        vel = recon[:, :, :, 3:]
        
        if objective == "spread":
            # Maximize average pairwise distance between all nodes
            # Pairwise distances: shape (num_agents, T, N, N)
            diffs = pos.unsqueeze(3) - pos.unsqueeze(2)
            dists = torch.norm(diffs, dim=-1)
            # Average distance across time, pairs (excluding self-pairs)
            fitness = dists.mean(dim=(1, 2, 3))
            
        elif objective == "velocity":
            # Maximize average speed of particles
            speeds = torch.norm(vel, dim=-1)
            fitness = speeds.mean(dim=(1, 2))
            
        elif objective == "cohesion":
            # Minimize average pairwise distance (negative distance)
            diffs = pos.unsqueeze(3) - pos.unsqueeze(2)
            dists = torch.norm(diffs, dim=-1)
            fitness = -dists.mean(dim=(1, 2, 3))
            
        else:
            # Neutral / Novelty search (no goal force, pure stigmergy exploration)
            fitness = torch.zeros(recon.size(0), device=self.device)
            
        return fitness

    def step(
        self,
        objective="spread",
        w_inertia=0.7,
        c_stigmergy=0.3,
        c_mutual=0.3,
        c_goal=0.5,
        dt=0.1
    ):
        """
        Advances the swarm by one step in the latent space.
        """
        self.model.eval()
        
        # 1. Enable gradients on agent positions to compute goal forces
        pos_var = self.positions.clone().detach().requires_grad_(True)
        
        # 2. Decode trajectories from latent positions
        recon = self.model.decode(pos_var, self.edge_index, self.seq_len) # (num_agents, T, N, 6)
        
        # 3. Calculate fitness and backpropagate to positions to get goal gradients
        fitness = self.calculate_fitness(recon, objective)
        
        # Compute gradient for all agents in parallel using functional autograd
        goal_force = torch.zeros_like(self.positions)
        if objective != "novelty":
            # The derivative of the sum of independent fitnesses gives the individual gradients
            grad_all = torch.autograd.grad(outputs=fitness.sum(), inputs=pos_var)[0]
            # Normalize gradients per agent to avoid instability
            grad_norms = torch.norm(grad_all, dim=-1, keepdim=True) + 1e-6
            goal_force = grad_all / grad_norms
        
        # 4. Calculate Stigmergic Repulsion (from historical trail)
        stig_rep = torch.zeros_like(self.positions)
        if len(self.pheromone_history) > 0:
            # Convert history to tensor: (H, Z)
            history_tensor = torch.stack(self.pheromone_history)
            for i in range(self.num_agents):
                # Distance to all pheromone deposits
                diffs = self.positions[i] - history_tensor  # (H, Z)
                dists = torch.norm(diffs, dim=-1, keepdim=True) + 1e-3
                # Repulsive force inversely proportional to squared distance
                forces = diffs / (dists ** 3)  # Shape (H, Z)
                # Apply decay weighting (older pheromones have less influence)
                decay = torch.linspace(0.1, 1.0, steps=len(self.pheromone_history), device=self.device).view(-1, 1)
                weighted_forces = forces * decay
                stig_rep[i] = weighted_forces.sum(dim=0)
                
            # Normalize stigmergy force
            stig_rep_norm = torch.norm(stig_rep, dim=-1, keepdim=True) + 1e-6
            stig_rep = (stig_rep / stig_rep_norm) * torch.clamp(stig_rep_norm, 0, 1.0)
            
        # 5. Calculate Mutual Repulsion (between active agents)
        mutual_rep = torch.zeros_like(self.positions)
        for i in range(self.num_agents):
            diffs = self.positions[i] - self.positions
            dists = torch.norm(diffs, dim=-1, keepdim=True) + 1e-3
            forces = diffs / (dists ** 3)
            # Zero out force from self
            forces[i] = 0
            mutual_rep[i] = forces.sum(dim=0)
            
        # Normalize mutual repulsion
        mutual_rep_norm = torch.norm(mutual_rep, dim=-1, keepdim=True) + 1e-6
        mutual_rep = (mutual_rep / mutual_rep_norm) * torch.clamp(mutual_rep_norm, 0, 1.0)
        
        # 6. Record current positions in pheromone history (with length limit)
        for i in range(self.num_agents):
            self.pheromone_history.append(self.positions[i].clone().detach())
        if len(self.pheromone_history) > 500:
            self.pheromone_history = self.pheromone_history[-500:]
            
        # 7. Update Velocities and Positions
        new_velocities = (
            w_inertia * self.velocities +
            c_stigmergy * stig_rep +
            c_mutual * mutual_rep +
            c_goal * goal_force
        )
        
        # Keep velocities bounded
        vel_norm = torch.norm(new_velocities, dim=-1, keepdim=True) + 1e-6
        max_vel = 0.5
        new_velocities = torch.where(vel_norm > max_vel, (new_velocities / vel_norm) * max_vel, new_velocities)
        
        self.velocities = new_velocities
        self.positions = self.positions + self.velocities * dt
        
        # Boundary clipping to keep agents on the learned manifold
        self.positions = torch.clamp(self.positions, -4.0, 4.0)
        
        # Return decoded trajectories and fitnesses for rendering
        with torch.no_grad():
            decoded_trajectories = self.model.decode(self.positions, self.edge_index, self.seq_len)
            
        # Denormalize if stats are available
        if self.data_mean is not None and self.data_std is not None:
            decoded_trajectories = decoded_trajectories * self.data_std + self.data_mean
            
        return decoded_trajectories.cpu().numpy(), self.positions.cpu().numpy(), fitness.detach().cpu().numpy()
