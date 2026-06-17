import os
import numpy as np

def generate_spring_system_data(
    num_particles=5,
    num_steps=1000,
    num_simulations=200,
    dt=0.05,
    k_spring=1.0,
    rest_len=1.5,
    damping=0.1,
    center_attraction=0.2,
    noise_level=0.05,
    seed=42
):
    """
    Simulates a spring-mass network of particles in 3D space.
    Each particle is connected to all other particles (fully connected graph).
    The system is subjected to:
    - Spring forces between connected particles
    - Gravity/Central attraction force (pulling to origin to keep systems bound)
    - Damping (velocity resistance)
    - Small random perturbations (driving force)
    """
    np.random.seed(seed)
    
    # Store trajectories: (simulations, steps, particles, features)
    # Features: [x, y, z, vx, vy, vz]
    all_trajectories = []
    
    # Fully connected graph adjacency list
    edge_index = []
    for i in range(num_particles):
        for j in range(num_particles):
            if i != j:
                edge_index.append([i, j])
    edge_index = np.array(edge_index, dtype=np.int64).T  # Shape: (2, N * (N - 1))
    
    for sim in range(num_simulations):
        # Initial positions randomly distributed around origin
        pos = np.random.randn(num_particles, 3) * 2.0
        # Initial velocities
        vel = np.random.randn(num_particles, 3) * 0.5
        
        sim_data = []
        
        for step in range(num_steps):
            forces = np.zeros_like(pos)
            
            # 1. Spring forces (fully connected)
            for i in range(num_particles):
                for j in range(num_particles):
                    if i == j:
                        continue
                    diff = pos[j] - pos[i]
                    dist = np.linalg.norm(diff) + 1e-6
                    dir_vec = diff / dist
                    # Hooke's Law: F = k * (dist - rest_len) * dir
                    forces[i] += k_spring * (dist - rest_len) * dir_vec
            
            # 2. Central attraction (keeps the system bound)
            forces -= center_attraction * pos
            
            # 3. Damping force
            forces -= damping * vel
            
            # 4. Small random noise (driving force)
            forces += np.random.randn(num_particles, 3) * noise_level
            
            # Update positions and velocities (Euler-Maruyama integration)
            vel += forces * dt
            pos += vel * dt
            
            # Save state: features = [x, y, z, vx, vy, vz]
            state = np.hstack([pos, vel])
            sim_data.append(state)
            
        all_trajectories.append(sim_data)
        
    all_trajectories = np.array(all_trajectories) # (num_simulations, num_steps, num_particles, 6)
    
    return all_trajectories, edge_index

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic spring-mass system trajectories.")
    parser.add_argument('--num_simulations', type=int, default=200, help="Number of independent simulation runs")
    parser.add_argument('--num_steps', type=int, default=1000, help="Number of frames per simulation")
    parser.add_argument('--num_particles', type=int, default=5, help="Number of particles in the system")
    args = parser.parse_args()
    
    print(f"Generating synthetic spring-mass system data ({args.num_simulations} simulations, {args.num_steps} steps)...")
    
    data, edge_index = generate_spring_system_data(
        num_particles=args.num_particles,
        num_steps=args.num_steps,
        num_simulations=args.num_simulations
    )
    
    os.makedirs("backend/data", exist_ok=True)
    np.savez(
        "backend/data/spring_data.npz",
        data=data,
        edge_index=edge_index
    )
    
    print(f"Data saved successfully to backend/data/spring_data.npz")
    print(f"Data shape: {data.shape}")
    print(f"Edge index shape: {edge_index.shape}")
