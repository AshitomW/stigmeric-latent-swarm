# Stigmergic Latent Swarms

A generative physics surrogate and latent path-planning framework. This repository implements a **Spatio-Temporal Graph Variational Autoencoder (ST-GVAE)** that compresses complex multi-agent trajectories into a continuous, structured, low-dimensional latent space ($d_{\text{latent}}$, configured as 256 for this implementation). Once trained, a differentiable decoder acts as a fast surrogate model, allowing a gradient-guided **Particle Swarm Optimization (PSO)** loop to navigate the latent manifold and generate coordinated paths in real time.

---

## Key Features

- **Trajectory Compression:** Encodes multi-agent trajectories (time, position, velocity, and connectivity) into a compact latent manifold.
- **Differentiable Surrogate:** Decodes latent coordinates back into 3D particle state sequences. The decoder is fully differentiable, permitting analytical gradient calculations through PyTorch's autograd.
- **Gradient-Guided Exploration:** Features a Particle Swarm Optimizer (PSO) that updates search coordinates using a balance of mutual agent repulsion, pheromone decay (stigmergy), and gradient-based goal forces.
- **Interactive 3D Web Visualizer:** Includes a real-time web dashboard using **Three.js** (WebGL) for 3D trajectory rendering and **Chart.js** for projecting and tracking the swarm exploration inside the latent manifold.

---

## Directory Structure

```text
├── backend/                  # FastAPI web server and PyTorch modeling code
│   ├── check_dists.py        # Physics distance and safety checks
│   ├── eval_metrics.py       # Metrics evaluator and plot generator
│   ├── models/               # ST-GVAE neural network architecture (GAT, GRU)
│   ├── server.py             # FastAPI entry point & API endpoints
│   └── train.py              # Model training script
├── docs/                     # LaTeX source code of the paper
├── frontend/                 # Web client dashboard (HTML, CSS, JS/Three.js)
├── start.sh                  # One-click environment installer and startup script
└── train.sh                  # One-click model training execution script
```

---

## Setup & Quickstart

### Prerequisites

- Python 3.12
- [uv](https://github.com/astral-sh/uv) (recommended Python package installer)

### Running the Visualizer

To automatically set up a virtual environment, install the dependencies, generate the synthetic spring-mass dataset, and launch the web dashboard:

```bash
chmod +x start.sh
./start.sh
```

This script will launch the FastAPI backend on `http://localhost:8000` and open the visualizer in your default browser.

### Re-training the Model

If you wish to re-train the ST-GVAE model on the simulation data:

```bash
chmod +x train.sh
./train.sh
```

---
