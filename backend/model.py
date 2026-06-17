import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

def batch_edge_index(edge_index, batch_size, num_nodes):
    """
    Batches a static edge_index of shape (2, E) for a given batch_size.
    Returns a batched edge_index of shape (2, batch_size * E) where each
    graph is offset by i * num_nodes.
    """
    E = edge_index.size(1)
    offsets = torch.arange(batch_size, device=edge_index.device).view(-1, 1, 1) * num_nodes
    batched_edges = edge_index.unsqueeze(0).repeat(batch_size, 1, 1) + offsets
    return batched_edges.permute(1, 0, 2).reshape(2, -1)

class STGVAE(nn.Module):
    def __init__(self, num_nodes=5, in_features=6, spatial_dim=32, temporal_dim=64, latent_dim=8):
        super(STGVAE, self).__init__()
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.latent_dim = latent_dim
        
        # --- Encoder ---
        # Spatial encoder: GAT layer
        self.enc_gat = GATConv(in_channels=in_features, out_channels=spatial_dim, heads=2, concat=True)
        gat_out_dim = spatial_dim * 2  # due to heads=2 and concat=True
        
        # Temporal encoder: GRU
        self.enc_gru = nn.GRU(input_size=gat_out_dim, hidden_size=temporal_dim, batch_first=True)
        
        # Latent projection
        self.fc_mu = nn.Linear(temporal_dim * num_nodes, latent_dim)
        self.fc_logvar = nn.Linear(temporal_dim * num_nodes, latent_dim)
        
        # --- Decoder ---
        # Learnable node embeddings to prevent spatial over-smoothing in the decoder
        # Each of the N nodes gets a unique D=16 identity embedding
        self.node_emb = nn.Parameter(torch.randn(num_nodes, 16) * 0.1)
        
        # Recurrent temporal decoder: GRU
        dec_in_dim = 16 + 1  # node_emb + normalized time
        self.fc_dec_init = nn.Linear(latent_dim, num_nodes * temporal_dim)
        self.dec_gru = nn.GRU(input_size=dec_in_dim, hidden_size=temporal_dim, batch_first=True)
        self.dec_fc = nn.Linear(temporal_dim, in_features)
        
    def encode(self, x, edge_index):
        """
        x: (B, T, N, F)
        edge_index: (2, E)
        """
        B, T, N, num_feats = x.shape
        
        # 1. Spatial encoding: run GAT Conv on all graphs in parallel
        B_eff = B * T
        x_flat = x.reshape(B_eff * N, num_feats)
        
        # Create batched edge_index for B_eff graphs
        batched_edges = batch_edge_index(edge_index, B_eff, N)
        
        # Apply GAT
        h_spatial = self.enc_gat(x_flat, batched_edges) # Shape: (B_eff * N, gat_out_dim)
        h_spatial = F.relu(h_spatial)
        
        # Reshape back to sequence form: (B * N, T, gat_out_dim)
        h_seq = h_spatial.reshape(B, T, N, -1).permute(0, 2, 1, 3).reshape(B * N, T, -1)
        
        # 2. Temporal encoding: run GRU over sequence
        out_gru, h_n = self.enc_gru(h_seq) # h_n shape: (1, B * N, temporal_dim)
        
        # Extract last hidden state and reshape to (B, N * temporal_dim)
        h_n = h_n.squeeze(0).reshape(B, N, -1).reshape(B, -1)
        
        # 3. Latent bottleneck projection
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        
        return mu, logvar
        
    def reparameterize(self, mu, logvar, noise_scale=1.0):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std * noise_scale
        
    def decode(self, z, edge_index, T):
        """
        z: (B, Z)
        edge_index: (2, E)
        T: length of sequence to decode
        Recurrent GRU decoder initialized with projected latent state z
        """
        B, N = z.size(0), self.num_nodes
        
        # Project z to GRU decoder initial hidden state: (1, B * N, temporal_dim)
        h_0 = self.fc_dec_init(z)  # (B, N * temporal_dim)
        h_0 = h_0.reshape(B, N, -1).reshape(B * N, -1).unsqueeze(0)  # (1, B * N, temporal_dim)
        
        # Normalized time indices: (1, T, 1, 1)
        time = torch.linspace(0, 1, T, device=z.device).view(1, T, 1, 1)
        
        # Expand to full grid: (B, T, N, 1)
        time_exp = time.expand(B, -1, N, -1)
        
        # Expand node embeddings: (1, 1, N, 16) -> (B, T, N, 16)
        nemb_exp = self.node_emb.view(1, 1, N, 16).expand(B, T, N, -1)
        
        # Concatenate inputs: (B, T, N, 16 + 1)
        dec_input = torch.cat([time_exp, nemb_exp], dim=-1)
        
        # Reshape to (B * N, T, 16 + 1) for GRU
        dec_input_flat = dec_input.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        
        # Run GRU decoder over T steps starting from h_0
        out_gru, _ = self.dec_gru(dec_input_flat, h_0)  # (B * N, T, temporal_dim)
        
        # Project output to feature space
        recon_flat = self.dec_fc(out_gru)  # (B * N, T, F)
        
        # Reshape back to (B, T, N, F)
        recon = recon_flat.reshape(B, N, T, -1).permute(0, 2, 1, 3)
        
        return recon
        
    def forward(self, x, edge_index, noise_scale=1.0):
        T = x.size(1)
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar, noise_scale=noise_scale)
        recon = self.decode(z, edge_index, T)
        return recon, mu, logvar
