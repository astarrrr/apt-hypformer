"""Hyperbolic edge decoder operating in tangent space.

Takes Lorentz source and destination embeddings, maps to tangent space via logmap0,
concatenates, runs through an MLP, and maps back to Lorentz space via expmap0.
"""

import torch
import torch.nn as nn


class HypEdgeDecoder(nn.Module):
    """Decode edge representations from source and destination Lorentz embeddings.

    Pipeline:
        1. logmap0: Lorentz -> tangent space at origin
        2. Take spatial components [..., 1:]
        3. Concatenate src + dst -> [E, 2*in_dim]
        4. MLP decode -> [E, out_dim]
        5. Prepend zero -> expmap0 -> Lorentz point [E, out_dim+1]
    """

    def __init__(self, manifold, in_dim, hid_dim, out_dim, num_layers=2, dropout=0.1):
        """
        Args:
            manifold: Lorentz manifold instance
            in_dim: Spatial dimension of input (without time component)
            hid_dim: Hidden dimension for MLP
            out_dim: Spatial dimension of output (without time component)
            num_layers: Number of MLP layers
            dropout: Dropout rate
        """
        super().__init__()
        self.manifold = manifold
        self.in_dim = in_dim
        self.out_dim = out_dim

        layers = []
        current_dim = 2 * in_dim  # concatenated src + dst

        for i in range(num_layers):
            next_dim = hid_dim if i < num_layers - 1 else out_dim
            layers.append(nn.Linear(current_dim, next_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            current_dim = next_dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, h_src, h_dst):
        """
        Args:
            h_src: Source Lorentz embeddings [E, dim+1]
            h_dst: Destination Lorentz embeddings [E, dim+1]
        Returns:
            h_edge_hat: Predicted edge Lorentz embeddings [E, out_dim+1]
        """
        # Map to tangent space at origin, take spatial components
        t_src = self.manifold.logmap0(h_src)[..., 1:]  # [E, in_dim]
        t_dst = self.manifold.logmap0(h_dst)[..., 1:]  # [E, in_dim]

        # Concatenate and decode
        t_cat = torch.cat([t_src, t_dst], dim=-1)  # [E, 2*in_dim]
        t_out = self.mlp(t_cat)  # [E, out_dim]

        # Map back to Lorentz: prepend zero time component then expmap0
        zeros = torch.zeros_like(t_out[..., :1])
        t_full = torch.cat([zeros, t_out], dim=-1)  # [E, out_dim+1]
        h_edge_hat = self.manifold.expmap0(t_full)  # [E, out_dim+1]

        return h_edge_hat
