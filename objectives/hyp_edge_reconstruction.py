"""Hyperbolic edge reconstruction objective.

Compatible with PIDSMaker's objective interface:
    forward(h_src, h_dst, inference, **kwargs) -> {"loss": scalar_or_per_edge}

Uses Lorentz midpoint as reconstruction target and Lorentz distance as loss.
"""

import torch
import torch.nn as nn


class HypEdgeReconstruction(nn.Module):
    """Reconstruct edge embeddings in hyperbolic space.

    Training: learns to predict the Lorentz midpoint of (src, dst) pairs.
    Inference: per-edge reconstruction error serves as anomaly score.
    """

    def __init__(self, decoder, manifold, loss_type='lorentz_dist'):
        """
        Args:
            decoder: HypEdgeDecoder instance
            manifold: Lorentz manifold instance
            loss_type: 'lorentz_dist' or 'tangent_mse'
        """
        super().__init__()
        self.decoder = decoder
        self.manifold = manifold
        self.loss_type = loss_type

    def forward(self, h_src, h_dst, inference=False, **kwargs):
        """
        Args:
            h_src: Source Lorentz embeddings [E, dim+1]
            h_dst: Destination Lorentz embeddings [E, dim+1]
            inference: If True, return per-edge scores instead of scalar loss
        Returns:
            dict: {"loss": scalar (training) or [E,] tensor (inference)}
        """
        # Compute reconstruction target: Lorentz midpoint of src and dst
        h_pair = torch.stack([h_src, h_dst], dim=1)  # [E, 2, dim+1]
        h_edge_target = self.manifold.mid_point(h_pair)  # [E, dim+1]

        # Decoder prediction
        h_edge_hat = self.decoder(h_src, h_dst)  # [E, dim+1]

        # Compute loss
        if self.loss_type == 'lorentz_dist':
            sq_dist = self.manifold.square_dist(h_edge_hat, h_edge_target).squeeze(-1)  # [E,]
        elif self.loss_type == 'tangent_mse':
            pred_t = self.manifold.logmap0(h_edge_hat)[..., 1:]
            target_t = self.manifold.logmap0(h_edge_target)[..., 1:]
            sq_dist = ((pred_t - target_t) ** 2).sum(dim=-1)  # [E,]
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if inference:
            return {"loss": sq_dist}  # per-edge anomaly scores
        else:
            return {"loss": sq_dist.mean()}  # scalar training loss
