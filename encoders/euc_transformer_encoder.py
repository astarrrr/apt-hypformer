"""Euclidean Transformer + GNN dual-branch encoder for baseline comparison.

This mirrors HypTransformerEncoder's high-level topology but keeps all
transformer computations in Euclidean space.
"""

import torch
import torch.nn as nn

from encoders.hyp_transformer_encoder import GraphConv


class EuclideanTransformerBlock(nn.Module):
    """Standard pre-norm Transformer block."""

    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Nodes are treated as a token sequence (single graph at a time).
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(attn_out)

        h = self.norm2(x)
        x = x + self.dropout(self.ffn(h))
        return x


class EuclideanTransformerEncoder(nn.Module):
    """Dual-branch baseline: Euclidean Transformer (global) + Euclidean GNN (local)."""

    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        trans_num_layers=2,
        trans_num_heads=4,
        trans_dropout=0.3,
        gnn_num_layers=2,
        gnn_dropout=0.3,
        graph_weight=0.5,
        use_bn=True,
        use_residual=True,
    ):
        super().__init__()
        self.graph_weight = graph_weight
        self.use_graph = graph_weight > 0.0

        self.input_proj = nn.Linear(in_dim, hid_dim)
        self.blocks = nn.ModuleList(
            [
                EuclideanTransformerBlock(
                    dim=hid_dim,
                    num_heads=trans_num_heads,
                    dropout=trans_dropout,
                )
                for _ in range(trans_num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(hid_dim)

        if self.use_graph:
            self.graph_conv = GraphConv(
                in_channels=in_dim,
                hidden_channels=hid_dim,
                num_layers=gnn_num_layers,
                dropout=gnn_dropout,
                use_bn=use_bn,
                use_residual=use_residual,
                use_weight=True,
                use_init=False,
                use_act=True,
            )

        self.output_proj = nn.Linear(hid_dim, out_dim)

    def forward(self, x, edge_index, **kwargs):
        # Transformer branch: [N, in_dim] -> [N, hid_dim]
        z = self.input_proj(x).unsqueeze(0)
        for block in self.blocks:
            z = block(z)
        x1 = self.out_norm(z.squeeze(0))

        if self.use_graph and edge_index.shape[1] > 0:
            x2 = self.graph_conv(x, edge_index)  # [N, hid_dim]
            z = (1 - self.graph_weight) * x1 + self.graph_weight * x2
        else:
            z = x1

        h_space = self.output_proj(z)  # [N, out_dim]

        # Keep output shape aligned with hyperbolic path: [N, out_dim+1].
        h_time = torch.zeros(h_space.shape[0], 1, dtype=h_space.dtype, device=h_space.device)
        h = torch.cat([h_time, h_space], dim=-1)
        return {"h": h}
