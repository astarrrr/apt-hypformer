"""Hyperbolic Transformer + GNN dual-branch encoder for provenance graph intrusion detection.

Adapted from HypFormer (hypformer.py, gnns.py) to work within the PIDSMaker pipeline.
The Transformer branch operates in Lorentz hyperbolic space for global attention,
while the GNN branch operates in Euclidean space for local structure.
Both branches are fused in hyperbolic space via weighted Lorentz midpoint.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree

from manifolds.lorentz import Lorentz
from manifolds.layers import HypLinear, HypLayerNorm, HypActivation, HypDropout


# ---------------------------------------------------------------------------
# Transformer branch (adapted from HypFormer TransConvLayer / TransConv)
# ---------------------------------------------------------------------------

class TransConvLayer(nn.Module):
    """Single hyperbolic Transformer attention layer with multi-head attention."""

    def __init__(self, manifold, in_channels, out_channels, num_heads,
                 use_weight=True, attention_type='linear_focused', power_k=2,
                 trans_heads_concat=False):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight
        self.attention_type = attention_type

        self.Wk = nn.ModuleList()
        self.Wq = nn.ModuleList()
        for i in range(self.num_heads):
            self.Wk.append(HypLinear(self.manifold, self.in_channels, self.out_channels))
            self.Wq.append(HypLinear(self.manifold, self.in_channels, self.out_channels))

        if use_weight:
            self.Wv = nn.ModuleList()
            for i in range(self.num_heads):
                self.Wv.append(HypLinear(self.manifold, in_channels, out_channels))

        self.scale = nn.Parameter(torch.tensor([math.sqrt(out_channels)]))
        self.bias = nn.Parameter(torch.zeros(()))
        self.norm_scale = nn.Parameter(torch.ones(()))
        self.v_map_mlp = nn.Linear(in_channels, out_channels, bias=True)
        self.power_k = power_k
        self.trans_heads_concat = trans_heads_concat

        if self.trans_heads_concat:
            self.final_linear = nn.Linear(out_channels * self.num_heads, out_channels, bias=True)

    def full_attention(self, qs, ks, vs):
        # negative squared distance (less than 0)
        att_weight = 2 + 2 * self.manifold.cinner(qs.transpose(0, 1), ks.transpose(0, 1))
        att_weight = att_weight / self.scale + self.bias

        att_weight = nn.Softmax(dim=-1)(att_weight)
        att_output = self.manifold.mid_point(vs.transpose(0, 1), att_weight)
        att_output = att_output.transpose(0, 1)

        att_output = self.manifold.mid_point(att_output)
        return att_output

    @staticmethod
    def fp(x, p=2):
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        norm_x_p = torch.norm(x ** p, p=2, dim=-1, keepdim=True)
        return (norm_x / (norm_x_p + 1e-8)) * x ** p

    def linear_focus_attention(self, hyp_qs, hyp_ks, hyp_vs):
        qs = hyp_qs[..., 1:]
        ks = hyp_ks[..., 1:]
        v = hyp_vs[..., 1:]
        phi_qs = (F.relu(qs) + 1e-6) / (self.norm_scale.abs() + 1e-6)
        phi_ks = (F.relu(ks) + 1e-6) / (self.norm_scale.abs() + 1e-6)

        phi_qs = self.fp(phi_qs, p=self.power_k)
        phi_ks = self.fp(phi_ks, p=self.power_k)

        k_transpose_v = torch.einsum('nhm,nhd->hmd', phi_ks, v)
        numerator = torch.einsum('nhm,hmd->nhd', phi_qs, k_transpose_v)

        denominator = torch.einsum('nhd,hd->nh', phi_qs, torch.einsum('nhd->hd', phi_ks))
        denominator = denominator.unsqueeze(-1)

        attn_output = numerator / (denominator + 1e-6)

        vss = self.v_map_mlp(v)
        attn_output = attn_output + vss

        if self.trans_heads_concat:
            attn_output = self.final_linear(attn_output.reshape(-1, self.num_heads * self.out_channels))
        else:
            attn_output = attn_output.mean(dim=1)

        attn_output_time = ((attn_output ** 2).sum(dim=-1, keepdims=True) + self.manifold.k) ** 0.5
        attn_output = torch.cat([attn_output_time, attn_output], dim=-1)

        return attn_output

    def forward(self, query_input, source_input):
        q_list = []
        k_list = []
        v_list = []
        for i in range(self.num_heads):
            q_list.append(self.Wq[i](query_input))
            k_list.append(self.Wk[i](source_input))
            if self.use_weight:
                v_list.append(self.Wv[i](source_input))
            else:
                v_list.append(source_input)

        query = torch.stack(q_list, dim=1)   # [N, H, D+1]
        key = torch.stack(k_list, dim=1)     # [N, H, D+1]
        value = torch.stack(v_list, dim=1)   # [N, H, D+1]

        if self.attention_type == 'linear_focused':
            final_output = self.linear_focus_attention(query, key, value)
        elif self.attention_type == 'full':
            final_output = self.full_attention(query, key, value)
        else:
            raise NotImplementedError(f"Unknown attention type: {self.attention_type}")

        return final_output


class TransConv(nn.Module):
    """Multi-layer hyperbolic Transformer with batch norm and residual connections."""

    def __init__(self, manifold, in_channels, hidden_channels, num_layers=2,
                 num_heads=1, dropout=0.5, use_bn=True, use_residual=True,
                 use_weight=True, use_act=True, attention_type='linear_focused',
                 power_k=2):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.use_bn = use_bn
        self.residual = use_residual
        self.use_act = use_act
        self.use_weight = use_weight

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input projection: Euclidean -> Lorentz
        self.fcs.append(HypLinear(self.manifold, self.in_channels, self.hidden_channels))
        self.bns.append(HypLayerNorm(self.manifold, self.hidden_channels))

        for i in range(self.num_layers):
            self.convs.append(
                TransConvLayer(self.manifold, self.hidden_channels, self.hidden_channels,
                               num_heads=self.num_heads, use_weight=self.use_weight,
                               attention_type=attention_type, power_k=power_k))
            self.bns.append(HypLayerNorm(self.manifold, self.hidden_channels))

        self.dropout = HypDropout(self.manifold, self.dropout_rate)
        self.activation = HypActivation(self.manifold, activation=F.relu)

        # Output projection (stays in Lorentz)
        self.fcs.append(HypLinear(self.manifold, self.hidden_channels, self.hidden_channels))

    def forward(self, x_input):
        """
        Args:
            x_input: Euclidean features [N, in_channels]
        Returns:
            Lorentz embeddings [N, hidden_channels+1]
        """
        layer_ = []

        # Project from Euclidean to Lorentz
        x = self.fcs[0](x_input, x_manifold='euc')

        if self.use_bn:
            x = self.bns[0](x)
        if self.use_act:
            x = self.activation(x)
        x = self.dropout(x, training=self.training)
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, x)
            if self.residual:
                x = self.manifold.mid_point(torch.stack((x, layer_[i]), dim=1))
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)

        x = self.fcs[-1](x)
        return x


# ---------------------------------------------------------------------------
# GNN branch (adapted from HypFormer GraphConvLayer / GraphConv)
# ---------------------------------------------------------------------------

class GraphConvLayer(nn.Module):
    """Single GNN layer with normalized adjacency message passing."""

    def __init__(self, in_channels, out_channels, use_weight=True, use_init=False):
        super(GraphConvLayer, self).__init__()
        self.use_init = use_init
        self.use_weight = use_weight
        if self.use_init:
            in_channels_ = 2 * in_channels
        else:
            in_channels_ = in_channels
        self.W = nn.Linear(in_channels_, out_channels)

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x, edge_index, x0):
        N = x.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm_in = (1. / d[col]).sqrt()
        d_norm_out = (1. / d[row]).sqrt()
        value = torch.ones_like(row, dtype=torch.float) * d_norm_in * d_norm_out
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

        # Sparse message passing
        # Build sparse adjacency and perform matmul
        adj = torch.sparse_coo_tensor(
            torch.stack([col, row]),
            value,
            (N, N)
        ).coalesce()
        x = torch.sparse.mm(adj, x)

        if self.use_init:
            x = torch.cat([x, x0], 1)
            x = self.W(x)
        elif self.use_weight:
            x = self.W(x)

        return x


class GraphConv(nn.Module):
    """Multi-layer Euclidean GNN with batch norm and residual connections."""

    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.5,
                 use_bn=True, use_residual=True, use_weight=True, use_init=False,
                 use_act=True):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers):
            self.convs.append(
                GraphConvLayer(hidden_channels, hidden_channels, use_weight, use_init))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index):
        """
        Args:
            x: Euclidean features [N, in_channels]
            edge_index: [2, E] edge index
        Returns:
            Euclidean embeddings [N, hidden_channels]
        """
        layer_ = []

        x = self.fcs[0](x)
        if self.use_bn:
            if x.shape[0] > 1:
                x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, layer_[0])
            if self.use_bn:
                if x.shape[0] > 1:
                    x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x + layer_[-1]
        return x


# ---------------------------------------------------------------------------
# Combined dual-branch encoder
# ---------------------------------------------------------------------------

class HypTransformerEncoder(nn.Module):
    """Dual-branch encoder: Hyperbolic Transformer (global) + GNN (local).

    The Transformer branch produces Lorentz embeddings via hyperbolic attention.
    The GNN branch produces Euclidean embeddings which are mapped to Lorentz space.
    Both are fused via weighted Lorentz midpoint.

    Output: {"h": lorentz_embeddings} with shape [N, out_dim+1]
    """

    def __init__(self, in_dim, hid_dim, out_dim,
                 trans_num_layers=2, trans_num_heads=4, trans_dropout=0.3,
                 gnn_num_layers=2, gnn_dropout=0.3, graph_weight=0.5,
                 k=1.0, attention_type='linear_focused', power_k=2,
                 use_bn=True, use_residual=True):
        super().__init__()
        self.manifold = Lorentz(k=float(k))
        self.graph_weight = graph_weight
        self.use_graph = graph_weight > 0.0

        # Transformer branch: Euclidean input -> Lorentz output [N, hid_dim+1]
        self.trans_conv = TransConv(
            manifold=self.manifold,
            in_channels=in_dim,
            hidden_channels=hid_dim,
            num_layers=trans_num_layers,
            num_heads=trans_num_heads,
            dropout=trans_dropout,
            use_bn=use_bn,
            use_residual=use_residual,
            use_weight=True,
            use_act=True,
            attention_type=attention_type,
            power_k=power_k,
        )

        # GNN branch: Euclidean input -> Euclidean output [N, hid_dim]
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
            # Map GNN Euclidean output to Lorentz space
            self.decode_graph = HypLinear(self.manifold, hid_dim, hid_dim)

        # Output projection: hid_dim -> out_dim in Lorentz space
        self.output_proj = HypLinear(self.manifold, hid_dim, out_dim)

    def forward(self, x, edge_index, **kwargs):
        """
        Args:
            x: Node features [N, in_dim] (Euclidean)
            edge_index: [2, E] edge connectivity
        Returns:
            dict: {"h": Lorentz embeddings [N, out_dim+1]}
        """
        # Transformer branch
        x1 = self.trans_conv(x)  # [N, hid_dim+1] Lorentz

        if self.use_graph and edge_index.shape[1] > 0:
            # GNN branch
            x2 = self.graph_conv(x, edge_index)  # [N, hid_dim] Euclidean
            # Map to Lorentz
            z_graph_hyp = self.decode_graph(x2, x_manifold='euc')  # [N, hid_dim+1]
            # Fuse via weighted Lorentz midpoint
            z_hyp = torch.stack(
                [(1 - self.graph_weight) * x1, self.graph_weight * z_graph_hyp],
                dim=1
            )
            z = self.manifold.mid_point(z_hyp)  # [N, hid_dim+1]
        else:
            z = x1

        # Output projection
        h = self.output_proj(z)  # [N, out_dim+1]

        return {"h": h}
