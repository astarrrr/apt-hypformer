"""Factory extension functions for integrating hyperbolic components into PIDSMaker.

Provides three factory functions:
    - create_hyp_encoder: builds HypTransformerEncoder
    - create_hyp_objective: builds HypEdgeReconstruction with decoder
    - create_dual_optimizer: builds DualOptimizer

Shared manifold registry ensures encoder and decoder/objective operate on the
same Lorentz instance so that a learnable k is jointly optimized.
"""

import sys
import os
import torch.nn as nn

# Ensure myproject is FIRST on the path (PIDSMaker/pidsmaker/ also has
# encoders/, decoders/, etc. and sits at sys.path[0] when run via
# `python pidsmaker/main.py`, so we must prepend to avoid name clashes)
_myproject_dir = os.path.dirname(os.path.abspath(__file__))
if sys.path[0] != _myproject_dir:
    sys.path.insert(0, _myproject_dir)

from manifolds.lorentz import Lorentz
from encoders.hyp_transformer_encoder import HypTransformerEncoder
from decoders.hyp_edge_decoder import HypEdgeDecoder
from objectives.hyp_edge_reconstruction import HypEdgeReconstruction
from optimizer import DualOptimizer

# Module-level registry so encoder_factory and objective_factory
# (called separately by PIDSMaker's build_model) share the same Lorentz instance.
_shared_manifold = None


def create_shared_manifold(cfg):
    """Create (or recreate) the shared Lorentz manifold and store it in the registry.

    Args:
        cfg: PIDSMaker configuration object
    Returns:
        Lorentz manifold instance (learnable k when learnable_k=true in config)
    """
    global _shared_manifold
    enc_cfg = cfg.training.encoder.hyperbolic_transformer
    k = float(enc_cfg.k)
    learnable_k = bool(getattr(enc_cfg, 'learnable_k', False))
    _shared_manifold = Lorentz(k=k, learnable=learnable_k)
    return _shared_manifold


def create_hyp_encoder(cfg, in_dim, manifold=None):
    """Create HypTransformerEncoder from PIDSMaker config.

    Creates a shared Lorentz manifold (stored in module registry) unless one
    is explicitly provided.  The shared manifold is later retrieved by
    create_hyp_objective so both components use the same k.

    Args:
        cfg: PIDSMaker configuration object
        in_dim: Input feature dimension
        manifold: Optional pre-built Lorentz instance (default: create new shared one)
    Returns:
        HypTransformerEncoder instance
    """
    if manifold is None:
        manifold = create_shared_manifold(cfg)

    hid_dim = cfg.training.node_hid_dim
    out_dim = cfg.training.node_out_dim
    enc_cfg = cfg.training.encoder.hyperbolic_transformer

    encoder = HypTransformerEncoder(
        in_dim=in_dim,
        hid_dim=hid_dim,
        out_dim=out_dim,
        trans_num_layers=enc_cfg.trans_num_layers,
        trans_num_heads=enc_cfg.trans_num_heads,
        trans_dropout=enc_cfg.trans_dropout,
        gnn_num_layers=enc_cfg.gnn_num_layers,
        gnn_dropout=enc_cfg.gnn_dropout,
        graph_weight=enc_cfg.graph_weight,
        k=float(enc_cfg.k),
        attention_type=enc_cfg.attention_type,
        power_k=enc_cfg.power_k,
        use_bn=enc_cfg.use_bn,
        use_residual=enc_cfg.use_residual,
        manifold=manifold,
    )
    return encoder


def create_hyp_objective(cfg, node_out_dim, manifold=None):
    """Create HypEdgeReconstruction objective with its decoder.

    Reuses the shared Lorentz manifold created by create_hyp_encoder so the
    decoder's logmap0/expmap0 and the loss's square_dist all use the same k.

    Args:
        cfg: PIDSMaker configuration object
        node_out_dim: Encoder output spatial dimension (without time component)
        manifold: Optional Lorentz instance; defaults to shared registry manifold
    Returns:
        HypEdgeReconstruction instance
    """
    global _shared_manifold
    if manifold is None:
        manifold = _shared_manifold
    if manifold is None:
        # Fallback: create standalone manifold (encoder was not built via this module)
        enc_cfg = cfg.training.encoder.hyperbolic_transformer
        manifold = Lorentz(k=float(enc_cfg.k))

    decoder = HypEdgeDecoder(
        manifold=manifold,
        in_dim=node_out_dim,
        hid_dim=node_out_dim * 2,
        out_dim=node_out_dim,
        num_layers=2,
        dropout=cfg.training.encoder.dropout,
    )

    objective = HypEdgeReconstruction(
        decoder=decoder,
        manifold=manifold,
        loss_type='lorentz_dist',
    )

    return objective


def create_dual_optimizer(cfg, parameters):
    """Create DualOptimizer splitting Euclidean and Riemannian parameters.

    When learnable_k is enabled, registers a post-step clamp so k stays in
    [0.1, 10.0] and never goes negative (which would break sqrt(k) calls).

    Args:
        cfg: PIDSMaker configuration object
        parameters: Model parameters (set or list)
    Returns:
        DualOptimizer instance
    """
    global _shared_manifold
    lr = cfg.training.lr
    weight_decay = cfg.training.weight_decay
    hyp_lr = getattr(cfg.training, 'hyp_lr', lr)
    hyp_weight_decay = getattr(cfg.training, 'hyp_weight_decay', 0.0)

    clamp_params = []
    if _shared_manifold is not None:
        k_param = _shared_manifold.k
        if isinstance(k_param, nn.Parameter):
            clamp_params = [(k_param, 0.1, 10.0)]

    enc_cfg = cfg.training.encoder.hyperbolic_transformer
    k_log_every = int(getattr(enc_cfg, 'k_log_every', 200))

    return DualOptimizer(
        parameters=parameters,
        lr=lr,
        hyp_lr=hyp_lr,
        weight_decay=weight_decay,
        hyp_weight_decay=hyp_weight_decay,
        clamp_params=clamp_params,
        k_log_every=k_log_every,
    )
