"""Factory extension functions for integrating hyperbolic components into PIDSMaker.

Provides three factory functions:
    - create_hyp_encoder: builds HypTransformerEncoder
    - create_hyp_objective: builds HypEdgeReconstruction with decoder
    - create_dual_optimizer: builds DualOptimizer
"""

import sys
import os
import atexit

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

_LAST_HYP_ENCODER = None
_INITIAL_K = None
_REGISTERED_K_LOGGER = False


def _register_final_k_logger():
    global _REGISTERED_K_LOGGER
    if _REGISTERED_K_LOGGER:
        return

    def _print_final_k():
        if _LAST_HYP_ENCODER is None or not hasattr(_LAST_HYP_ENCODER, "manifold"):
            return
        k = float(_LAST_HYP_ENCODER.manifold.k.detach().cpu().item())
        if _INITIAL_K is None:
            print(f"[hyperbolic_transformer] Final learned k={k:.8f}")
        else:
            print(f"[hyperbolic_transformer] Final learned k={k:.8f} (delta={k - _INITIAL_K:+.8f})")

    atexit.register(_print_final_k)
    _REGISTERED_K_LOGGER = True


def create_hyp_encoder(cfg, in_dim):
    """Create HypTransformerEncoder from PIDSMaker config.

    Args:
        cfg: PIDSMaker configuration object
        in_dim: Input feature dimension
    Returns:
        HypTransformerEncoder instance
    """
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
        k=enc_cfg.k,
        attention_type=enc_cfg.attention_type,
        power_k=enc_cfg.power_k,
        use_bn=enc_cfg.use_bn,
        use_residual=enc_cfg.use_residual,
    )
    global _LAST_HYP_ENCODER
    global _INITIAL_K
    _LAST_HYP_ENCODER = encoder
    _INITIAL_K = float(encoder.manifold.k.detach().cpu().item())
    _register_final_k_logger()
    return encoder


def create_hyp_objective(cfg, node_out_dim):
    """Create HypEdgeReconstruction objective with its decoder.

    The decoder operates in tangent space; the objective computes
    Lorentz distance between predicted and target edge embeddings.

    Args:
        cfg: PIDSMaker configuration object
        node_out_dim: Encoder output spatial dimension (without time component)
    Returns:
        HypEdgeReconstruction instance
    """
    enc_cfg = cfg.training.encoder.hyperbolic_transformer
    if _LAST_HYP_ENCODER is not None and hasattr(_LAST_HYP_ENCODER, "manifold"):
        manifold = _LAST_HYP_ENCODER.manifold
    else:
        manifold = Lorentz(k=float(enc_cfg.k), learnable=True)

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

    Args:
        cfg: PIDSMaker configuration object
        parameters: Model parameters (set or list)
    Returns:
        DualOptimizer instance
    """
    lr = cfg.training.lr
    weight_decay = cfg.training.weight_decay

    # Use hyperbolic-specific learning rates if available
    hyp_lr = getattr(cfg.training, 'hyp_lr', lr)
    hyp_weight_decay = getattr(cfg.training, 'hyp_weight_decay', 0.0)
    curvature_params = []
    if _LAST_HYP_ENCODER is not None and hasattr(_LAST_HYP_ENCODER, "manifold"):
        curvature_params.append(_LAST_HYP_ENCODER.manifold.k)

    return DualOptimizer(
        parameters=parameters,
        lr=lr,
        hyp_lr=hyp_lr,
        weight_decay=weight_decay,
        hyp_weight_decay=hyp_weight_decay,
        curvature_params=curvature_params,
    )
