"""Hyperbolic loss functions for Lorentz space operations."""

import torch


def lorentz_distance_loss(h_pred, h_target, manifold, inference=False):
    """Squared Lorentz distance loss.

    Args:
        h_pred: Predicted Lorentz points [*, dim+1]
        h_target: Target Lorentz points [*, dim+1]
        manifold: Lorentz manifold instance
        inference: If True, return per-sample losses
    Returns:
        Scalar mean loss (training) or per-sample losses (inference)
    """
    sq_dist = manifold.square_dist(h_pred, h_target).squeeze(-1)
    return sq_dist if inference else sq_dist.mean()


def tangent_mse_loss(h_pred, h_target, manifold, inference=False):
    """MSE loss in tangent space at origin.

    Maps both points to the tangent space at origin via logmap0,
    then computes MSE on spatial components.

    Args:
        h_pred: Predicted Lorentz points [*, dim+1]
        h_target: Target Lorentz points [*, dim+1]
        manifold: Lorentz manifold instance
        inference: If True, return per-sample losses
    Returns:
        Scalar mean loss (training) or per-sample losses (inference)
    """
    pred_t = manifold.logmap0(h_pred)[..., 1:]
    target_t = manifold.logmap0(h_target)[..., 1:]
    per_sample = ((pred_t - target_t) ** 2).sum(dim=-1)
    return per_sample if inference else per_sample.mean()
