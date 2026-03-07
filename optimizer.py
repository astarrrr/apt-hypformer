"""Dual optimizer wrapping Adam (Euclidean params) and RiemannianAdam (hyperbolic params).

Compatible with PIDSMaker's training loop which only calls zero_grad() and step().
"""

import torch
from geoopt import ManifoldParameter
from geoopt.optim.radam import RiemannianAdam


class DualOptimizer:
    """Optimizer that splits parameters into Euclidean and Riemannian groups.

    Euclidean parameters use standard Adam.
    ManifoldParameter instances use RiemannianAdam with periodic stabilization.
    """

    def __init__(self, parameters, lr=1e-4, hyp_lr=1e-4,
                 weight_decay=1e-5, hyp_weight_decay=0.0, clamp_params=None,
                 k_log_every=200):
        params = list(parameters)
        euc_params = [p for p in params if not isinstance(p, ManifoldParameter)]
        hyp_params = [p for p in params if isinstance(p, ManifoldParameter)]

        self.clamp_params = clamp_params or []  # list of (param, min, max)
        self.k_log_every = k_log_every
        self._step_count = 0
        self.optimizers = []
        if euc_params:
            self.optimizers.append(
                torch.optim.Adam(euc_params, lr=lr, weight_decay=weight_decay)
            )
        if hyp_params:
            self.optimizers.append(
                RiemannianAdam(hyp_params, lr=hyp_lr, stabilize=10, weight_decay=hyp_weight_decay)
            )

        # Fallback: if no params at all, create a dummy optimizer
        if not self.optimizers:
            self.optimizers.append(torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=lr))

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()
        for param, lo, hi in self.clamp_params:
            param.data.clamp_(lo, hi)
        self._step_count += 1
        if self.clamp_params and self.k_log_every > 0 and self._step_count % self.k_log_every == 0:
            k_val = self.clamp_params[0][0].item()
            print(f"[HypCurvature] step={self._step_count}  k={k_val:.4f}")
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({"hyp/k": k_val}, commit=False)
            except Exception:
                pass

    @property
    def param_groups(self):
        """Compatibility with PIDSMaker training loop param_groups access."""
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups
