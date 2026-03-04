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
                 weight_decay=1e-5, hyp_weight_decay=0.0):
        params = list(parameters)
        euc_params = [p for p in params if not isinstance(p, ManifoldParameter)]
        hyp_params = [p for p in params if isinstance(p, ManifoldParameter)]

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

    @property
    def param_groups(self):
        """Compatibility with PIDSMaker training loop param_groups access."""
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups
