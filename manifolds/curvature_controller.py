import random
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class CurvatureRLConfig:
    enabled: bool = False
    k_min: float = 0.2
    k_max: float = 10.0
    num_state_bins: int = 16
    ace_action_deltas: tuple = (-0.15, 0.0, 0.15)
    alpha: float = 0.1
    beta: float = 0.9
    epsilon: float = 0.2
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.999
    gamma_mix: float = 0.3
    log_every: int = 100
    verbose: bool = True


class CurvatureRLController:
    """Two-agent Q-learning controller for adaptive Lorentz curvature k.

    Agent-1 (ACE): proposes candidate k.
    Agent-2 (HGNN): accept/reject candidate k.
    Reward uses performance delta; in practice we use loss improvement.
    """

    def __init__(self, manifold, cfg: CurvatureRLConfig):
        self.manifold = manifold
        self.cfg = cfg

        n_states = max(2, int(cfg.num_state_bins))
        n_ace_actions = len(cfg.ace_action_deltas)
        self.q_ace = torch.zeros(n_states, n_ace_actions, dtype=torch.float32)
        self.q_hgnn = torch.zeros(n_states, 2, dtype=torch.float32)  # 0 reject, 1 accept

        self.prev_metric: Optional[float] = None
        self.pending_transition = None
        self.step_count = 0

    def _current_k(self) -> float:
        k_tensor = self.manifold.k.detach()
        return float(k_tensor.item())

    def _set_k(self, new_k: float):
        new_k = float(max(self.cfg.k_min, min(self.cfg.k_max, new_k)))
        if isinstance(self.manifold.k, torch.nn.Parameter):
            self.manifold.k.data.fill_(new_k)
        else:
            self.manifold.k.fill_(new_k)

    def _state_index(self, k_value: float) -> int:
        k = max(self.cfg.k_min, min(self.cfg.k_max, k_value))
        span = self.cfg.k_max - self.cfg.k_min
        if span <= 0:
            return 0
        pos = (k - self.cfg.k_min) / span
        idx = int(round(pos * (self.q_ace.shape[0] - 1)))
        return max(0, min(self.q_ace.shape[0] - 1, idx))

    def _epsilon_greedy(self, q_values: torch.Tensor) -> int:
        if random.random() < self.cfg.epsilon:
            return random.randrange(q_values.numel())
        return int(torch.argmax(q_values).item())

    def _estimate_sectional_curvature(self, h: Optional[torch.Tensor]) -> float:
        # Proxy estimate: use negative squared tangent norm magnitude.
        if h is None or h.numel() == 0:
            return -1.0 / max(self._current_k(), 1e-8)
        with torch.no_grad():
            tangent = self.manifold.logmap0(h.detach())
            sq = (tangent[..., 1:] ** 2).sum(dim=-1)
            est = -float(torch.clamp(sq.mean(), min=1e-8).item())
        return est

    def propose_and_maybe_apply(self, embedding_hint: Optional[torch.Tensor] = None):
        if not self.cfg.enabled:
            return

        k_old = self._current_k()
        state = self._state_index(k_old)

        ace_action = self._epsilon_greedy(self.q_ace[state])
        delta = float(self.cfg.ace_action_deltas[ace_action])

        kappa_hat = self._estimate_sectional_curvature(embedding_hint)
        candidate = (1.0 - self.cfg.gamma_mix) * k_old + (max(-self.cfg.gamma_mix * kappa_hat, 1e-8) ** 0.5)
        candidate = candidate * (1.0 + delta)
        candidate = max(self.cfg.k_min, min(self.cfg.k_max, candidate))

        hgnn_action = self._epsilon_greedy(self.q_hgnn[state])
        accepted = int(hgnn_action == 1)

        if accepted:
            self._set_k(candidate)
            next_k = candidate
        else:
            next_k = k_old

        next_state = self._state_index(next_k)
        self.pending_transition = {
            "state": state,
            "ace_action": ace_action,
            "hgnn_action": hgnn_action,
            "next_state": next_state,
            "k_old": k_old,
            "k_candidate": candidate,
            "accepted": accepted,
        }

    def update_with_metric(self, metric_value: Optional[float]):
        if not self.cfg.enabled or metric_value is None:
            return

        metric = float(metric_value)
        if self.prev_metric is None or self.pending_transition is None:
            self.prev_metric = metric
            return

        reward = metric - self.prev_metric
        t = self.pending_transition
        s = t["state"]
        a_ace = t["ace_action"]
        a_hgnn = t["hgnn_action"]
        ns = t["next_state"]

        alpha = self.cfg.alpha
        beta = self.cfg.beta

        self.q_ace[s, a_ace] += alpha * (
            reward + beta * torch.max(self.q_ace[ns]) - self.q_ace[s, a_ace]
        )
        self.q_hgnn[s, a_hgnn] += alpha * (
            reward + beta * torch.max(self.q_hgnn[ns]) - self.q_hgnn[s, a_hgnn]
        )

        self.prev_metric = metric
        self.cfg.epsilon = max(self.cfg.epsilon_min, self.cfg.epsilon * self.cfg.epsilon_decay)
        self.step_count += 1

        if self.cfg.verbose and self.cfg.log_every > 0 and (self.step_count % self.cfg.log_every == 0):
            k_now = self._current_k()
            status = "accept" if t["accepted"] == 1 else "reject"
            print(
                f"[CurvatureRL] step={self.step_count} "
                f"k_old={t['k_old']:.4f} k_cand={t['k_candidate']:.4f} k_now={k_now:.4f} "
                f"decision={status} reward={reward:.6f} eps={self.cfg.epsilon:.4f}"
            )
