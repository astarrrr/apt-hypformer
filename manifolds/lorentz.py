import torch.nn
from typing import Tuple, Optional
import manifolds.lorentz_math as math
import geoopt
from geoopt import Lorentz as LorentzOri
from geoopt.utils import size2shape
import torch


class Lorentz(LorentzOri):
    def __init__(self, k=1.0, learnable=False):
        super().__init__(k, learnable)

    def _check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5, dim=-1) -> Tuple[bool, Optional[str]]:
        dn = x.size(dim) - 1
        x = x ** 2
        quad_form = -x.narrow(dim, 0, 1) + x.narrow(dim, 1, dn).sum(dim=dim, keepdim=True)
        ok = torch.allclose(quad_form, -self.k, atol=atol, rtol=rtol)
        reason = None if ok else f"'x' minkowski quadratic form is not equal to {-self.k.item()}"
        return ok, reason

    def _check_vector_on_tangent(self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5, dim=-1) -> Tuple[
        bool, Optional[str]]:
        inner_ = math.inner(u, x, dim=dim)
        ok = torch.allclose(inner_, torch.zeros(1), atol=atol, rtol=rtol)
        reason = None if ok else "Minkowski inner product is not equal to zero"
        return ok, reason

    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False, dim=-1) -> torch.Tensor:
        return math.dist(x, y, k=self.k, keepdim=keepdim, dim=dim)

    def dist0(self, x: torch.Tensor, *, dim=-1, keepdim=False) -> torch.Tensor:
        return math.dist0(x, k=self.k, dim=dim, keepdim=keepdim)

    def cdist(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, keepdim=False) -> torch.Tensor:
        x = x.clone()
        x.narrow(dim, 0, 1).mul_(-1)
        return torch.sqrt(self.k) * math.acosh(-(x @ y.transpose(-1, -2)) / self.k)

    def lorentz_to_klein(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        spatial_coords = x.narrow(dim, 1, x.size(dim) - 1)
        time_like = x.narrow(dim, 0, 1)
        klein_coords = spatial_coords / time_like
        return klein_coords

    def klein_to_lorentz(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        u_norm_sq = (x * x).sum(dim=dim, keepdim=True)
        denominator = torch.sqrt(1 - u_norm_sq).clamp_min(1e-8)
        scalar = torch.sqrt(self.k) / denominator
        time_like = scalar
        spatial_coords = scalar * x
        return torch.cat((time_like, spatial_coords), dim=dim)

    def lorentz_to_poincare(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.lorentz_to_poincare(x, self.k, dim=dim)

    def norm(self, u: torch.Tensor, *, keepdim=False, dim=-1) -> torch.Tensor:
        return math.norm(u, keepdim=keepdim, dim=dim)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.egrad2rgrad(x, u, k=self.k, dim=dim)

    def projx(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.project(x, k=self.k, dim=dim)

    def proju(self, x: torch.Tensor, v: torch.Tensor, *, dim=-1) -> torch.Tensor:
        v = math.project_u(x, v, k=self.k, dim=dim)
        return v

    def proju0(self, v: torch.Tensor) -> torch.Tensor:
        return math.project_u0(v)

    def expmap(self, x: torch.Tensor, u: torch.Tensor, *, norm_tan=True, project=True, dim=-1) -> torch.Tensor:
        if norm_tan:
            u = self.proju(x, u, dim=dim)
        res = math.expmap(x, u, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def expmap0(self, u: torch.Tensor, *, project=True, dim=-1) -> torch.Tensor:
        res = math.expmap0(u, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def logmap(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.logmap(x, y, k=self.k, dim=dim)

    def logmap0(self, y: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.logmap0(y, k=self.k, dim=dim)

    def logmap0back(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.logmap0back(x, k=self.k, dim=dim)

    def inner(self, x: torch.Tensor, u: torch.Tensor, v: Optional[torch.Tensor] = None, *, keepdim=False,
              dim=-1) -> torch.Tensor:
        if v is None:
            v = u
        return math.inner(u, v, dim=dim, keepdim=keepdim)

    def inner0(self, v: torch.Tensor, *, keepdim=False, dim=-1) -> torch.Tensor:
        return math.inner0(v, k=self.k, dim=dim, keepdim=keepdim)

    def cinner(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, keepdim=False) -> torch.Tensor:
        x = x.clone()
        x.narrow(dim, 0, 1).mul_(-1)
        return (x @ y.transpose(dim, -2)) / self.k

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.parallel_transport(x, y, v, k=self.k, dim=dim)

    def transp0(self, y: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.parallel_transport0(y, u, k=self.k, dim=dim)

    def transp0back(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.parallel_transport0back(x, u, k=self.k, dim=dim)

    def transp_follow_expmap(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, dim=-1,
                             project=True) -> torch.Tensor:
        y = self.expmap(x, u, dim=dim, project=project)
        return self.transp(x, y, v, dim=dim)

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1) -> torch.Tensor:
        v = self.logmap0(y, dim=dim)
        v = self.transp0(x, v, dim=dim)
        return self.expmap(x, v, dim=dim)

    def geodesic_unit(self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor, *, dim=-1, project=True) -> torch.Tensor:
        res = math.geodesic_unit(t, x, u, k=self.k)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def random_normal(self, *size, mean=0, std=1, dtype=None, device=None) -> geoopt.ManifoldTensor:
        self._assert_check_shape(size2shape(*size), "x")
        if device is not None and device != self.k.device:
            raise ValueError("`device` does not match the projector `device`, set the `device` argument to None")
        if dtype is not None and dtype != self.k.dtype:
            raise ValueError("`dtype` does not match the projector `dtype`, set the `dtype` argument to None")
        tens = torch.randn(*size, device=self.k.device, dtype=self.k.dtype) * std + mean
        tens /= tens.norm(dim=-1, keepdim=True)
        return geoopt.ManifoldTensor(self.expmap0(tens), manifold=self)

    def origin(self, *size, dtype=None, device=None, seed=42) -> geoopt.ManifoldTensor:
        if dtype is None:
            dtype = self.k.dtype
        if device is None:
            device = self.k.device

        zero_point = torch.zeros(*size, dtype=dtype, device=device)
        zero_point[..., 0] = torch.sqrt(self.k)
        return geoopt.ManifoldTensor(zero_point, manifold=self)

    def mid_point(self, x: torch.Tensor, w: Optional[torch.Tensor] = None, *, dim=-1) -> torch.Tensor:
        if w is not None:
            ave = w @ x
        else:
            ave = x.mean(dim=-2)
        denom = (-self.inner(ave, ave, dim=dim, keepdim=True))
        denom = denom.abs().clamp_min(1e-8).sqrt()
        return torch.sqrt(self.k) * ave / denom

    def square_dist(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, keepdim=True) -> torch.Tensor:
        return -2 * self.k - 2 * self.inner(x, y, dim=dim, keepdim=keepdim)
