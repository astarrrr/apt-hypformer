# RL 曲率学习 k：代码研读与落地方案

## 1. 现状盘点

### 1.1 已完成的工作

| 文件 | 状态 | 说明 |
|------|------|------|
| `docs/rl-learnk.md` | 完成 | ACE-HGNN 论文逻辑梳理 |
| `docs/rl-learnk-proposal.md` | 完成 | 针对本项目的适配提案 |
| `manifolds/curvature_controller.py` | 已实现 | `CurvatureRLController` 完整实现，含双 Q 表、epsilon-greedy、`propose_and_maybe_apply()`、`update_with_metric()` |
| `manifolds/lorentz.py` | 已实现 | `Lorentz(k, learnable)` 构造器已支持梯度学习 |

**关键发现：`CurvatureRLController` 已经写好，但尚未接入任何训练流程。** 编码器、解码器和 PIDSMaker 的训练循环都还不知道它的存在。

### 1.2 当前架构中的 manifold 实例问题

研读代码后发现一个关键缺陷：目前存在**两个独立的 `Lorentz` 实例**：

```
factory_ext.create_hyp_encoder()  ->  HypTransformerEncoder.__init__
    self.manifold = Lorentz(k=float(k))   # 实例 A

factory_ext.create_hyp_objective() ->  HypEdgeReconstruction.__init__
    manifold = Lorentz(k=float(k))        # 实例 B（与 A 无关）
```

如果 `CurvatureRLController` 只修改实例 A 的 `k`，解码器的 Lorentz 距离计算仍使用旧的实例 B，导致训练目标与编码几何不一致。**必须共享同一个 manifold 实例。**

---

## 2. 可行性分析

### 2.1 梯度下降学习 k（Path A）

`geoopt.Lorentz` 在 `learnable=True` 时会将 `k` 注册为 `nn.Parameter`，标准反向传播即可优化。本项目的 `Lorentz` 类直接继承 `geoopt.Lorentz`，已完整支持这一能力：

```python
# manifolds/lorentz.py
class Lorentz(LorentzOri):
    def __init__(self, k=1.0, learnable=False):
        super().__init__(k, learnable)   # geoopt 已处理 learnable 逻辑
```

当 `learnable=True` 时，`manifold.k` 是普通 `nn.Parameter`（非 `ManifoldParameter`），在 `DualOptimizer` 里会自动分配到 Adam 组，**无需额外修改优化器**。

**路径 A 的工作量：约 3 处改动，30 行代码以内。**

### 2.2 RL 双 Agent 学习 k（Path B）

核心阻力是 **Reward 信号极度稀疏**：

- 当前配置 `num_epochs: 20`
- ADP（验证集检测分数）只在每个 epoch 结束后才可得
- 20 epochs = 最多 20 条 reward 信号
- Q-table 收敛通常需要数百到数千条信号

此外，PIDSMaker 的训练循环目前没有 epoch 结束的回调钩子，需要在 `pidsmaker/factory.py` 增加第 5 处改动。

**路径 B 的工作量：约 6-7 处改动，但实验效果在 20 epoch 内不可靠。**

### 2.3 建议

> **先跑 Path A（梯度学习 k），作为 baseline。**
> 若 baseline 效果好，再开启 Path B 做消融对比。
> 两条路径的 manifold 共享改动是共有前提，Path A 完成后 Path B 改动量更小。

---

## 3. Path A：梯度下降学习 k（推荐首选）

### 3.1 总体架构

```
factory_ext.py
    shared_manifold = Lorentz(k=float(k), learnable=True)   # 唯一实例
        |
        +---> HypTransformerEncoder(manifold=shared_manifold)
        |         所有 HypLinear、TransConv、layers 都引用同一个 manifold
        |
        +---> HypEdgeDecoder(manifold=shared_manifold)
        |         square_dist、logmap0 使用同一个 k
        |
        +---> HypEdgeReconstruction(manifold=shared_manifold)

DualOptimizer:
    euc_params = [..., manifold.k, ...]    # k 是 nn.Parameter，进 Adam 组
```

k 的梯度由重建损失 `HypEdgeReconstruction.forward` 反向传播自动计算，无需任何额外损失项。

### 3.2 变更一：`encoders/hyp_transformer_encoder.py`

**目标**：`HypTransformerEncoder.__init__` 接受外部传入的 `manifold`，不再自己创建。

```python
# 原来（第 336 行附近）
def __init__(self, in_dim, hid_dim, out_dim, ..., k=1.0, ...):
    super().__init__()
    self.manifold = Lorentz(k=float(k))

# 改为
def __init__(self, in_dim, hid_dim, out_dim, ..., k=1.0, manifold=None, ...):
    super().__init__()
    self.manifold = manifold if manifold is not None else Lorentz(k=float(k))
```

其余代码（`TransConv`、`GraphConv`、`output_proj` 等）均通过 `self.manifold` 引用，无需修改。

**影响范围**：仅 `__init__` 签名，完全向下兼容（`manifold=None` 时行为不变）。

### 3.3 变更二：`decoders/hyp_edge_decoder.py`

与编码器相同的模式：增加 `manifold=None` 参数。

```python
# 改造前
def __init__(self, ..., manifold):
    self.manifold = manifold

# 不需要改：decoder 本来就接收 manifold 参数，只需确保调用时传入共享实例
```

读取 `decoders/hyp_edge_decoder.py` 确认后，若构造函数已经接收 `manifold`，则不需要修改 decoder 本身。

### 3.4 变更三：`factory_ext.py`（核心改动）

将两个工厂函数改为共享同一个 manifold 实例，并开启 `learnable=True`：

```python
# factory_ext.py 新增：共享 manifold 工厂
def create_shared_manifold(cfg):
    """Create a single shared Lorentz manifold, optionally with learnable k."""
    enc_cfg = cfg.training.encoder.hyperbolic_transformer
    k = float(enc_cfg.k)
    learnable_k = getattr(enc_cfg, 'learnable_k', False)
    return Lorentz(k=k, learnable=learnable_k)


def create_hyp_encoder(cfg, in_dim, manifold=None):
    """改动：接受外部 manifold，不再内部创建。"""
    hid_dim = cfg.training.node_hid_dim
    out_dim = cfg.training.node_out_dim
    enc_cfg = cfg.training.encoder.hyperbolic_transformer

    encoder = HypTransformerEncoder(
        in_dim=in_dim,
        hid_dim=hid_dim,
        out_dim=out_dim,
        ...
        manifold=manifold,      # 传入共享实例；None 时内部自建（向下兼容）
    )
    return encoder


def create_hyp_objective(cfg, node_out_dim, manifold=None):
    """改动：接受外部 manifold。"""
    enc_cfg = cfg.training.encoder.hyperbolic_transformer
    k = enc_cfg.k

    if manifold is None:
        manifold = Lorentz(k=float(k))   # 向下兼容

    decoder = HypEdgeDecoder(
        manifold=manifold,     # 共享
        ...
    )
    objective = HypEdgeReconstruction(
        decoder=decoder,
        manifold=manifold,     # 共享
        loss_type='lorentz_dist',
    )
    return objective
```

`pidsmaker/factory.py` 中的调用点（CLAUDE.md 第 2、3 处插入点）需同步更新为：

```python
# pidsmaker/factory.py，encoder_factory() 内（原第 238-239 行附近）
if encoder_name == 'hyperbolic_transformer':
    shared_manifold = create_shared_manifold(cfg)         # 新建共享 manifold
    encoder = create_hyp_encoder(cfg, in_dim, manifold=shared_manifold)
    encoder._shared_manifold = shared_manifold            # 存储引用，后续 objective 用

# pidsmaker/factory.py，objective_factory() 内（原第 392-395 行附近）
if objective_name == 'hyperbolic_edge_reconstruction':
    # 取出 encoder 上存储的共享 manifold
    shared_manifold = getattr(encoder_instance, '_shared_manifold', None)
    objective = create_hyp_objective(cfg, node_out_dim, manifold=shared_manifold)
    continue
```

### 3.5 变更四：`configs/hyp_pids.yml`

```yaml
training:
  encoder:
    hyperbolic_transformer:
      k: 1.0             # k 初始值
      learnable_k: true  # 新增：开启梯度学习
```

同步在 `pidsmaker/config/config.py` 的 `ENCODERS_CFG["hyperbolic_transformer"]` 里加入 `learnable_k: False`（默认关闭，向下兼容）。

### 3.6 数值稳定性注意事项

k 在梯度更新中可能越过正值域变负，需在 `Lorentz.__init__` 或 `factory_ext.create_shared_manifold` 中添加约束：

```python
# 方案一：激活函数约束（推荐）
# 在所有使用 self.k 的地方改为 self.k.abs().clamp(min=0.1)
# 但这样改动面太广

# 方案二：在 optimizer step 后 clamp（推荐更简单）
# DualOptimizer.step() 之后：
if hasattr(shared_manifold, 'k') and isinstance(shared_manifold.k, nn.Parameter):
    shared_manifold.k.data.clamp_(min=0.1, max=10.0)
```

最简单的做法：在 `DualOptimizer.step()` 后，由 `factory.py` 的训练循环执行 clamp，或在 `Lorentz` 的所有使用 `self.k` 的方法里改为 `self.k.clamp(min=1e-2)`。

---

## 4. Path B：RL 双 Agent（ACE-HGNN）

Path A 完成后，`CurvatureRLController` 已有实现，需要连接三个点。

### 4.1 变更一：共享 manifold（与 Path A 相同）

`CurvatureRLController` 的 `_set_k()` 直接写入 `manifold.k.data`：

```python
def _set_k(self, new_k: float):
    ...
    if isinstance(self.manifold.k, torch.nn.Parameter):
        self.manifold.k.data.fill_(new_k)
    else:
        self.manifold.k.fill_(new_k)
```

因此 controller 持有的 manifold 必须是共享实例，否则 decoder 的几何不更新。

### 4.2 变更二：`factory_ext.py` 增加 controller 工厂

```python
from manifolds.curvature_controller import CurvatureRLConfig, CurvatureRLController

def create_curvature_controller(cfg, manifold):
    """Create RL controller for adaptive curvature. Returns None if disabled."""
    rl_cfg_raw = getattr(cfg.training.encoder.hyperbolic_transformer, 'curvature_rl', None)
    if rl_cfg_raw is None or not getattr(rl_cfg_raw, 'enabled', False):
        return None

    rl_cfg = CurvatureRLConfig(
        enabled=True,
        k_min=getattr(rl_cfg_raw, 'k_min', 0.5),
        k_max=getattr(rl_cfg_raw, 'k_max', 5.0),
        num_state_bins=getattr(rl_cfg_raw, 'num_state_bins', 16),
        ace_action_deltas=tuple(getattr(rl_cfg_raw, 'ace_action_deltas', (-0.2, -0.1, 0.0, 0.1, 0.2))),
        alpha=getattr(rl_cfg_raw, 'alpha', 0.05),
        beta=getattr(rl_cfg_raw, 'beta', 0.9),
        epsilon=getattr(rl_cfg_raw, 'epsilon', 0.3),
        epsilon_min=getattr(rl_cfg_raw, 'epsilon_min', 0.05),
        epsilon_decay=getattr(rl_cfg_raw, 'epsilon_decay', 0.995),
        log_every=getattr(rl_cfg_raw, 'log_every', 1),
        verbose=True,
    )
    return CurvatureRLController(manifold=manifold, cfg=rl_cfg)
```

在 `encoder_factory()` 内：

```python
if encoder_name == 'hyperbolic_transformer':
    shared_manifold = create_shared_manifold(cfg)
    encoder = create_hyp_encoder(cfg, in_dim, manifold=shared_manifold)
    controller = create_curvature_controller(cfg, shared_manifold)
    encoder._shared_manifold = shared_manifold
    encoder._curvature_controller = controller   # 挂载到 encoder，供后续访问
```

### 4.3 变更三：`encoders/hyp_transformer_encoder.py` 存储最后一个 embedding

Controller 的 `propose_and_maybe_apply(embedding_hint)` 使用 embedding 的 logmap0 切空间范数估计截面曲率。需要在 forward 时缓存：

```python
def forward(self, x, edge_index, **kwargs):
    ...
    h = self.output_proj(z)   # [N, out_dim+1]
    # 缓存最后一批 embedding（用于曲率估计，不参与反向传播）
    self._last_h = h.detach()
    return {"h": h}
```

### 4.4 变更四：PIDSMaker 训练循环（第 5 处修改点）

这是唯一需要修改 PIDSMaker 训练流程的地方。在每个 epoch 的验证步结束后，取出验证 ADP 分数，调用 controller：

```python
# pidsmaker/factory.py 或其调用的训练循环内
# 位置：每个 epoch 验证结束后，获得 adp_val 之后

controller = getattr(encoder_instance, '_curvature_controller', None)
if controller is not None:
    hint = getattr(encoder_instance, '_last_h', None)
    controller.propose_and_maybe_apply(embedding_hint=hint)
    controller.update_with_metric(metric_value=adp_val)
```

**注意**：`propose_and_maybe_apply()` 必须在 `update_with_metric()` 之前调用，因为 `update_with_metric` 使用 `pending_transition`（由 propose 设置）计算 Q 值更新。

### 4.5 变更五：`configs/hyp_pids.yml`

```yaml
training:
  encoder:
    hyperbolic_transformer:
      k: 1.0
      curvature_rl:
        enabled: true
        k_min: 0.5
        k_max: 5.0
        num_state_bins: 16
        ace_action_deltas: [-0.2, -0.1, 0.0, 0.1, 0.2]
        alpha: 0.05
        beta: 0.9
        epsilon: 0.3
        epsilon_min: 0.05
        epsilon_decay: 0.995
        log_every: 1        # 每 epoch 打印一次（共 20 epoch，不嫌多）
```

同步在 `pidsmaker/config/config.py` 的 `ENCODERS_CFG` 里注册 `curvature_rl` 子配置（含上述所有字段）。

---

## 5. 两条路径对比

| 维度 | Path A（梯度学习） | Path B（RL 双 Agent） |
|------|-------------------|-----------------------|
| 代码改动量 | ~4 处，~50 行 | ~7 处，~120 行 |
| 收敛速度 | 与模型同步，每 batch 更新 | epoch 级，20 epoch 内样本极少 |
| 理论基础 | 黎曼梯度下降，成熟 | 强化学习，需要充足探索步数 |
| 可解释性 | 直接看 k 梯度和 loss 变化 | 需额外监控 Q 值、reward 轨迹 |
| 风险 | k 需 clamp，防止梯度炸飞 | Q 表 20 步内几乎无法有效收敛 |
| 推荐场景 | 生产/首次实验 | 学术对比消融 |

---

## 6. 关键陷阱与解决方案

### 6.1 k 为负的数值崩溃

**问题**：梯度更新可能让 k 越过 0，导致 `sqrt(k)` 变 NaN，整个训练崩溃。

**解决**：在 `DualOptimizer.step()` 内部（或外部），对 learnable k 做后处理：

```python
# optimizer.py，DualOptimizer.step() 末尾
def step(self):
    for opt in self.optimizers:
        opt.step()
    # clamp learnable curvature parameters
    for group in self.param_groups:
        for p in group['params']:
            if hasattr(p, '_is_curvature_k') and p._is_curvature_k:
                p.data.clamp_(min=0.1, max=10.0)
```

或更简单：在 `create_shared_manifold` 里给 `manifold.k` 打标记 `manifold.k._is_curvature_k = True`，然后在训练循环里每步 clamp。

### 6.2 decoder 与 encoder 几何不同步

**问题**：若 decoder 持有旧 manifold 实例，loss 计算用旧 k，梯度不传到新 k。

**解决**：Path A/B 的核心前提——必须共享同一个 manifold 实例（见变更三）。验证方式：

```python
assert encoder.manifold is objective.manifold is objective.decoder.manifold
```

### 6.3 BatchNorm 与 HypLayerNorm 对 k 的依赖

`HypLayerNorm.forward`（`manifolds/layers.py:24`）计算 time 分量时用了 `self.manifold.k`：

```python
x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
```

由于共享同一个 manifold 实例，k 变化后 LayerNorm 的 time 分量计算自动更新，无需额外处理。`linear_focus_attention`（`encoders/hyp_transformer_encoder.py:102`）同理：

```python
attn_output_time = ((attn_output ** 2).sum(dim=-1, keepdims=True) + self.manifold.k) ** 0.5
```

这是使用共享 manifold 的隐性收益——所有依赖 k 的操作自动联动。

### 6.4 Path B 的 Reward 信号稀疏问题（务必重视）

20 epoch 内，Q 表每个 `(state, action)` 对平均被访问不超过 2-3 次，epsilon-greedy 探索基本等价于随机游走。建议：

- 先用 Path A 确认 k 收敛到合理范围（如 0.8~3.0）
- 以此范围初始化 Path B 的 Q 表偏向（`Q_hgnn[:, 1] += 0.1` 让 accept 动作有微弱先验）
- 或者将 Path B 的 reward 改用 batch 级 loss 改善量（而非 epoch 级 ADP），但这会引入更大噪声

---

## 7. 推荐实施顺序

```
阶段 1（1天）：
  - 实现 manifold 共享（变更三，factory_ext.py）
  - HypTransformerEncoder 接受 manifold 参数（变更一）
  - 验证：assert encoder.manifold is objective.manifold

阶段 2（0.5天）：
  - 开启 learnable_k: true，添加 k clamp
  - 运行 smoke test：k 梯度非零，loss 正常下降
  - 监控：k 轨迹是否收敛、是否 NaN

阶段 3（1-2天，可选）：
  - 接入 CurvatureRLController
  - PIDSMaker factory.py 第 5 处修改（epoch 回调）
  - 与 Path A 结果对比
```

---

## 8. 快速验证脚本

```python
import sys; sys.path.insert(0, '/home/astar/projects/myproject')
import torch
from manifolds.lorentz import Lorentz
from encoders.hyp_transformer_encoder import HypTransformerEncoder
from decoders.hyp_edge_decoder import HypEdgeDecoder
from objectives.hyp_edge_reconstruction import HypEdgeReconstruction

# 1. 共享 manifold，learnable k
manifold = Lorentz(k=1.0, learnable=True)

# 2. 构建模型
enc = HypTransformerEncoder(
    in_dim=128, hid_dim=64, out_dim=32,
    trans_num_layers=2, trans_num_heads=4, trans_dropout=0.3,
    gnn_num_layers=2, gnn_dropout=0.3, graph_weight=0.5,
    attention_type='linear_focused', power_k=2,
    use_bn=True, use_residual=True,
    manifold=manifold,  # 传入共享实例
)

# 3. 验证 manifold 是同一个对象
assert enc.manifold is manifold, "manifold 未共享！"
assert enc.trans_conv.manifold is manifold, "TransConv manifold 未共享！"

# 4. 前向传播，检查 k 梯度
x = torch.randn(50, 128)
edge_index = torch.randint(0, 50, (2, 100))
out = enc(x, edge_index)
h = out["h"]

# 模拟损失
loss = h.norm()
loss.backward()

print(f"k = {manifold.k.item():.4f}")
print(f"k.grad = {manifold.k.grad}")   # 应非 None 且非零
assert manifold.k.grad is not None, "k 没有梯度！"
print("Path A 验证通过")
```

---

## 9. 结论

**可以将 ACE-HGNN 的曲率学习思想用在本模型上。**

- **Path A（梯度学习）** 是落地成本最低、最可靠的路径，核心改动只有 manifold 共享和 `learnable=True`，geoopt 和现有代码已经完整支持。
- **Path B（RL 双 Agent）** 代码框架（`CurvatureRLController`）已经写好，主要缺少与 PIDSMaker 训练循环的接口（1 处 epoch 回调），以及 manifold 共享（与 Path A 共用）。在 20 epoch 的训练规模下，RL 效果存疑，建议作为消融实验而非主路径。
- **两条路径的共同前提是 manifold 共享**，这是最关键的一处架构修正，且与业务逻辑无关，应当最先完成。
