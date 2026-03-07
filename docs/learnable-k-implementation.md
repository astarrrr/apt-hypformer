# 可学习曲率 k 的实现思路与方法

## 1. 核心思想

将 Lorentz 流形的曲率参数 `k`（对应曲率 `-1/k`）从固定超参数改为可训练参数，通过标准反向传播随模型一起优化。损失信号来自下游任务（`predict_edge_type` 交叉熵），无需额外损失项。

## 2. 为什么 k 能获得梯度

k 出现在编码器内部每一层的时间分量计算中：

```python
# HypLinear / HypLayerNorm / HypActivation 里
x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()

# TransConvLayer attention 里
attn_output_time = ((attn_output ** 2).sum(...) + self.manifold.k) ** 0.5

# expmap0 里（Euclidean -> Lorentz 映射）
l_v = torch.cosh(nomin / torch.sqrt(k)) * torch.sqrt(k)
```

梯度路径：`交叉熵 loss → h_src/h_dst（空间分量）→ 各层 x_time（含 k）→ k.grad`

## 3. 关键问题：manifold 实例共享

**问题**：原代码中 encoder 和 decoder/objective 各自创建独立的 `Lorentz` 实例，k 改了一边另一边不知道。

**解决**：在 `factory_ext.py` 中引入模块级注册表，确保三者共享同一个 `Lorentz` 实例：

```python
_shared_manifold = None  # 模块级注册表

def create_shared_manifold(cfg):
    global _shared_manifold
    _shared_manifold = Lorentz(k=k, learnable=learnable_k)
    return _shared_manifold
```

调用顺序（由 PIDSMaker 的 `build_model` 驱动）：
1. `encoder_factory` → `create_hyp_encoder` → `create_shared_manifold` → 注册表写入
2. `objective_factory` → `create_hyp_objective` → 从注册表读取同一实例
3. `optimizer_factory` → `create_dual_optimizer` → 从注册表读取 k 参数

验证：`assert enc.manifold is obj.manifold is obj.decoder.manifold`

## 4. 数值稳定性：k 的 clamp

梯度更新可能让 k 越过 0 变负，导致 `sqrt(k)` 变 NaN 崩溃。

**解决**：在 `DualOptimizer.step()` 末尾强制 clamp：

```python
# optimizer.py
for param, lo, hi in self.clamp_params:
    param.data.clamp_(lo, hi)   # k 被 clamp 在 [0.1, 10.0]
```

clamp 范围由 `factory_ext.create_dual_optimizer` 注册：
```python
clamp_params = [(manifold.k, 0.1, 10.0)]
```

## 5. 编码器输出维度适配

Lorentz 点形状为 `[N, out_dim+1]`（含时间分量 index 0），而 `predict_edge_type` 的 MLP decoder 期望 `[N, out_dim]`。

**解决**：encoder forward 输出时去掉时间分量：

```python
# hyp_transformer_encoder.py
h = self.output_proj(z)[..., 1:]  # [N, out_dim]，去掉 index 0
return {"h": h}
```

k 的梯度仍然有效，因为 x_space 通过各层的 x_time（含 k）间接依赖 k。

## 6. 涉及的文件改动

| 文件 | 改动内容 |
|------|---------|
| `manifolds/lorentz.py` | 无需改动，`geoopt.Lorentz(learnable=True)` 已支持 |
| `encoders/hyp_transformer_encoder.py` | `__init__` 增加 `manifold=None` 参数；`forward` 输出去掉时间分量 `[..., 1:]` |
| `factory_ext.py` | 增加 `_shared_manifold` 注册表、`create_shared_manifold()`；三个工厂函数共享同一 manifold |
| `optimizer.py` | `DualOptimizer` 增加 `clamp_params`、`k_log_every`；`step()` 后执行 clamp 和 wandb 上报 |
| `configs/hyp_pids.yml` | 增加 `learnable_k: true`、`k_log_every: 200` |
| `PIDSMaker/config/config.py` | `ENCODERS_CFG` 注册 `learnable_k`、`k_log_every` 两个字段 |

## 7. 配置项

```yaml
# configs/hyp_pids.yml
training:
  encoder:
    hyperbolic_transformer:
      k: 1.0          # k 初始值
      learnable_k: true   # 开启梯度学习
      k_log_every: 200    # 每 200 个 edge batch 打印/上报一次 k；0 表示关闭
```

## 8. k 的几何含义

| k 值 | 曲率 (-1/k) | 空间特性 |
|------|------------|---------|
| 小（如 0.5）| 更负（-2.0）| 更弯曲，适合层次/树状结构 |
| 大（如 5.0）| 接近 0（-0.2）| 接近平坦欧氏空间 |

k 收敛的方向反映了数据的内在几何结构——溯源图若具有强层次性，k 倾向变小；若结构偏平坦，k 倾向变大。

## 9. 监控 k 的学习过程

训练时会打印：
```
[HypCurvature] step=200  k=0.9873
[HypCurvature] step=400  k=0.9651
```

加 `--wandb` 时自动上报到 wandb 的 `hyp/k` 指标，可在面板查看 k 的变化曲线。
