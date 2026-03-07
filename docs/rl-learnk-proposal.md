# 基于 `rl-learnk.md` 的可学习双曲曲率 `k` 方案（改进版）

## 1. 目标与约束

目标：在训练 HGNN/Hyperbolic Encoder 的同时，自动学习更合适的曲率参数 `k`（对应曲率 `-1/k`），提升下游检测性能并保持训练稳定。

约束：
- `k > 0`，并限制在 `[k_min, k_max]`。
- 曲率更新不能过快，避免几何空间频繁抖动导致 loss 爆炸。
- 奖励应和任务目标直接对齐（优先验证集指标，其次训练损失改进）。

---

## 2. 两个 Agent 的职责

参考 `docs/rl-learnk.md` 的 ACE/HGNN 分工，定义：

1. `ACE-Agent`（Proposal Agent）  
作用：在当前状态下提出候选曲率 `k_candidate`。

2. `HGNN-Agent`（Gate Agent）  
作用：决定是否接受 `ACE-Agent` 的提议（`accept/reject`）。

两者共享同一个环境反馈（reward），但维护各自 Q 函数。

---

## 3. 状态、动作、奖励定义

## 3.1 状态 `s_t`

建议使用离散状态，避免 Q-table 维度过大。

基础状态（必选）：
- `bin(k_t)`：当前 `k` 所在区间（`num_state_bins` 个桶）。

增强状态（可选）：
- `bin(Δloss_t)`：最近一步 loss 改变量；
- `bin(var(h_t))`：embedding 范数/方差的粗粒度统计；
- `accept_rate_window`：最近窗口内接受率。

若先追求稳定，第一版建议仅用 `bin(k_t)`。

## 3.2 ACE 动作 `a_t^ace`

动作不是直接给绝对值，而是给相对步长，更稳：
- `A_ace = { -η2, -η1, 0, +η1, +η2 }`
- 例如 `{ -0.20, -0.10, 0, +0.10, +0.20 }`

更新为：
`k_candidate = clip(k_base * (1 + a_t^ace), [k_min, k_max])`

其中 `k_base` 可定义为：
- 方案 A：`k_base = k_t`（最稳，推荐）；
- 方案 B：`k_base = (1-γ)k_t + sqrt(-γ * kappa_hat)`（沿用文档估计项）。

## 3.3 HGNN 动作 `a_t^hgnn`

二值动作：
- `0`: reject
- `1`: accept

执行：
- accept：`k_{t+1} = k_candidate`
- reject：`k_{t+1} = k_t`

## 3.4 奖励 `r_t`（改为 ADP）

主奖励定义为 ADP 提升：

`r_t = ADP_val(t) - ADP_val(t-1)`

解释：
- `r_t > 0`：本轮曲率决策让检测效果提升；
- `r_t < 0`：本轮决策让检测效果下降。

再加稳定性正则（建议）：
`r_t = r_t - λ1 * |k_{t+1}-k_t| - λ2 * I[nan_or_inf]`

这样可显著减少“曲率抖动”和极端动作。

---

## 4. 学习与更新

使用双 Q-learning：

- `Q_ace(s,a)`：提议动作价值
- `Q_hgnn(s,a)`：门控动作价值

更新：

`Q(s_t,a_t) <- Q(s_t,a_t) + α * (r_t + β * max_a Q(s_{t+1},a) - Q(s_t,a_t))`

策略：
- `epsilon-greedy`
- `epsilon` 线性或指数衰减到 `epsilon_min`

---

## 5. 训练节奏（关键）

不建议每个 batch 都改 `k`。建议“慢时钟更新”：

- 每 `T` step（如 20/50）才执行一次 agent 决策；
- 其余 step 固定 `k` 训练。

理由：曲率变化会重定义几何距离，更新太频繁会引入高噪声反馈。

---

## 6. 与当前代码的落地接口建议

在当前工程中可按以下接口组织：

1. `manifolds/curvature_controller.py`
- `propose_and_maybe_apply()`：执行 ACE+HGNN 决策；
- `update_with_metric(metric)`：接收 reward 信号并更新 Q。

2. `encoders/hyp_transformer_encoder.py`
- 在 forward 前判断是否到达 `T` 步，触发一次 `propose_and_maybe_apply()`。

3. 训练/评估循环（PIDSMaker）
- 在每个 epoch 结束后的验证阶段拿到 `ADP_val(epoch)`；
- 调用 `update_with_metric(metric=ADP_val(epoch))` 更新双 Q。

4. `configs/hyp_pids.yml`
- 增加 `curvature_rl` 配置：`enabled/k_min/k_max/epsilon/alpha/beta/update_interval/log_every` 等。

---

## 7. 建议的默认超参数

- `k_min=0.5`, `k_max=5.0`
- `actions=[-0.2,-0.1,0,+0.1,+0.2]`
- `alpha=0.05`, `beta=0.9`
- `epsilon=0.3`, `epsilon_min=0.05`, `epsilon_decay=0.995`
- `update_interval=20`
- `λ1=0.05`（曲率变动惩罚）

---

## 8. 评估与消融

至少做三组对比：

1. 固定 `k`（baseline）
2. 仅 ACE（无 accept/reject 门控）
3. ACE + HGNN（完整双 agent）

关注：
- 验证集 ADP（主指标）
- `k` 轨迹是否收敛
- reward 均值是否逐步上升
- 训练是否出现 NaN/梯度异常

---

## 9. 预期收益与风险

收益：
- 自动找到更合适的几何尺度；
- 降低手工调 `k` 成本；
- 对不同数据集可自适应。

风险：
- ADP 属于 epoch 级稀疏奖励，收敛较慢；
- 更新频率过高导致训练不稳定；
- 不同数据划分下 ADP 方差较大，需要平滑（如 EMA）。

结论：先用“慢时钟 + 小动作 + 稳定性惩罚”的保守策略上线，再逐步放开探索强度。
