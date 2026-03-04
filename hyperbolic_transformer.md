# HypFormer：双曲空间 Transformer 项目文档

> 论文：[Hypformer: Exploring Efficient Hyperbolic Transformer Fully in Hyperbolic Space](https://dl.acm.org/doi/10.1145/3637528.3672039)（KDD 2024）
> 代码：`/home/astar/projects/hyperbolicTransformer-master`
> 作者：Menglin Yang, Harshit Verma, Delvin Ce Zhang, Jiahong Liu, Irwin King, Rex Ying（Yale University）

---

## 1. 项目概述

HypFormer 是首个**完全在双曲空间中运行**的 Transformer 架构实现。与仅在嵌入层使用双曲几何的方法不同，HypFormer 的注意力机制、线性变换、归一化等全部在 **Lorentz 双曲模型**中进行。

**核心优势**：
- 双曲空间天然适合表示树形/层次化数据，同等维度下表达能力远超欧式空间
- 支持 Full Attention（Softmax 注意力）和 Linear Focused Attention（线性核近似，O(n) 复杂度）
- 可与图卷积网络（GNN）分支联合使用

---

## 2. 目录结构

```
hyperbolicTransformer-master/
├── Hypformer/                    # 简化版可复用模块（推荐入门）
│   ├── hypformer.py             # 核心 HypFormer 模型类
│   ├── main.py                  # 简单使用示例
│   └── manifolds/
│       ├── lorentz.py           # Lorentz 流形实现（核心）
│       ├── lorentz_math.py      # Lorentz 数学运算
│       ├── hyp_layer.py         # 双曲层组件
│       ├── layer.py             # 其他层实现
│       └── utils.py             # 工具函数
├── medium/                       # 中等规模数据集（Cora, Citeseer, Pubmed）
│   ├── main.py
│   ├── parse.py
│   ├── dataset.py
│   └── examples/                # 各数据集运行脚本
├── large/                        # 大规模数据集（OGB）
│   ├── main.py
│   ├── hypformer.py
│   ├── gnns.py
│   ├── parse.py
│   ├── eval.py
│   └── examples/                # 各数据集运行脚本
├── data/                         # 数据集
│   ├── Planetoid/               # Cora, Citeseer, Pubmed
│   ├── OGB/
│   ├── hgcn_data/               # Airport, Disease
│   └── mini_imagenet/
├── requirements.txt
└── README.md
```

---

## 3. 核心概念：Lorentz 双曲模型

点表示为 `[time, x_1, x_2, ..., x_d]`，满足：
- Minkowski 内积：`⟨u, v⟩ = -u_0·v_0 + u_1·v_1 + ... + u_d·v_d`
- 曲率参数 `k`：控制空间弯曲程度（k 越小越"弯"）
- 支持与 Klein 模型、Poincaré 模型相互转换

---

## 4. 模型架构

### 4.1 HypFormer 总体结构

```
输入特征 (Euclidean)
     ↓  expmap0
Lorentz 流形
     ↓
TransConv × N 层    ←→  可选 GNN 分支
     ↓
聚合 (add / cat)
     ↓
解码器 (Euclidean / Hyperbolic)
     ↓
分类输出
```

### 4.2 TransConv 层（双曲注意力）

**Full Attention（全注意力）**：
```python
# 计算双曲内积（与负平方距离成正比）
att_weight = 2 + 2 * manifold.cinner(queries, keys)
att_weight = Softmax(att_weight)
output = manifold.mid_point(values, att_weight)  # 流形上加权平均
```

**Linear Focused Attention（线性近似，推荐用于大图）**：
```python
# 对空间分量应用核函数
phi_q = fp(ReLU(q[1:]), p=power_k)
phi_k = fp(ReLU(k[1:]), p=power_k)
# O(n) 近似
kv = einsum('nhm,nhd->hmd', phi_k, v)
output = einsum('nhm,hmd->nhd', phi_q, kv) / denom
```

### 4.3 双曲层组件（`manifolds/hyp_layer.py`）

| 类名 | 功能 |
|------|------|
| `HypLinear` | 双曲空间中的线性变换 |
| `HypLayerNorm` | 双曲层归一化 |
| `HypActivation` | 双曲空间激活函数 |
| `HypDropout` | 双曲 Dropout |
| `HypCLS` | 基于双曲距离的分类层 |
| `HypNormalization` | 流形上的归一化 |

### 4.4 Lorentz 流形核心操作（`manifolds/lorentz.py`）

| 方法 | 功能 |
|------|------|
| `expmap0(u)` | 切空间向量 → 流形点 |
| `logmap0(x)` | 流形点 → 切空间向量 |
| `cinner(x, y)` | 跨内积（用于注意力计算） |
| `mid_point(x, w)` | 流形上加权平均 |
| `dist(x, y)` | 测地线距离 |

---

## 5. 安装与运行

### 5.1 安装依赖

```bash
pip install -r requirements.txt
```

关键依赖：
- `torch==2.2.1`
- `geoopt==0.5.0`（Riemannian 流形操作，**必须**）
- `torch_geometric==2.5.3`
- `torch_scatter`, `torch_sparse`, `torch_cluster`
- `ogb==1.3.6`（OGB 数据集）

### 5.2 快速验证（简化版模块）

```bash
cd /home/astar/projects/hyperbolicTransformer-master/Hypformer
python main.py
# 预期输出：
# Input shape: torch.Size([10, 16])
# Output shape: torch.Size([10, 5])
```

### 5.3 中等规模数据集

```bash
cd medium
bash examples/cora.sh
bash examples/citeseer.sh
bash examples/pubmed.sh
bash examples/airport.sh
```

### 5.4 大规模数据集（OGB）

```bash
cd large
bash examples/arxiv.sh
bash examples/amazon2M.sh
bash examples/protein.sh
```

---

## 6. 主要配置参数

### 6.1 模型结构参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hidden_channels` | 32 | 隐藏层维度 |
| `trans_num_layers` | 2 | Transformer 层数 |
| `trans_num_heads` | 1 | 注意力头数 |
| `trans_dropout` | 0.0~0.5 | Dropout 比例 |
| `trans_use_bn` | 1 | 是否使用 BatchNorm |
| `trans_use_residual` | 1 | 是否使用残差连接 |
| `trans_use_weight` | 1 | 是否使用可学习权重 |
| `trans_use_act` | 1 | 是否使用激活函数 |
| `attention_type` | `linear_focused` | `linear_focused` 或 `full` |

### 6.2 双曲几何参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `k_in` | 1.0 | 输入流形曲率 |
| `k_hidden` | 1.0 | 隐藏层曲率 |
| `k_out` | 1.0 | 输出层曲率 |
| `power_k` | 2.0 | Linear Attention 核幂次 |
| `decoder_type` | `euc` | `euc`（欧式解码）或 `hyp`（双曲解码） |

### 6.3 图/注意力参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_graph` | 1 | 是否使用 GNN 分支 |
| `graph_weight` | 0.2~0.8 | GNN 分支权重 |
| `gnn_num_layers` | 2~3 | GNN 层数 |
| `gnn_dropout` | 0.0~0.5 | GNN Dropout |
| `aggregate` | `add` | GNN 与 Transformer 特征聚合方式：`add` 或 `cat` |

### 6.4 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lr` | 0.01 | 欧式参数学习率 |
| `hyp_lr` | 0.01 | 双曲参数学习率 |
| `weight_decay` | 0.0~0.005 | L2 正则化 |
| `optimizer_type` | `adam` | 欧式优化器：`adam` 或 `sgd` |
| `hyp_optimizer_type` | `radam` | 双曲优化器：`radam` 或 `rsgd` |
| `epochs` | 500~1000 | 训练轮数 |
| `patience` | 200 | 早停耐心值 |

---

## 7. 支持的数据集

**中等规模**：Cora, Citeseer, Pubmed, Airport, Mini-ImageNet, 20 Newsgroups

**大规模（OGB）**：
- `ogbn-arxiv`（学术论文引用网络）
- `ogbn-products`（亚马逊商品网络，240万节点）
- `ogbn-proteins`（蛋白质互作网络，13.2万节点）

---

## 8. 优化策略

模型对欧式参数和双曲参数使用**分开的优化器**：

```python
# 欧式参数：标准 Adam/SGD
euc_optimizer = Adam(euc_params, lr=lr)

# 双曲参数：Riemannian Adam/SGD（尊重流形几何）
hyp_optimizer = RiemannianAdam(hyp_params, lr=hyp_lr)
```

---

## 9. 关键文件速查

| 文件路径 | 作用 |
|----------|------|
| `Hypformer/hypformer.py` | 核心 HypFormer 模型 |
| `Hypformer/manifolds/lorentz.py` | Lorentz 流形实现 |
| `Hypformer/manifolds/hyp_layer.py` | 双曲层组件 |
| `Hypformer/main.py` | 简单使用示例 |
| `large/hypformer.py` | 大规模版本（含 GNN 分支） |
| `large/parse.py` | 所有命令行参数定义 |
| `large/eval.py` | 评估函数 |
| `medium/main.py` | 中等规模训练脚本 |

---

## 10. 引用

```bibtex
@inproceedings{yang2024hypformer,
  title={Hypformer: Exploring Efficient Hyperbolic Transformer Fully in Hyperbolic Space},
  author={Yang, Menglin and Verma, Harshit and Zhang, Delvin Ce and Liu, Jiahong and King, Irwin and Ying, Rex},
  booktitle={Proceedings of the 2024 ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year={2024}
}
```
