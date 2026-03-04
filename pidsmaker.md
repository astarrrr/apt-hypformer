# PIDSMaker：溯源型入侵检测系统框架文档

> 论文：Bilot et al. (2026). "PIDSMaker: Building and Evaluating Provenance-based Intrusion Detection Systems." arXiv:2601.22983
> 代码：`/home/astar/projects/PIDSMaker`
> 版本：4.0.0 | Python >= 3.9 | License: Apache 2.0

---

## 1. 项目概述

PIDSMaker 是首个专门用于构建和实验**溯源型入侵检测系统（PIDS, Provenance-based Intrusion Detection System）**的综合框架。它：

- 提供 8 种最新 PIDS 方法的统一实现，支持一键复现论文结果
- 通过 YAML 配置文件或 CLI 参数灵活定制和组合各模块
- 支持针对 APT（高级持续性威胁）攻击的检测与评估
- 全流程自动化：图构建 → 特征化 → GNN 训练 → 评估 → 告警溯源

**核心思想**：将系统调用日志解析为**溯源图（Provenance Graph）**，节点为系统实体（进程/文件/网络流），边为系统调用事件，然后用深度学习方法识别异常。

---

## 2. 支持的 PIDS 系统

| 系统 | 会议/期刊 | 年份 |
|------|-----------|------|
| Velox | USENIX Security | 2025 |
| Orthrus | USENIX Security | 2025 |
| R-Caid | IEEE S&P | 2024 |
| Flash | IEEE S&P | 2024 |
| Kairos | IEEE S&P | 2024 |
| Magic | USENIX Security | 2024 |
| NodLink | NDSS | 2024 |
| ThreaTrace | IEEE TIFS | 2022 |

---

## 3. 支持的数据集

来自 **DARPA TC（Transparent Computing）** 和 **OpTC** 项目，包含真实 APT 攻击场景的溯源追踪数据：

**E3 数据集**（较小，4~12 GB）：
- CADETS_E3, THEIA_E3, CLEARSCOPE_E3, FIVEDIRECTIONS_E3, TRACE_E3

**E5 数据集**（较大，36~710 GB）：
- CADETS_E5, THEIA_E5, CLEARSCOPE_E5, FIVEDIRECTIONS_E5, TRACE_E5

**OpTC 数据集**：
- optc_h201, optc_h501, optc_h051

---

## 4. 目录结构

```
PIDSMaker/
├── pidsmaker/
│   ├── main.py                  # 统一入口（所有模式）
│   ├── model.py                 # 模型（Encoder + Objectives）
│   ├── factory.py               # 组件工厂
│   ├── config/
│   │   ├── config.py            # 参数定义 & 数据集配置
│   │   └── pipeline.py          # 流水线编排逻辑
│   ├── tasks/                   # 7 个流水线阶段
│   │   ├── construction.py      # Stage 1：图构建
│   │   ├── transformation.py    # Stage 2：图变换
│   │   ├── featurization.py     # Stage 3：节点特征化
│   │   ├── feat_inference.py    # Stage 4：特征推断
│   │   ├── batching.py          # Stage 4b：批次构建
│   │   ├── training.py          # Stage 5：GNN 训练
│   │   ├── evaluation.py        # Stage 6：评估
│   │   └── triage.py            # Stage 7：告警溯源
│   ├── encoders/                # GNN 编码器实现
│   ├── decoders/                # 解码器实现
│   ├── objectives/              # 训练目标（自监督任务）
│   ├── detection/               # 训练/评估循环
│   ├── featurization/           # 文本嵌入方法
│   ├── preprocessing/           # 图构建 & 变换
│   ├── triage/                  # 后处理（攻击路径重建）
│   ├── losses.py                # 损失函数
│   ├── tgn.py                   # TGN 实现
│   └── experiments/             # 超参数调优 & 不确定性量化
├── config/                      # 各系统 YAML 配置文件
│   ├── orthrus.yml
│   ├── velox.yml
│   ├── kairos.yml
│   └── ...（8 个系统配置）
├── Ground_Truth/                # 攻击标注数据
├── dataset_preprocessing/       # 数据集预处理脚本
├── tests/                       # 单元测试
└── Dockerfile
```

---

## 5. 7 阶段流水线

```
原始溯源追踪（PostgreSQL）
         ↓
[Stage 1] Construction（图构建）
  · 按时间窗口切分溯源追踪
  · 构建 NetworkX 图（进程/文件/网络节点 + 系统调用边）
         ↓
[Stage 2] Transformation（图变换）
  · 无向化 / DAG 化 / 合成攻击注入等
         ↓
[Stage 3] Featurization（节点特征化）
  · 训练 Word2Vec/FastText 等嵌入模型
  · 生成节点特征向量
         ↓
[Stage 4] Feature Inference（特征推断）
  · 将嵌入应用到所有图
  · 生成边消息向量
         ↓
[Stage 4b] Batching（批次构建）
  · 全局批次 / 图内批次 / 图间批次 / TGN 邻居采样
         ↓
[Stage 5] Training（GNN 训练）
  · 自监督学习（预测/重建）
  · 推断生成每条边的异常分数
         ↓
[Stage 6] Evaluation（评估）
  · 计算 ADP, F1, Precision, Recall
  · 生成可视化图表
         ↓
[Stage 7] Triage（告警溯源，可选）
  · 重建攻击路径（最短路径、1/2/3-hop、连通分量）
```

---

## 6. 核心组件详解

### 6.1 模型（`model.py`）

`Model` 类组合了：
- **Encoder**：生成节点嵌入的 GNN
- **多个 Objectives**：计算损失/异常分数的自监督任务

关键方法：
```python
model.embed()           # 生成节点嵌入
model.forward()         # 训练时计算损失，推断时计算异常分数
model.to_fine_tuning()  # 切换 SSL 预训练 / 少样本检测模式
model.reset_state()     # 重置 TGN 时序状态
```

### 6.2 支持的 GNN 编码器（`encoders/`）

| 编码器 | 说明 |
|--------|------|
| `tgn` | Temporal Graph Network，带可学习记忆模块 |
| `graph_attention` | 自定义图注意力网络 |
| `sage` | GraphSAGE |
| `gat` | Graph Attention Networks |
| `gin` | Graph Isomorphism Networks |
| `glstm` | GRU+LSTM 组合（NodLink 专用） |
| `rcaid_gat` | R-Caid 专用编码器 |
| `magic_gat` | MAGIC 专用编码器 |
| `custom_mlp` | 简单 MLP |

### 6.3 训练目标（`objectives/`）

**预测类（自监督）**：
- `predict_edge_type`：预测边类型
- `predict_node_type`：预测节点类型
- `predict_edge_contrastive`：对比边预测

**重建类**：
- `reconstruct_node_features`：重建节点特征向量
- `reconstruct_node_embeddings`：重建文本嵌入
- `reconstruct_edge_embeddings`：重建边嵌入
- `reconstruct_masked_features`：掩码特征重建（GMAE 风格）

### 6.4 节点特征化方法（`featurization/`）

| 方法 | 说明 |
|------|------|
| `word2vec` | Word2Vec 嵌入 |
| `doc2vec` | Doc2Vec 嵌入 |
| `fasttext` | FastText 嵌入 |
| `alacarte` | 基于随机游走的嵌入 |
| `temporal_rw` | 时序随机游走嵌入 |
| `flash` | Flash 系统专用 |
| `hierarchical_hashing` | 层次哈希 |
| `magic` | 可学习嵌入（MAGIC 专用） |
| `only_type` | 仅使用节点类型 |

### 6.5 评估方法

| 方法 | 说明 |
|------|------|
| `node_evaluation` | 节点级异常分数 |
| `tw_evaluation` | 时间窗口级检测 |
| `node_tw_evaluation` | 节点+时间窗口联合评估 |
| `queue_evaluation` | Kairos 风格队列检测 |
| `edge_evaluation` | 边级检测 |

**主要评估指标**：
- **ADP（Attack Detection Performance）**：主要指标
- **Discrimination Score**：真正例/假正例分数分离度
- Precision, Recall, F1

---

## 7. 配置系统

### 7.1 YAML 配置文件

```yaml
# config/my_system.yml
_include_yml: orthrus          # 继承已有系统配置

construction:
  time_window_size: 15.0       # 时间窗口大小（分钟）
  node_label_features:
    subject: type, path, cmd_line
    file: type, path
    netflow: type, remote_address, remote_port

featurization:
  used_method: fasttext
  emb_dim: 64

training:
  lr: 0.00001
  node_hid_dim: 128
  encoder:
    used_methods: tgn,graph_attention
  decoder:
    used_methods: predict_edge_type,reconstruct_node_features

evaluation:
  used_method: node_evaluation
  threshold:
    used_method: max_val_loss
```

### 7.2 CLI 参数覆盖

```bash
python pidsmaker/main.py kairos CADETS_E3 \
  --training.lr=0.0001 \
  --training.node_hid_dim=256 \
  --featurization.emb_dim=64
```

### 7.3 图变换配置

| 变换方法 | 说明 |
|----------|------|
| `undirected` | 有向图转无向图 |
| `dag` | 转为有向无环图 |
| `rcaid_pseudo_graph` | R-Caid 专用变换 |
| `synthetic_attack_naive` | 注入合成攻击（数据增强） |

### 7.4 批次构建选项

- **全局批次**：按边数/分钟合并时间窗口
- **图内批次**：将大图分割为小批次
- **图间批次**：并行训练多个图
- **TGN 邻居采样**：TGN 专用最近邻采样

### 7.5 不确定性量化

```yaml
experiment:
  used_method: uncertainty

  uncertainty:
    mc_dropout:
      iterations: 5
      dropout: 0.5
    deep_ensemble:
      iterations: 3
      restart_from: training
    hyperparameter:
      hyperparameters: "lr, num_epochs"
      iterations: 5
      delta: 0.1
```

---

## 8. 安装与运行

### 8.1 安装

```bash
# Docker 方式（推荐）
# 参考官方文档：https://ubc-provenance.github.io/PIDSMaker/ten-minute-install/

# 或直接安装（Python >= 3.9）
pip install -e .
```

### 8.2 基础运行

```bash
# 运行现有系统
python pidsmaker/main.py orthrus CADETS_E3
python pidsmaker/main.py kairos CADETS_E3

# 后台运行
./run.sh velox CADETS_E3
```

### 8.3 高级运行选项

```bash
# 启用 W&B 日志
python pidsmaker/main.py velox CADETS_E3 --wandb --project my_project

# 从特定阶段重新开始（利用缓存跳过之前阶段）
python pidsmaker/main.py velox CADETS_E3 --force_restart=training

# 全新运行（清除所有缓存）
python pidsmaker/main.py velox CADETS_E3 --restart_from_scratch

# 强制 CPU 运行
python pidsmaker/main.py velox CADETS_E3 --cpu
```

### 8.4 超参数调优

```bash
# 创建调优配置文件 config/my_tune.yml
# 内容：
# method: grid
# parameters:
#   training.lr:
#     values: [0.001, 0.0001]
#   training.node_hid_dim:
#     values: [64, 128, 256]

./run.sh my_system CADETS_E3 --tuning_mode=hyperparameters
```

### 8.5 创建新系统

```bash
# 1. 创建配置文件
cat > config/my_pids.yml << 'EOF'
_include_yml: magic

training:
  node_hid_dim: 128
  decoder:
    used_methods: predict_node_type,reconstruct_edge_embeddings
EOF

# 2. 运行
python pidsmaker/main.py my_pids CADETS_E3
```

---

## 9. 任务缓存机制

PIDSMaker 对每个 pipeline 阶段自动计算 **SHA256 哈希**（基于该阶段所有配置参数），完成后写入 `done.txt`。后续运行时，若配置未变，自动跳过该阶段，复用上次结果。

```
artifacts/
└── my_experiment/
    ├── construction_<hash>/
    │   ├── done.txt            ← 存在则跳过该阶段
    │   └── graphs/
    ├── featurization_<hash>/
    │   ├── done.txt
    │   └── embeddings/
    └── training_<hash>/
        └── model.pt
```

---

## 10. 关键配置参数速查

### 数据库配置

```bash
--database_host postgres
--database_user postgres
--database_password postgres
--database_port 5432
```

### 训练参数

| 参数路径 | 说明 |
|----------|------|
| `training.lr` | 学习率 |
| `training.node_hid_dim` | 节点隐藏维度 |
| `training.num_epochs` | 训练轮数 |
| `training.encoder.used_methods` | GNN 编码器（逗号分隔） |
| `training.decoder.used_methods` | 训练目标（逗号分隔） |

### 图构建参数

| 参数路径 | 说明 |
|----------|------|
| `construction.time_window_size` | 时间窗口大小（分钟） |
| `construction.node_label_features.subject` | 进程节点特征字段 |
| `construction.node_label_features.file` | 文件节点特征字段 |
| `construction.node_label_features.netflow` | 网络节点特征字段 |

---

## 11. 已知限制

- **PIDS 不稳定性**：自监督训练对随机种子和超参数敏感，推荐多次运行取平均
- **TGN 限制**：不支持 RCaid pseudo-graph 变换 和图间批次
- **大规模数据集**：E5 数据集批次文件需要 300+ GB 存储空间
- **部分功能待完善**：`tw_evaluation`, `node_tw_evaluation`, `queue_evaluation`, 少样本学习功能

---

## 12. 项目规模

- Python 文件：113 个
- 项目大小：72 MB
- 支持 PIDS 系统：8 种
- 支持数据集：13 个变体
- GNN 编码器：8+ 种
- 特征化方法：11 种
- 训练目标：10+ 种
- 评估方法：5+ 种

---

## 13. 参考文献

```
Bilot et al. (2026). "PIDSMaker: Building and Evaluating Provenance-based
Intrusion Detection Systems." arXiv:2601.22983

Bilot et al. (2025). "Sometimes Simpler is Better: A Comprehensive Analysis
of State-of-the-Art Provenance-Based Intrusion Detection Systems."
USENIX Security 2025
```
