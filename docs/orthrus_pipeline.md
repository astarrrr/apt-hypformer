# Orthrus 全流程技术文档

## 1. 概览

Orthrus 是 PIDSMaker 框架中的一个溯源图入侵检测模型，采用 **TGN + GraphAttention 双分支编码器** 和 **边类型预测** 作为自监督目标。异常检测原理：模型学习预测正常边的类型，推理时预测误差大的边/节点被判为异常。

### 流水线执行顺序

```
Construction → Transformation → Featurization → Feat_Inference → Batching → Training → Evaluation → Triage
```

---

## 2. Construction：从数据库构建溯源图

**关键文件**：
- `pidsmaker/tasks/construction.py`
- `pidsmaker/preprocessing/build_graph_methods/build_default_graphs.py`

### 处理流程

1. **提取节点元数据**（`compute_indexid2msg`）
   - 查询 PostgreSQL 数据库
   - 提取三类节点属性：
     - **Subject/Process**: type, path, cmd_line
     - **File**: type, path
     - **Netflow**: type, remote_ip, remote_port
   - 创建映射：`{node_id: [node_type, label_string]}`

2. **构建时间窗口图**（`gen_edge_fused_tw`）
   - 按时间窗口（orthrus: **15 秒**）划分事件
   - 每个窗口创建一个有向多重图（NetworkX）
   - 边属性：src, dst, timestamp, event_label, edge_type
   - `fuse_edge: True`：合并同一对节点间的重复边

3. **输出**
   - 按 train/val/test 分割保存 `.pt` 图文件
   - 路径：`cfg.construction._graphs_dir/{train,val,test}/*.pt`

### 配置

```yaml
construction:
  used_method: default
  time_window_size: 15.0        # 15 秒时间窗口
  fuse_edge: True               # 合并重复边
  use_all_files: True
  multi_dataset: none
  node_label_features:
    subject: type, path, cmd_line
    file: type, path
    netflow: type, remote_ip, remote_port
```

---

## 3. Transformation：图变换

**关键文件**：`pidsmaker/tasks/transformation.py`

Orthrus 配置为 `used_methods: none`，直接将 Construction 的图原样复制到下一阶段，不做任何变换。

---

## 4. Featurization：训练节点嵌入模型

**关键文件**：
- `pidsmaker/tasks/featurization.py`
- `pidsmaker/featurization/featurization_methods/featurization_word2vec.py`

### Word2Vec 训练流程

1. **语料生成**（`get_corpus`，`featurization_utils.py:29-56`）
   - 从 `indexid2msg.pkl`（Construction 阶段从数据库提取并保存的字典）读取所有节点的 `{node_id: (node_type, label_string)}`
   - 按 `training_split` 过滤节点（orthrus: `all` = train+val+test）
   - 对每个**唯一的 node_label** 调用 `tokenize_label` 分词：
     - subject: 按路径分隔符拆分 path、cmd_line，如 `"/usr/bin/nginx"` → `["usr", "/", "bin", "/", "nginx"]`
     - file: 按 `/` `:` `.` 拆分路径
     - netflow: 拆分 IP 和端口
   - 每个唯一 label 的分词结果作为一个"句子"
   - **注意**：不涉及图结构，不做随机游走，仅使用节点标签文本

2. **模型训练**
   - 算法：Gensim Word2Vec
   - 参数：
     - 向量维度：**128**
     - 窗口大小：5
     - Skip-gram 模式
     - 负采样：5
     - 训练轮数：50

3. **输出**
   - 保存模型：`cfg.featurization._model_dir/word2vec.model`

### 配置

```yaml
featurization:
  emb_dim: 128
  epochs: 50
  training_split: all           # 用全部数据训练 word2vec
  use_seed: True
  used_method: word2vec
  word2vec:
    alpha: 0.025
    window_size: 5
    min_count: 1
    use_skip_gram: True
    num_workers: 1
    negative: 5
    decline_rate: 30
```

---

## 5. Feat_Inference：生成特征向量

**关键文件**：`pidsmaker/tasks/feat_inference.py`

### 处理流程

对每个图的每条边 `(u, v)`：

```
msg = [src_type_onehot(3D), src_emb(128D), edge_type_onehot(~10D), dst_type_onehot(3D), dst_emb(128D)]
```

### 输出数据结构（CollatableTemporalData）

| 字段 | 形状 | 描述 |
|------|------|------|
| `src` | `[E]` | 源节点索引 |
| `dst` | `[E]` | 目标节点索引 |
| `t` | `[E]` | 时间戳 |
| `msg` | `[E, ~272D]` | 边特征向量 |
| `y` | `[E]` | 边标签（0=正常, 1=攻击） |

---

## 6. Batching：数据批处理

**关键文件**：
- `pidsmaker/tasks/batching.py`
- `pidsmaker/utils/data_utils.py`

### 四阶段批处理

| 阶段 | Orthrus 配置 | 描述 |
|------|-------------|------|
| Global Batching | `none` | 不分组 |
| Intra-Graph Batching | `edges, tgn_last_neighbor` | 每个图内按 **1024 条边** 切分 mini-batch |
| Node Reindexing | 自动 | 将节点索引重映射为连续的 `[0, N)` |
| Inter-Graph Batching | `none` | 不跨图合并 |

### TGN 邻居采样

- 邻居数量：20
- 跳数：1 hop
- 方向：无向

### 最终 Batch 数据格式

| 字段 | 形状 | 描述 |
|------|------|------|
| `edge_index` | `[2, E]` | 边索引 |
| `x` | `[N, in_dim]` | 节点特征（重索引后） |
| `node_type` | `[N]` | 节点类型索引 |
| `msg` | `[E, D]` | 边特征 |
| `edge_type` | `[E, num_edge_types]` | 边类型 one-hot |
| `t` | `[E]` | 时间戳 |
| `y` | `[E]` | 边标签 |

### 配置

```yaml
batching:
  save_on_disk: False
  node_features: node_emb,node_type
  edge_features: edge_type
  intra_graph_batching:
    used_methods: edges, tgn_last_neighbor
    edges:
      intra_graph_batch_size: 1024
    tgn_last_neighbor:
      tgn_neighbor_size: 20
      tgn_neighbor_n_hop: 1
  inter_graph_batching:
    used_method: none
```

---

## 7. Training：双分支编码器训练

**关键文件**：
- `pidsmaker/tasks/training.py`
- `pidsmaker/detection/training_methods/training_loop.py`
- `pidsmaker/model.py`
- `pidsmaker/factory.py`

### 7.1 模型架构

```
Batch
  |
  v
TGN Encoder Branch
  ├─ (Optional) Memory Module       [use_memory=False]
  ├─ (Optional) Time Encoding       [use_time_order_encoding=False]
  ├─ Node Feature Projection         x_src [E, 128] + x_dst [E, 128]
  │   └─ src_linear(x_src) + dst_linear(x_dst) → [N, 128]
  ├─ Edge Features 拼接              edge_type + msg
  └─ Inner GraphAttention GNN
      ├─ Layer 0: TransformerConv(128 → 512, 8 heads, concat=True) + ReLU + Dropout
      └─ Layer 1: TransformerConv(512 → 64, 1 head, concat=False)
  |
  v
h: [N, 64]                          # node_out_dim = 64
  |
  v
gather_h(batch, res)
  h_src = h[edge_index[0]]          # [E, 64]
  h_dst = h[edge_index[1]]          # [E, 64]
  |
  v
EdgeTypePrediction Decoder
  ├─ CustomEdgeMLP
  │   ├─ 拼接: [h_src, h_dst]       → [E, 128]
  │   ├─ Projection (coef=2)        → [E, 256]
  │   ├─ Linear(256 → 128)
  │   ├─ ReLU + Dropout(0.5)
  │   └─ Linear(128 → 10)           → [E, num_edge_types]
  └─ CrossEntropyLoss(logits, edge_type_argmax)
      → scalar loss (training)
      → [E,] per-edge loss (inference)
```

### 7.2 训练循环

```python
for epoch in range(12):
    # 1. 重置 TGN 编码器状态
    model.encoder.reset_state()

    # 2. 训练
    for dataset in train_data:
        for batch in dataset:
            loss = model(batch)               # forward pass
            loss.backward()                   # backward pass
            optimizer.step()                  # update (grad_accumulation=1)

    # 3. 验证/测试推理（每 2 个 epoch）
    for split in [val_data, test_data]:
        for batch in split:
            scores = model(batch, inference=True)  # per-edge anomaly scores

    # 4. Early stopping (patience=3)
    if val_score not improved for 3 epochs:
        break
```

### 7.3 配置

```yaml
training:
  used_method: default
  num_epochs: 12
  patience: 3
  lr: 0.00001
  weight_decay: 0.00001
  node_hid_dim: 128
  node_out_dim: 64
  grad_accumulation: 1
  encoder:
    dropout: 0.5
    used_methods: tgn,graph_attention
    graph_attention:
      activation: relu
      num_heads: 8
      concat: True
      flow: source_to_target
      num_layers: 2
    tgn:
      tgn_memory_dim: 100
      tgn_time_dim: 100
      use_node_feats_in_gnn: True
      use_memory: False
      use_time_order_encoding: False
      project_src_dst: True
  decoder:
    used_methods: predict_edge_type
    predict_edge_type:
      decoder: edge_mlp
      use_triplet_types: False
      edge_mlp:
        architecture_str: linear(0.5) | relu
        src_dst_projection_coef: 2
    use_few_shot: False
```

---

## 8. Evaluation：异常检测评估

**关键文件**：
- `pidsmaker/tasks/evaluation.py`
- `pidsmaker/detection/evaluation_methods/node_evaluation.py`
- `pidsmaker/detection/training_methods/inference_loop.py`

### 8.1 推理阶段

对 val/test 集的每个 batch：
1. 前向传播 `model(batch, inference=True)`
2. 返回每条边的异常分数（CrossEntropy loss）：`[E,]`
3. 保存到 CSV

### 8.2 节点级评估流程

```
每条边的异常分数 [E,]
        |
        v
聚合到节点级
  score[node] = sum(该节点关联的所有边的 loss)
  (use_dst_node_loss=True: 同时累加到 src 和 dst 节点)
        |
        v
阈值选择: max_val_loss
  threshold = max(验证集所有节点的 score)
        |
        v
预测
  score > threshold → 异常节点
  score ≤ threshold → 正常节点
        |
        v
与 Ground Truth 对比 → 计算指标
```

### 8.3 评估指标

| 指标 | 描述 |
|------|------|
| **Precision** | 预测为异常的节点中，真正异常的比例 |
| **Recall** | 真实异常节点中，被正确检测的比例 |
| **F1** | Precision 和 Recall 的调和平均 |
| **ADP** | Average Detection Precision：排名靠前的节点中异常节点的比例 |
| **Discrimination** | 异常节点的分数是否显著高于正常节点 |
| **ROC-AUC** | ROC 曲线下面积 |

### 8.4 K-Means 聚类

- 对节点分数进行 K-Means 聚类
- 取 Top-K（30）个聚类为异常
- 作为辅助检测手段

### 8.5 模型选择

- `best_model_selection: best_adp`
- 选择验证集上 ADP 最高的 epoch 作为最终模型

### 配置

```yaml
evaluation:
  viz_malicious_nodes: False
  ground_truth_version: orthrus
  best_model_selection: best_adp
  used_method: node_evaluation
  node_evaluation:
    threshold_method: max_val_loss
    use_dst_node_loss: True
    use_kmeans: True
    kmeans_top_K: 30
```

---

## 9. 数据流与维度汇总

```
PostgreSQL DB
    │
    ▼
[Construction]  时间窗口=15s, fuse_edge
    │  NetworkX 多重有向图, 节点属性: type+label
    ▼
[Transformation]  none (直接复制)
    │
    ▼
[Featurization]  Word2Vec, 128D, 50 epochs (基于节点标签文本，非图结构)
    │  word2vec.model
    ▼
[Feat_Inference]  节点嵌入 + 类型 one-hot → 边特征
    │  CollatableTemporalData: msg [E, ~272D]
    ▼
[Batching]  edges=1024/batch, TGN neighbor=20
    │  Batch: x [N, 128], edge_index [2, E], msg [E, D]
    ▼
[Training]  TGN + GraphAttention → [N, 64]
    │         → EdgeTypePrediction → CE Loss
    ▼
[Evaluation]  边级 loss → 节点级聚合 → 阈值 → 指标
    │  Precision, Recall, F1, ADP
    ▼
  结果输出
```

### 各阶段张量维度

| 位置 | 形状 | 维度 |
|------|------|------|
| 节点特征 x | `[N, 128]` | emb_dim=128 |
| 节点类型 | `[3]` | subject / file / netflow |
| 边特征 msg | `[E, ~272]` | src_type + src_emb + edge_type + dst_type + dst_emb |
| 边类型 | `[E, 10]` | one-hot |
| 编码器隐层 | `[N, 128]` | node_hid_dim=128 |
| 编码器输出 h | `[N, 64]` | node_out_dim=64 |
| gather_h 输出 | h_src `[E, 64]`, h_dst `[E, 64]` | |
| MLP 拼接 | `[E, 128]` | concat(h_src, h_dst) |
| MLP 投影 | `[E, 256]` | src_dst_projection_coef=2 |
| 解码器输出 | `[E, 10]` | num_edge_types |
| 异常分数 | `[E]` | per-edge CE loss |

---

## 10. 关键文件索引

| 组件 | 路径 |
|------|------|
| 主入口 | `pidsmaker/main.py` |
| 任务模块 | `pidsmaker/tasks/{construction,transformation,featurization,feat_inference,batching,training,evaluation}.py` |
| 配置定义 | `pidsmaker/config/config.py` |
| Orthrus 配置 | `config/orthrus.yml` |
| 默认配置 | `config/default.yml` |
| 模型定义 | `pidsmaker/model.py` |
| 组件工厂 | `pidsmaker/factory.py` |
| TGN 编码器 | `pidsmaker/encoders/tgn_encoder.py` |
| GraphAttention | `pidsmaker/encoders/graph_attention.py` |
| EdgeMLP 解码器 | `pidsmaker/decoders/custom_edge_mlp.py` |
| 边类型预测 | `pidsmaker/objectives/predict_edge_type.py` |
| 训练循环 | `pidsmaker/detection/training_methods/training_loop.py` |
| 推理循环 | `pidsmaker/detection/training_methods/inference_loop.py` |
| 节点评估 | `pidsmaker/detection/evaluation_methods/node_evaluation.py` |
| 数据工具 | `pidsmaker/utils/data_utils.py` |

---

## 11. 运行命令

```bash
cd /home/astar/projects/PIDSMaker
python pidsmaker/main.py orthrus CADETS_E3 --wandb
```
