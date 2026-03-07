下面给出 ACE-HGNN 两个 agent 协作更新曲率与训练 HGNN 的完整逻辑，按照实现代码的角度整理。重点是每一步谁在做决策、什么时候更新曲率、什么时候更新模型。

一、整体思想

ACE-HGNN 做两件事：

ACE-Agent
探索新的曲率（提出候选曲率）

HGNN-Agent
决定是否采用这个曲率并训练模型

两者通过 任务性能变化作为奖励进行强化学习。

核心流程：

ACE-Agent 提出新曲率
        ↓
HGNN-Agent 决定是否接受
        ↓
HGNN 用该曲率训练
        ↓
计算任务性能
        ↓
两个 agent 根据奖励更新策略
二、两个 Agent 的定义
1 ACE-Agent

作用：

探索新的曲率
元素	含义
状态	当前各层曲率
动作	提出新的曲率
奖励	任务性能变化
2 HGNN-Agent

作用：

决定是否使用 ACE-Agent 提议的曲率
元素	含义
状态	当前曲率
动作	接受 / 拒绝 新曲率
奖励	任务性能变化
三、系统初始化

初始化：

curvature = [-1 for _ in range(L)]   # 每层一个曲率
embedding = init_embedding()

ACE_Q = init_Q_table()
HGNN_Q = init_Q_table()

其中

L = HGNN层数
四、每一轮训练流程

每个 epoch 的逻辑如下。

Step1 获取当前状态

状态是当前曲率：

state = curvature

例如：

state = (-0.8 , -1.2)
Step2 ACE-Agent 提议新曲率

ACE-Agent 根据当前状态选择动作：

action_ACE = policy_ACE(state)

然后计算候选曲率：

kappa = estimate_curvature(graph, embedding)

new_curvature = (1-gamma)*curvature + sqrt(-gamma*kappa)

得到

candidate_curvature
Step3 HGNN-Agent 决定是否接受

HGNN-Agent 根据状态选择动作：

action_HGNN = policy_HGNN(state)

动作空间：

0 = 拒绝新曲率
1 = 接受新曲率

如果接受：

curvature = candidate_curvature

否则：

curvature 保持不变
Step4 映射 embedding 到新曲率空间

如果曲率改变，需要重新映射 embedding：

embedding = exp_map(
        log_map(embedding, old_curvature),
        new_curvature
)
Step5 HGNN 前向传播

用当前曲率训练模型：

embedding = HGNN(graph, embedding, curvature)
Step6 计算任务性能

例如：

metric = evaluate(embedding)

例如：

node classification → F1
link prediction → AUC
Step7 计算奖励

奖励定义：

reward = metric_new - metric_old

如果性能提高：

reward > 0

否则：

reward < 0
Step8 更新两个 agent 的 Q 函数

ACE-Agent：

Q_ACE[state, action_ACE] += alpha * (
        reward + beta * max(Q_ACE[next_state])
        - Q_ACE[state, action_ACE]
)

HGNN-Agent：

Q_HGNN[state, action_HGNN] += alpha * (
        reward + beta * max(Q_HGNN[next_state])
        - Q_HGNN[state, action_HGNN]
)
Step9 更新状态

新的状态：

state = curvature

进入下一轮训练。

五、完整伪代码

实现时可以按这个结构写：

initialize curvature
initialize embedding

for epoch in range(E):

    state = curvature

    # ACE-Agent propose curvature
    action_ACE = epsilon_greedy(Q_ACE, state)
    kappa = estimate_curvature(G, embedding)
    candidate_curvature = (1-gamma)*curvature + sqrt(-gamma*kappa)

    # HGNN-Agent decide accept or reject
    action_HGNN = epsilon_greedy(Q_HGNN, state)

    if action_HGNN == ACCEPT:
        curvature = candidate_curvature

    # map embedding
    embedding = exp_map(log_map(embedding))

    # train HGNN
    embedding = HGNN(G, embedding, curvature)

    metric_new = evaluate(embedding)

    reward = metric_new - metric_old

    # update Q
    update_Q(ACE)
    update_Q(HGNN)

    metric_old = metric_new
六、训练收敛条件

当策略稳定时：

ACE-Agent 不再改变曲率
HGNN-Agent 不再拒绝

系统达到：

Nash equilibrium

此时曲率固定，继续正常训练 HGNN。

七、整个系统本质

如果把 RL 去掉，本质其实是：

自动搜索最优曲率
+
训练 HGNN

可以理解为：

Hyperparameter search + GNN training

只不过作者用 强化学习来做搜索。