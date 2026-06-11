# NLPCC 2026 Task 6 AIGT 文本分类模型

  
按6:2:2划分训练集进行初步训练
## 基础模型评估结果

| 指标 | chinese-roberta-wwm-ext | Erlangshen-Roberta-330M-NLI |

|------|------|------|

| **Macro F1-Score（官方指标）** | **0.7973** | **0.9709** |

| 准确率 (Accuracy) | 0.8072 |0.9709 |

| Macro Precision | 0.8294 |0.9714 |

| Macro Recall | 0.8072 |0.9709 |
后续模型选定 Erlangshen

---
## 二郎神（Erlangshen-Roberta-330M-NLI）评估结果

**硬件**：NVIDIA A100-SXM4-40GB
训练过程 `train_data.json` 清洗后 **8 : 2**（train : val）
### 训练过程

| Step | Training Loss | Validation Loss | Macro F1 | Accuracy | F1 Hwt   | F1 Lgt   | F1 Hlt   |

| ---- | ------------- | --------------- | -------- | -------- | -------- | -------- | -------- |

| 200  | 0.645975      | 0.362068        | 0.859262 | 0.860538 | 0.813043 | 0.923775 | 0.840967 |

| 400  | 0.353940      | 0.259122        | 0.903381 | 0.904337 | 0.857025 | 0.967075 | 0.886043 |

| 600  | 0.313254      | 0.166020        | 0.942853 | 0.943044 | 0.918490 | 0.983727 | 0.926343 |

| 800  | 0.231873      | 0.174284        | 0.951595 | 0.951787 | 0.934810 | 0.982302 | 0.937674 |

| 1000 | 0.211947      | 0.177161        | 0.945270 | 0.945421 | 0.922918 | 0.981275 | 0.931617 |

| 1200 | 0.246468      | 0.126683        | 0.960974 | 0.961124 | 0.948638 | 0.989766 | 0.944517 |

| 1400 | 0.201232      | 0.137536        | 0.959121 | 0.959256 | 0.942009 | 0.986167 | 0.949188 |

| 1600 | 0.193684      | 0.191140        | 0.943788 | 0.944232 | 0.919067 | 0.969570 | 0.942729 |

| 1800 | 0.207765      | 0.156423        | 0.951786 | 0.951787 | 0.928785 | 0.995796 | 0.930777 |

| 2200 | 0.165780      | 0.130368        | 0.965464 | 0.965538 | 0.951220 | 0.991023 | 0.954150 |

| 2400 | 0.136052      | 0.150469        | 0.958406 | 0.958408 | 0.940178 | 0.995035 | 0.940005 |

| 2600 | 0.143164      | 0.128558        | 0.965049 | 0.965113 | 0.950691 | 0.993789 | 0.950668 |

| 2800 | 0.111717      | 0.148486        | 0.966387 | 0.966471 | 0.952779 | 0.990399 | 0.955983 |

| 3000 | 0.104260      | 0.140991        | 0.970576 | 0.970631 | 0.958709 | 0.994675 | 0.958344 |

| 3200 | 0.086649      | 0.157480        | 0.967956 | 0.967999 | 0.954724 | 0.993783 | 0.955362 |

| 3400 | 0.099264      | 0.182851        | 0.960283 | 0.960445 | 0.943508 | 0.985153 | 0.952187 |

| 3600 | 0.126673      | 0.139463        | 0.968467 | 0.968509 | 0.955629 | 0.993529 | 0.956243 |

| 3685 | 0.116600      | 0.137070        | 0.970248 | 0.970291 | 0.958070 | 0.993912 | 0.958763 |

-  训练到约 3000 step 后验证指标基本饱和，3685 step 时 val Macro-F1 = 0.9702。
 -  Codabench 评估 testp1.json： **0.4320** 

选型阶段的 0.9709 与 Codabench 的 0.4320 出现 0.54 的巨大差异
测试集来自与训练集不同的模型、领域，而模型泛化能力不够

---
## 二郎神 + Supervised Contrastive Learning

### 损失形式

```

L_total = λ · L_SupCon + (1 − λ) · L_CE

```

### λ 调参
（epoch = 1，val验证 Macro-F1）
| λ      | 0.00 | 0.05 | 0.10 | 0.20 | 0.50 |
|--------|------|------|------|------|------|
| Macro F1 | 0.5553 | **0.6402** | 0.6260 | 0.6241 | 0.6112 |
> 选 λ = 0.05。
### 选定 λ = 0.05，epoch = 5 训练
| 指标 | 值 |

|------|------|

| 验证集 Macro-F1 | **0.6890** |

| **Codabench 官方 F1** | **0.3768** |
验证集F1比全参微调（0.97）低 0.28，Codabench 测试集F1也比基线（0.4320）低 0.06。两个数据集表现一致地变差。说明SCL 学到的特征空间对齐对分布外数据可能没有太大帮助。

---
## Late Fusion 模型

基于语义特征（semantic）与统计特征的 Late Fusion 融合方法
### 4.1 特征定义

| 特征名 | 说明 |

|--------|------|

| **semantic** | 语义特征，来自 BERT/RoBERTa 的 [CLS] 向量经投影层后的 128 维表示 |

| **ppl**（Perplexity） | 困惑度，衡量文本流畅度，AI 生成文本通常 ppl 较低 |

| **freq_burstiness** | 词频突发性，衡量高频词的使用集中程度，AI 生成文本倾向于重复使用相同词汇 |

| **sentence_burstiness** | 句子突发性，衡量句子长度变化的均匀程度，AI 生成文本句子长度变化较小 |

| **repetition** | 重复率，文本中 n-gram 重复的比例，AI 生成文本重复率较高 |

| **lexical_diversity** | 词汇多样性（unique tokens / total tokens），AI 生成文本词汇多样性较低 |

### 特征组合对比（**内部 val 集**，in-dist）

| 特征组合 | Val Loss | Val Acc | Val Macro-F1 |

|----------|----------|---------|--------------|

| semantic + ppl | 0.1896 | 0.9527 | **0.9526** |

| semantic + freq_burstiness | 0.2144 | 0.9447 | 0.9444 |

| semantic + sentence_burstiness | 0.2242 | 0.9423 | 0.9419 |

| semantic + lexical_diversity | 0.2419 | 0.9408 | 0.9404 |

| semantic + repetition | 0.2080 | 0.9368 | 0.9364 |

| **全部6维特征融合** | **0.2116** | **0.9337** | **0.9331** |
> 注：全部6维特征 = semantic + ppl + freq_burstiness + sentence_burstiness + repetition + lexical_diversity

由此认为PPL带来的增益最佳

### Codabench 官方评估

| 方案 | 验证集 Macro-F1 | **Codabench F1** |

|------|----------------|------------------|

| semantic + ppl | 0.9526 | **0.3075** |
Late Fusion 拿到在验证集上达到0.95+ ，但 Codabench 跌到 0.3075。
比全参微调基线 0.4320 低 0.12。
统计特征在 OOD 上没帮助。
Late Fusion 同样没能解决 OOD 问题。**Codabench 仅 0.3075**，是迄今所有方案里最差的。统计特征（ppl / 突发性 / 重复率）在 in-dist 上对 AI 文本的"风格印记"似乎有判别力，但放到 testp1 上完全没用——很可能 OOD 文本的风格特征分布已经偏移。

---
## FAID 适配（多任务 + 多级对比 + Fuzzy k-NN）

参考 FAID 论文 *Fine-Grained AI-Generated Text Detection Using Multi-Task Auxiliary and Multi-Level Contrastive Learning*，把算法核心适配到 NLPCC 2026 Task 6 的中文 AIGT 3 分类任务上。**关键差异：保留 Erlangshen-Roberta-330M-NLI 作为 backbone，不换 XLM-R。**

| 论文组件 | 我们的实现 | 文件 |

|----------|----------|------|

| Encoder | FaidChineseModel.encode（Erlangshen） | [faid_chinese/model.py](faid_chinese/model.py) |

| 3 类主分类头 | head_main | 同上 |

| 辅助多任务头 | head_model / head_domain / head_transform | 同上 |

| 多级对比损失 5 项 | multi_level_loss / five_level_mcl_loss | [faid_chinese/losses.py](faid_chinese/losses.py) |

| SupCon 投影头 | SupConProjectionHead | [faid_chinese/model.py](faid_chinese/model.py) |

| Fuzzy k-NN 推理 | VectorDB + knn_vote | [faid_chinese/infer_faid_chinese.py](faid_chinese/infer_faid_chinese.py) |

| 富标签数据（family + style_level） | v2 新增 | [faid_chinese/data_pipeline.py](faid_chinese/data_pipeline.py) |

5 项对比子损失：HWT 拉同 label、LGT 拉同 label + 同 model 家族、HLT 拉同 label + 同 model 家族（FAID 核心项 loss_mixed_set）。

### 数据处理
处理脚本：[faid_chinese/data_pipeline.py](faid_chinese/data_pipeline.py)
| Step | 操作 | 输出 |

|------|------|------|

| 1 | 6 维清洗（长度 / 极端比 / ROUGE / 中文占比 / 纯英文 / 123 重复） | 清洗后三元组 |

| 2 | 展开三元组 → 3 条分类样本 | 富标签样本 |

| 3 | 按 base ID 切 80/20（防同一三元组泄露） | train / val |

| 4 | 类内 1:1:1 严格平衡 | 最终 train / val |

**关键设计**：

- 4 家族（GPT4 / Qwen / ChatGLM / Baichuan）+ 2 域（News / Thesis）+ 2 变换（Rewrite / Polish）**全部保留在训练集**，不留任何 axis-isolated OOD
- 真实 OOD 测试 = Codabench leaderboard
- 富标签格式：`{id, text, label, family, style_level, is_mixed}`，其中 `style_level` 是为后续序数回归（HLT 应靠近对应来源 LLM）做准备

  

**清洗阈值**：

| 维度 | 阈值 |

|------|------|

| 最短 / 最长字符数 | 30 / 4000 |

| HLT/HWT 长度比下限 | 0.25 |

| LGT/HWT 长度比上限 | 3.0 |

| ROUGE-2 (HWT, LGT) | ≤ 0.85 |

| 中文字符占比下限 | 0.5 |
### 训练配置

- backbone：Erlangshen-Roberta-330M-NLI，max_length = 512
- `lambda_main_ce = 1.0`，`lambda_aux_ce = 0.2`，`lambda_mcl = 0.3`（**关键**：原论文 1.0 会让分类头被压制）
- `temperature = 0.07`，`use_5level_mcl = True`
- 优化器：AdamW + linear warmup (5%) + fp16
- batch_size = 16，10 epoch，~3.5 小时 on 3090
- 推理：`alpha_knn = 0.7`（k-NN 软投票话语权加大），`top_k = 20`
### Round 1：FAID 原文配置（lambda_mcl = 1.0）
| 指标 | 值 |

|------|---|

| 训练 L_main（终） | 0.50 |

| 训练 L_mcl（终） | 4.13 |

| in-dist val F1 | 0.674 |

| OOD model_ood F1 | 0.566 |

| OOD domain_ood F1 | 0.606 |

| OOD transform_ood F1 | 0.647 |

| 4 集平均 | 0.623 |

| **Codabench F1** | **0.3766** |

> `L_mcl` 占总梯度 88%，分类头被压制，LGT F1 仅 0.45。Codabench 0.3766 **低于** Erlangshen 清洗基线 0.4320，**FAID 配置反而更差**。

### Round 2：调权重（lambda_mcl = 0.3, max_length = 512, alpha_knn = 0.7）
**关键改动**：

- `lambda_mcl: 1.0 → 0.3`（让分类头真正学）

- `lambda_aux_ce: 0.5 → 0.2`

- `max_length: 256 → 512`（避免长文本截断）

- `alpha_knn: 0.5 → 0.7`，`knn_temperature: 0.7 → 0.5`
| 指标 | Round 1 | Round 2 |

|------|---------|---------|

| 训练 L_main（终） | 0.50 | 0.50 |

| 训练 L_mcl（终） | 4.13 | 4.13 |

| in-dist val F1 | 0.674 | **0.978** |

| OOD model_ood F1 | 0.566 | **0.890** |

| OOD domain_ood F1 | 0.606 | **0.956** |

| OOD transform_ood F1 | 0.647 | **0.956** |

| 4 集平均 | 0.623 | **0.945** |

| **Codabench F1** | 0.3766 | **0.4139** |

| 预测分布 (HWT/LGT/HLT) | 27/20/54 | 32/4/64 |

> 内部 OOD 暴涨 0.62→0.95，**算法在"自己留的 OOD"上几乎完美**。Codabench 0.4139 只比基线 0.4320 低 0.018，仍**没有提升**。

> **核心问题**：预测分布严重偏 HLT（64%），LGT 仅 4%——k-NN 把"中间/混合"特征都拉成 HLT。3 个 axis-isolated OOD 互相污染，**永远反映不了 testp1 的多轴叠加漂移**。

### Round 3：换数据策略（v2 全量 + 富标签）

**核心动机**：Round 1/2 内部 OOD 0.94+ 但 Codabench 0.38–0.41，**说明在"自己留的 OOD"上调出来的指标，对真实 testp1 几乎没有预测力**。根因是 axis-isolated OOD 互相污染，永远反映不了 testp1 的多轴叠加漂移。所以本轮直接换数据策略：

- 6 维清洗保留
- **不留任何 OOD**：4 家族（GPT4 / Qwen / ChatGLM / Baichuan）+ 2 域（News / Thesis）+ 2 变换（Rewrite / Polish）全部进 train
- 80/20 随机分层（按 base ID 切，杜绝同一三元组泄露）
- 类内 1:1:1 严格平衡
- 输出**富标签格式** `{id, text, label, family, style_level, is_mixed}`，其中 `family` 注入到 5 项对比损失里的 `loss_set` / `loss_mixed_set`，让模型显式学"哪个 LLM 家族生成的"

> **相对 Round 2 的关键变化**：训练样本 22,236 → **41,817**（×1.88），家族数 3 → 4，Thesis 覆盖 50% → 100%，Polish 覆盖 50% → 100%。

#### 训练过程（10 epoch，~3.5h on 3090）

| Epoch | in-dist val F1 | model_ood F1 | domain_ood F1 | transform_ood F1 | 4 集平均 | 用时 (s) |
|-------|---------------|--------------|---------------|------------------|----------|---------|
| 1     | 0.900         | 0.826        | 0.842         | 0.879            | 0.862    | 695 |
| 2     | 0.904         | 0.846        | 0.873         | 0.890            | 0.878    | 699 |
| 3     | 0.974         | 0.852        | 0.939         | 0.942            | 0.927    | 698 |
| 4     | 0.974         | 0.864        | 0.944         | 0.946            | 0.932    | 698 |
| 5     | 0.974         | 0.885        | 0.940         | 0.948            | 0.937    | 697 |
| 6     | 0.977         | 0.874        | 0.949         | 0.951            | 0.938    | 696 |
| 7     | **0.980**     | 0.851        | 0.949         | 0.948            | 0.932    | 697 |
| 8     | 0.978         | 0.890        | 0.956         | 0.956            | 0.945    | 699 |
| 9     | **0.980**     | 0.885        | 0.956         | 0.955            | 0.944    | 697 |
| 10    | 0.979         | **0.890**    | 0.956         | **0.956**        | 0.945    | 700 |

> **观察**：
> - **epoch 2 → 3 出现一次明显跳变**（in-dist 0.904 → 0.974），意味着模型在 epoch 3 突然"开窍"，可能是辅助头与对比头终于和分类头对齐。
> - epoch 5 之后 in-dist / domain_ood / transform_ood 都稳定在 0.95+；**model_ood 是唯一难以稳定的指标**（在 0.85–0.89 之间反复震荡），说明"未见过的 LLM 家族"仍然是最大短板。
> - 选 **epoch 9** 作 best checkpoint（in-dist 0.980、4 集平均 0.944）。

#### Codabench 官方评估

| 指标 | 值 |
|------|---|
| in-dist val F1（epoch 9） | 0.980 |
| 内部 4 集平均 F1 | 0.944 |
| **Codabench F1（testp1）** | **0.4004** |

#### 结论

| | in-dist F1 | 内部 4 集平均 | **Codabench F1** |
|------|-----------|---------------|------------------|
| Erlangshen 清洗基线 | 0.9709 | — | **0.4320**（基线）|
| FAID Round 1 | 0.674 | 0.623 | 0.3766 |
| FAID Round 2 | 0.978 | 0.945 | 0.4139 |
| **FAID Round 3** | **0.980** | **0.944** | **0.4004** |

> 训练样本翻倍 + 4 家族全见 + 全域全覆盖，in-dist 进一步提升到 0.980，**但 Codabench 不升反降（0.4139 → 0.4004）**。
>
> 一个值得注意的信号：**Round 3 的 in-dist 0.980 / 内部 4 集平均 0.944 / Codabench 0.4004 是 3 个差距最大的一组**——in-dist 和内部 OOD 几乎完美，但真实 OOD 暴跌 0.55，再次印证：
>
> 1. axis-isolated OOD（哪怕是"全家族可见"）**不能作为 testp1 难度的代理**；
> 2. **增加训练数据 / 增强模型容量不能突破"单轴分离的 OOD"与"多轴叠加的真实 OOD"之间的本质差距**；
> 3. FAID 论文给出的"作者风格对比 + 多任务辅助"在原作者设定的 in-domain 设置里 work，**但 task-specific 的 OOD 漂移要靠 task-specific 的解法**。
>
> 既然 Round 3 也未突破基线，**FAID 这条路可以判定为走不通**，应将精力转向对 testp1 真实分布更鲁棒的方向（见 §七 后续待做）。


### 5.7 FAID 三轮总评

| | in-dist F1 | 内部 4 集平均 F1 | **Codabench F1（testp1）** |
|------|-----------|------------------|----------------------------|
| Erlangshen 全参微调（清洗基线）| 0.9709 | — | **0.4320** |
| FAID Round 1 | 0.674 | 0.623 | 0.3766 |
| FAID Round 2 | 0.978 | 0.945 | 0.4139 |
| **FAID Round 3** | **0.980** | **0.944** | **0.4004** |

**所有 FAID 变体都低于或接近基线 0.4320**。该方案带来的所有复杂度（5 项对比损失、3 个辅助头、Fuzzy k-NN、序数回归）都没能转化为 OOD 泛化的提升——核心瓶颈在 testp1 的分布漂移本身，不在算法。

> 完整训练日志见 [faid_chinese/models/train_log.jsonl](faid_chinese/models/train_log.jsonl)，逐 epoch 每 phase（val_in / model_ood / domain_ood / transform_ood）的 F1/Acc/per-class 都在里面。

---

## Binoculars Cascade（Stage 1 零样本 + Stage 2 二分类）

- **Stage 1**：Binoculars 零样本（Qwen2.5-1.5B-Instruct + Qwen2.5-1.5B 双模型对），在 train+val 标注数据上扫阈值 τ
- **Stage 2**：Erlangshen-Roberta-330M-NLI 在 label ∈ {1, 2} 子集上做二分类微调
- **合并**：Stage 1 判 HWT → label 0；其余样本由 Stage 2 决定 label 1 / 2

---
