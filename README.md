# NLPCC 2026 Task 6 AIGT 文本分类模型评估结果

## 一、RoBERTa 基础模型评估结果

| 指标 | 值 |
|------|------|
| **Macro F1-Score（官方指标）** | **0.7973** |
| 准确率 (Accuracy) | 0.8072 |
| Macro Precision | 0.8294 |
| Macro Recall | 0.8072 |

### 各类别指标

| 类别 | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| HWT（人类写作） | 0.9328 | 0.5554 | 0.6963 |
| LGT（模型生成） | 0.7422 | 0.9348 | 0.8274 |
| HLT（模型增强） | 0.8133 | 0.9312 | 0.8683 |

### 混淆矩阵

|  | 预测 HWT | 预测 LGT | 预测 HLT |
|--|----------|----------|----------|
| 真实 HWT | 6543 | 3408 | 1829 |
| 真实 LGT | 78 | 11012 | 690 |
| 真实 HLT | 393 | 417 | 10970 |

### 结果分析

- **HWT（人类写作）**：精确率高（0.93）但召回率低（0.56），模型倾向于将人类写作误分类为模型生成文本
- **LGT（模型生成）**：召回率高（0.93）但精确率较低（0.74），模型对模型生成文本的识别能力较强
- **HLT（模型增强）**：整体表现最好，F1 达 0.87

---

## 二、二郎神（Erlangshen-Roberta-330M-NLI）评估结果

**硬件**：NVIDIA A100-SXM4-40GB

### 2.1 训练过程

基于 IDEA-CCNL/Erlangshen-Roberta-330M-NLI 全参数微调模型，训练日志如下（每 200 step 评估一次）：

| Step | Training Loss | Validation Loss | Macro F1 | Accuracy | F1 Hwt   | F1 Lgt   | F1 Hlt   |
| ---- | ------------- | --------------- | -------- | -------- | -------- | -------- | -------- |
| 200  | 0.645975      | 0.362068        | 0.859262 | 0.860538 | 0.813043 | 0.923775 | 0.840967 |
| 400  | 0.353940      | 0.259122        | 0.903381 | 0.904337 | 0.857025 | 0.967075 | 0.886043 |
| 600  | 0.313254      | 0.166020        | 0.942853 | 0.943044 | 0.918490 | 0.983727 | 0.926343 |
| 800  | 0.231873      | 0.174284        | 0.951595 | 0.951787 | 0.934810 | 0.982302 | 0.937674 |
| 1000 | 0.211947      | 0.177161        | 0.945270 | 0.945421 | 0.922918 | 0.981275 | 0.931617 |
| 1200 | 0.246468      | 0.126683        | 0.960974 | 0.961124 | 0.948638 | 0.989766 | 0.944517 |
| 1400 | 0.201232      | 0.137536        | 0.959121 | 0.959256 | 0.942009 | 0.986167 | 0.949188 |
| 1600 | 0.193684      | 0.191140        | 0.943788 | 0.944232 | 0.919067 | 0.969570 | 0.942729 |
| 1800 | 0.207765      | 0.156423        | 0.951786 | 0.951787 | 0.928785 | 0.995796 | 0.930777 |
| 2200 | 0.165780      | 0.130368        | 0.965464 | 0.965538 | 0.951220 | 0.991023 | 0.954150 |
| 2400 | 0.136052      | 0.150469        | 0.958406 | 0.958408 | 0.940178 | 0.995035 | 0.940005 |
| 2600 | 0.143164      | 0.128558        | 0.965049 | 0.965113 | 0.950691 | 0.993789 | 0.950668 |
| 2800 | 0.111717      | 0.148486        | 0.966387 | 0.966471 | 0.952779 | 0.990399 | 0.955983 |
| 3000 | 0.104260      | 0.140991        | 0.970576 | 0.970631 | 0.958709 | 0.994675 | 0.958344 |
| 3200 | 0.086649      | 0.157480        | 0.967956 | 0.967999 | 0.954724 | 0.993783 | 0.955362 |
| 3400 | 0.099264      | 0.182851        | 0.960283 | 0.960445 | 0.943508 | 0.985153 | 0.952187 |
| 3600 | 0.126673      | 0.139463        | 0.968467 | 0.968509 | 0.955629 | 0.993529 | 0.956243 |
| 3685 | 0.116600      | 0.137070        | 0.970248 | 0.970291 | 0.958070 | 0.993912 | 0.958763 |

> 训练到约 3000 step 后验证指标基本饱和，3685 step 时 in-dist 验证 Macro-F1 = 0.9702。

### 2.2 测试集评估结果

| 指标 | 值 |
|------|------|
| **Macro F1-Score（官方指标）** | **0.9709** |
| 准确率 (Accuracy) | 0.9709 |
| Macro Precision | 0.9714 |
| Macro Recall | 0.9709 |

### 2.3 各类别指标

| 类别 | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| HWT（人类写作） | 0.9401 | 0.9789 | 0.9591 |
| LGT（模型生成） | 0.9934 | 0.9972 | 0.9953 |
| HLT（模型增强） | 0.9808 | 0.9366 | 0.9582 |

### 2.4 混淆矩阵

|  | 预测 HWT | 预测 LGT | 预测 HLT |
|--|----------|----------|----------|
| 真实 HWT | 3844 | 11 | 72 |
| 真实 LGT | 11 | 3916 | 0 |
| 真实 HLT | 234 | 15 | 3678 |

### 2.5 结果分析

- **LGT（模型生成）**：表现最佳，F1 达 0.995，召回率 0.997，几乎完美识别模型生成文本
- **HWT（人类写作）** 和 **HLT（模型增强）**：F1 均约 0.96，表现均衡
- 模型整体准确率达 0.97，是三个模型中表现最好的

### 2.6 Codabench 官方评估（in-dist 验证 vs 真实 OOD testp1）

| 提交 | 验证集 Macro-F1 | **Codabench 官方 F1** | 差距 |
|------|----------------|----------------------|------|
| 原始训练数据 | 0.9709 | **0.4063** | -0.56 |
| 6 维清洗数据 | 0.9709 | **0.4320** | -0.54 |

> 内部 in-dist 验证接近完美（0.97），但 Codabench OOD 测试集骤降至 0.43——**典型的分布漂移 / OOD 泛化崩塌**。清洗数据仅带来 0.026 的提升，与 in-dist 表现完全脱节。这是后续 SCL / Late Fusion / FAID / Binoculars Cascade 等一系列工作试图解决的核心问题。

---

## 三、二郎神 + Supervised Contrastive Learning

> **结论先行**：SCL 没能解决 OOD 问题。验证集 Macro-F1 ≈ 0.69，**Codabench 仅 0.3768，反而比全参微调基线 0.4320 更差**。SCL 把同类样本在特征空间拉近的做法在 in-dist 看似有道理，但面对 testp1 的真实多轴 OOD 完全失效。

### 3.1 损失形式

```
L_total = λ · L_SupCon + (1 − λ) · L_CE
```

L_SupCon 为有监督对比损失（同 label 拉近 / 异 label 推远），L_CE 为主任务 3 分类交叉熵。

### 3.2 λ 调参（epoch = 1，验证 Macro-F1）

| λ      | 0.00 | 0.05 | 0.10 | 0.20 | 0.50 |
|--------|------|------|------|------|------|
| Macro F1 | 0.5553 | **0.6402** | 0.6260 | 0.6241 | 0.6112 |

> 选 λ = 0.05。

### 3.3 选定 λ = 0.05，epoch = 5 训练

| 指标 | 值 |
|------|---|
| 验证集 Macro-F1 | **0.6890** |
| **Codabench 官方 F1** | **0.3768** |

in-dist 验证比全参微调（0.97）低 0.28，但 Codabench 也比基线（0.4320）低 0.06——**两个数据集表现一致地变差**。SCL 学到的特征空间对齐对分布外数据没有任何帮助。

---

## 四、Late Fusion 模型评估结果

> **结论先行**：Late Fusion 同样没能解决 OOD 问题。in-dist 验证最高 0.9526（semantic + ppl），**Codabench 仅 0.3075**，是迄今所有方案里最差的。统计特征（ppl / 突发性 / 重复率）在 in-dist 上对 AI 文本的"风格印记"似乎有判别力，但放到 testp1 上完全没用——很可能 OOD 文本的风格特征分布已经偏移。

基于语义特征（semantic）与统计特征的Late Fusion融合方法，在测试集上的结果如下：

### 4.1 特征定义

| 特征名 | 说明 |
|--------|------|
| **semantic** | 语义特征，来自 BERT/RoBERTa 的 [CLS] 向量经投影层后的 128 维表示 |
| **ppl**（Perplexity） | 困惑度，衡量文本流畅度，AI 生成文本通常 ppl 较低 |
| **freq_burstiness** | 词频突发性，衡量高频词的使用集中程度，AI 生成文本倾向于重复使用相同词汇 |
| **sentence_burstiness** | 句子突发性，衡量句子长度变化的均匀程度，AI 生成文本句子长度变化较小 |
| **repetition** | 重复率，文本中 n-gram 重复的比例，AI 生成文本重复率较高 |
| **lexical_diversity** | 词汇多样性（unique tokens / total tokens），AI 生成文本词汇多样性较低 |

### 4.2 特征组合对比

| 特征组合 | Test Loss | Test Acc | Test Macro-F1 |
|----------|----------|----------|---------------|
| semantic + ppl | 0.1896 | 0.9527 | **0.9526** |
| semantic + freq_burstiness | 0.2144 | 0.9447 | 0.9444 |
| semantic + sentence_burstiness | 0.2242 | 0.9423 | 0.9419 |
| semantic + lexical_diversity | 0.2419 | 0.9408 | 0.9404 |
| semantic + repetition | 0.2080 | 0.9368 | 0.9364 |
| **全部6维特征融合** | **0.2116** | **0.9337** | **0.9331** |

> 注：全部6维特征 = semantic + ppl + freq_burstiness + sentence_burstiness + repetition + lexical_diversity

### 4.3 各类别详细结果

#### semantic + ppl

| 类别 | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| HWT | 0.94 | 0.93 | 0.93 |
| HLT | 0.98 | 0.99 | 0.98 |
| LGT | 0.95 | 0.93 | 0.94 |

**混淆矩阵：**

|  | 预测 HWT | 预测 LGT | 预测 HLT |
|--|----------|----------|----------|
| 真实 HWT | 3662 | 78 | 187 |
| 真实 LGT | 8 | 3894 | 25 |
| 真实 HLT | 242 | 17 | 3668 |

#### semantic + freq_burstiness

| 类别 | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| HWT | 0.94 | 0.90 | 0.92 |
| HLT | 0.96 | 1.00 | 0.98 |
| LGT | 0.93 | 0.94 | 0.94 |

**混淆矩阵：**

|  | 预测 HWT | 预测 LGT | 预测 HLT |
|--|----------|----------|----------|
| 真实 HWT | 3533 | 137 | 257 |
| 真实 LGT | 3 | 3909 | 15 |
| 真实 HLT | 210 | 29 | 3688 |

#### semantic + sentence_burstiness

| 类别 | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| HWT | 0.94 | 0.90 | 0.92 |
| HLT | 0.95 | 0.99 | 0.97 |
| LGT | 0.94 | 0.93 | 0.94 |

**混淆矩阵：**

|  | 预测 HWT | 预测 LGT | 预测 HLT |
|--|----------|----------|----------|
| 真实 HWT | 3546 | 174 | 207 |
| 真实 LGT | 4 | 3904 | 19 |
| 真实 HLT | 231 | 45 | 3651 |

#### semantic + lexical_diversity

| 类别 | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| HWT | 0.94 | 0.90 | 0.92 |
| HLT | 0.94 | 0.99 | 0.96 |
| LGT | 0.95 | 0.93 | 0.94 |

**混淆矩阵：**

|  | 预测 HWT | 预测 LGT | 预测 HLT |
|--|----------|----------|----------|
| 真实 HWT | 3530 | 230 | 167 |
| 真实 LGT | 0 | 3901 | 26 |
| 真实 HLT | 235 | 39 | 3653 |

#### semantic + repetition

| 类别 | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| HWT | 0.92 | 0.91 | 0.92 |
| HLT | 0.93 | 0.99 | 0.96 |
| LGT | 0.97 | 0.91 | 0.94 |

**混淆矩阵：**

|  | 预测 HWT | 预测 LGT | 预测 HLT |
|--|----------|----------|----------|
| 真实 HWT | 3573 | 246 | 108 |
| 真实 LGT | 6 | 3901 | 20 |
| 真实 HLT | 302 | 63 | 3562 |

#### 全部6维特征融合

| 类别 | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| HWT | 0.94 | 0.87 | 0.91 |
| HLT | 0.93 | 0.99 | 0.96 |
| LGT | 0.93 | 0.93 | 0.93 |

**混淆矩阵：**

|  | 预测 HWT | 预测 LGT | 预测 HLT |
|--|----------|----------|----------|
| 真实 HWT | 3430 | 238 | 259 |
| 真实 LGT | 0 | 3904 | 23 |
| 真实 HLT | 218 | 43 | 3666 |

### 4.4 semantic + ppl + SCL

Supervised Contrastive Learning 通过将同类样本在特征空间中拉近、不同类样本推远，增强特征的判别性。

**模型结构：**
- 语义分支：BERT → `[CLS]` → `semantic_proj`（Linear(768→128) → ReLU → Dropout）
- 统计特征分支：PPL → `aux_proj`（Linear(1→16, BatchNorm → ReLU) → Linear(16→32, BatchNorm → ReLU)）
- 融合层：`fusion_classifier`（Linear(160→64) → ReLU → Dropout → Linear(64→3)）
- SupCon 投影头：`projection_head`（Linear(160→256) → ReLU → Dropout → Linear(256→128)），用于计算对比损失

**损失函数：**
```
Total Loss = λ × L_SupCon + (1 - λ) × L_CE
```
其中 λ = 0.5，Temperature = 0.07，Projection Dim = 128。

**效果：** scl有效提升了 HLT（模型增强）类别的区分度，使 HLT 的 F1 从 0.96 进一步提升至 0.99。

| 指标 | 值 |
|------|------|
| **Test Macro-F1** | **0.9453** |
| Test Loss | 0.1793 |
| Test Acc | 0.9454 |

**分类报告：**

| 类别 | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| HWT | 0.92 | 0.93 | 0.92 |
| HLT | 0.98 | 0.99 | 0.99 |
| LGT | 0.93 | 0.92 | 0.93 |

**混淆矩阵：**

|  | 预测 HWT | 预测 LGT | 预测 HLT |
|--|----------|----------|----------|
| 真实 HWT | 3642 | 37 | 248 |
| 真实 LGT | 26 | 3890 | 11 |
| 真实 HLT | 288 | 33 | 3606 |

### 4.5 Codabench 官方评估

| 方案 | 验证集 Macro-F1 | **Codabench F1** |
|------|----------------|------------------|
| semantic + ppl | 0.9526 | **0.3075** |
| semantic + 6 维全特征 | 0.9331 | （未单独提交） |

> Late Fusion 拿到 in-dist 最高的 0.95+ 之一，但 Codabench 跌到 **0.3075**——比全参微调基线 0.4320 还低 0.12。统计特征在 OOD 上不仅没帮助，反而成了噪声。

---

## 五、FAID 适配（多任务 + 多级对比 + Fuzzy k-NN）

> **结论先行**：FAID 是本次重点尝试的方案，但同样**没能解决 OOD 问题**。两轮调参下来 Codabench F1 始终在 0.37–0.42 之间，与"清洗数据 + 全参微调"基线 0.4320 持平或更低。算法在自己留出的 axis-isolated OOD 上几乎满分（0.94+），但放到真实多轴叠加漂移的 testp1 上完全失效。
>
> 详细实验记录与设计决策见 [faid_chinese/README.md](faid_chinese/README.md)。

### 5.1 论文与本任务的差异

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

### 5.2 数据处理（data/faid_v2_processed/）

处理脚本：[faid_chinese/data_pipeline.py](faid_chinese/data_pipeline.py)

| Step | 操作 | 输出 |
|------|------|------|
| 1 | 6 维清洗（长度 / 极端比 / ROUGE / 中文占比 / 纯英文 / 123 重复） | 清洗后三元组 |
| 2 | 展开三元组 → 3 条分类样本 | 富标签样本 |
| 3 | 按 base ID 切 80/20（防同一三元组泄露） | train / val |
| 4 | 类内 1:1:1 严格平衡 | 最终 train / val |

**关键设计**：
- 4 家族（GPT4 / Qwen / ChatGLM / Baichuan）+ 2 域（News / Thesis）+ 2 变换（Rewrite / Polish）**全部保留在训练集**，不留任何 axis-isolated OOD
- 真实 OOD 测试 = Codabench leaderboard，**不再自欺**
- 富标签格式：`{id, text, label, family, style_level, is_mixed}`，其中 `style_level` 是为后续序数回归（HLT 应靠近对应来源 LLM）做准备

**清洗阈值**：

| 维度 | 阈值 |
|------|------|
| 最短 / 最长字符数 | 30 / 4000 |
| HLT/HWT 长度比下限 | 0.25 |
| LGT/HWT 长度比上限 | 3.0 |
| ROUGE-2 (HWT, LGT) | ≤ 0.85 |
| 中文字符占比下限 | 0.5 |

**v2 数据规模**（`data/faid_v2_processed/`）：

| 文件 | 样本数 | 说明 |
|------|--------|------|
| train.json | 41,817 | 4 家族 + 2 域 + 2 变换全在 |
| val.json | 10,455 | in-dist 早停用 |

### 5.3 训练配置

- backbone：Erlangshen-Roberta-330M-NLI，max_length = 512
- `lambda_main_ce = 1.0`，`lambda_aux_ce = 0.2`，`lambda_mcl = 0.3`（**关键**：原论文 1.0 会让分类头被压制）
- `temperature = 0.07`，`use_5level_mcl = True`
- 优化器：AdamW + linear warmup (5%) + fp16
- batch_size = 16，10 epoch，~3.5 小时 on 3090
- 推理：`alpha_knn = 0.7`（k-NN 软投票话语权加大），`top_k = 20`

### 5.4 Round 1：FAID 原文配置（lambda_mcl = 1.0）

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

### 5.5 Round 2：调权重（lambda_mcl = 0.3, max_length = 512, alpha_knn = 0.7）

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
>
> **核心问题**：预测分布严重偏 HLT（64%），LGT 仅 4%——k-NN 把"中间/混合"特征都拉成 HLT。3 个 axis-isolated OOD 互相污染，**永远反映不了 testp1 的多轴叠加漂移**。

### 5.6 Round 3：换数据策略（v2 全量 + 富标签）— 待跑

详见 [faid_chinese/README.md](faid_chinese/README.md) §二 Round 3。训练样本从 22k 翻倍到 41.8k，4 家族 + 2 域 + 2 变换全覆盖，预期能缓解"预测偏 HLT"问题。但鉴于 Round 1/2 都没有突破基线，**FAID 这条路大概率走不通**。

### 5.7 FAID 三轮总评

| | in-dist F1 | 内部 OOD F1 | **Codabench F1** |
|------|-----------|-------------|------------------|
| Erlangshen 全参微调（清洗）| 0.9709 | — | **0.4320**（基线）|
| FAID Round 1 | 0.674 | 0.623 | 0.3766 |
| FAID Round 2 | 0.978 | 0.945 | 0.4139 |
| FAID Round 3（待跑）| — | — | — |

**所有 FAID 变体都低于或接近基线 0.4320**。该方案带来的所有复杂度（5 项对比损失、3 个辅助头、Fuzzy k-NN、序数回归）都没能转化为 OOD 泛化的提升——核心瓶颈在 testp1 的分布漂移本身，不在算法。

---

## 六、Binoculars Cascade（Stage 1 零样本 + Stage 2 二分类）

> **结论先行**：流水线已经搭好（stage1 算分 + 阈值调优 + stage2 训练 + 合并），但**未见 Codabench 显著超过 0.43 的提交**。Binoculars 在英文上的零样本能力迁移到中文 + Qwen 家族上不稳定；Stage 2 在 label∈{1,2} 子集上做二分类本身就继承了基线的 OOD 崩塌。详见 [binoculars_cascade/README.md](binoculars_cascade/README.md)。

### 6.1 设计动机

单阶段 Erlangshen 微调 in-dist 0.97、Codabench 0.43，过拟合 in-dist 分布 / OOD 崩塌。本方案让零样本的 Binoculars 接管对分布偏移最敏感的 label 0（HWT）检测，Stage 2 只学更专一的 LGT vs HLT 边界。

### 6.2 流水线

- **Stage 1**：Binoculars 零样本（Qwen2.5-1.5B-Instruct + Qwen2.5-1.5B 双模型对），在 train+val 标注数据上扫阈值 τ
- **Stage 2**：Erlangshen-Roberta-330M-NLI 在 label ∈ {1, 2} 子集上做二分类微调
- **合并**：Stage 1 判 HWT → label 0；其余样本由 Stage 2 决定 label 1 / 2

### 6.3 显存与运行（24GB 3090）

| 阶段 | 显存峰值 | 备注 |
|------|----------|------|
| Stage 1（Binoculars） | ~15GB | batch=4，OOM 时改 2 |
| Stage 2（Erlangshen 330M） | ~12GB | fp16 + gradient_checkpointing |

### 6.4 状态

- ✅ Stage 1 算分（一次性 ~4h，可缓存复用）
- ✅ Stage 2 数据准备 + 训练 + 预测
- ✅ 阈值调优（train+val 联合 Macro-F1）
- ⏳ 合并预测 + Codabench 提交：**未见显著超过 0.43 的结果**

---

## 七、总结与对比

| 方案 | in-dist 验证 | **Codabench F1** | 与基线差距 |
|------|-------------|------------------|-----------|
| RoBERTa 基础模型 | 0.7973 | — | — |
| **Erlangshen 全参微调（清洗数据）** | **0.9709** | **0.4320** | 基线 |
| 二郎神 + SCL | 0.6890 | 0.3768 | -0.055 |
| Late Fusion (semantic + ppl) | 0.9526 | 0.3075 | -0.125 |
| Late Fusion (6 维全特征) | 0.9331 | — | — |
| Late Fusion + SCL | 0.9453 | — | — |
| FAID Round 1 | 0.674 | 0.3766 | -0.055 |
| FAID Round 2 | 0.978 | 0.4139 | -0.018 |
| FAID Round 3 | — | — | — |
| Binoculars Cascade | — | 未显著超过基线 | — |

**关键观察**：
- 全部方案在 in-dist 验证上都能拿到 0.93+，**但 OOD 真实分布上的提升几乎为零**
- 最有 SOTA 感的方案（FAID Round 2 = 0.978 in-dist，0.4139 Codabench）**仍然打不过"清洗数据 + 全参微调"这条最朴素的基线**
- 瓶颈在 testp1 的多轴叠加分布漂移，**算法层面的复杂度提升无法转化**

---

> 后续待做：
> 1. FAID Round 3 训练 + Codabench 提交（看全量数据 + 4 家族可见能否突破 0.43）
> 2. Binoculars Cascade 合并预测 + Codabench 提交
> 3. 序数回归头（基于 `style_level` 字段）实验
> 4. 后处理校准（logit 调整、class-balanced sampling）缓解 HLT 偏置
> 5. 如果所有变体都 < 0.50，转向**多模型 ensemble** + 简单清洗后全参微调
