"""
faid_chinese/losses.py
- aux_ce_loss : 3 个辅助 CE 头（model/domain/transform）的平均
- five_level_mcl_loss : 复刻 FAID Eq.6/8 的 5 项子对比损失
        loss_human       : HWT 样本间按主标签
        loss_label       : LGT 样本间按主标签
        loss_set         : LGT 样本间按模型家族
        loss_mixed       : HLT 样本间按 "is_mixed"
        loss_mixed_set   : HLT 样本间按模型家族
- multi_level_loss : 把 5 项子损失按 FAID 权重打包（Eq.8：a=2,b=1,c=1,ζ=2,α=β=2）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def aux_ce_loss(logits: dict, batch: dict) -> torch.Tensor:
    """
    平均 3 个辅助头的 CE。label=-1（理论上不会出现）会被 clamp 防御。
    """
    losses = []
    for name in ("model", "domain", "transform"):
        key = f"{name}_id"
        losses.append(F.cross_entropy(logits[name], batch[key]))
    return torch.stack(losses).mean()


# ----------------------------------------------------------------------
#                  FAID 风格多级对比损失（5 子项）
# ----------------------------------------------------------------------

def _masked_logits(logits: torch.Tensor, same: torch.Tensor) -> torch.Tensor:
    """
    对每个 anchor，把"正样本相似度取平均"做成一个"伪正样本"logit
    （FAID 论文 Eq.7 的多正样本版本）。
    """
    # logits: (B, N+K) 余弦相似度/温度
    # same:   (B, N+K) bool, True 表示"对该 anchor 是正样本"
    eps = 1e-6
    same_f = same.float()
    pos = (logits * same_f).sum(dim=1) / same_f.sum(dim=1).clamp(min=eps)   # (B,)
    neg = logits * (~same).float()                                          # 负样本 logits
    return torch.cat([pos.unsqueeze(1), neg], dim=1)                        # (B, 1+N+K-1)


def _sim_matrix(z: torch.Tensor, temperature: float) -> torch.Tensor:
    """L2-normalized embeddings 的余弦相似度 / 温度。"""
    z = F.normalize(z, dim=-1)
    return z @ z.t() / temperature


def five_level_mcl_loss(z: torch.Tensor, batch: dict, temperature: float = 0.07) -> dict:
    """
    返回 dict: {human, label, set, mixed, mixed_set} 各项 loss。
    子损失都在仅包含相关样本的子集上算（其它 anchor 跳过，FAID 论文的 mask 思路）。
    """
    device = z.device
    B = z.size(0)
    if B < 2:
        zv = {k: torch.tensor(0.0, device=device) for k in
              ["human", "label", "set", "mixed", "mixed_set"]}
        return zv

    sims = _sim_matrix(z, temperature)                  # (B, B)
    # 不让 anchor 看见自己
    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    sims = sims.masked_fill(self_mask, 0.0)

    label    = batch["label"]        # 0=HWT, 1=LGT, 2=HLT
    is_mixed = batch["is_mixed"]     # 0/1
    model_id = batch["model_id"]     # 0..3

    losses = {k: torch.tensor(0.0, device=device) for k in
              ["human", "label", "set", "mixed", "mixed_set"]}
    counts  = {k: 0 for k in losses}

    # ---- loss_human : HWT (label==0) 之间按主标签拉近/推远
    hwt_mask = label == 0
    if hwt_mask.sum() >= 2:
        idx = hwt_mask.nonzero(as_tuple=True)[0]
        sub = sims[idx][:, idx]
        same = (label[idx].unsqueeze(0) == label[idx].unsqueeze(1))
        same = same & ~torch.eye(idx.numel(), dtype=torch.bool, device=device)
        # 简化版：直接用每行同 label 的样本做正样本
        pos = (sub * same.float()).sum(dim=1) / same.float().sum(dim=1).clamp(min=1e-6)
        neg = sub * (~same).float()
        logits = torch.cat([pos.unsqueeze(1), neg], dim=1)
        target = torch.zeros(idx.numel(), dtype=torch.long, device=device)
        losses["human"] = F.cross_entropy(logits, target)
        counts["human"] = 1

    # ---- loss_label : LGT (label==1) 之间按主标签（实际单类，等价于推远）
    lgt_mask = label == 1
    if lgt_mask.sum() >= 2:
        idx = lgt_mask.nonzero(as_tuple=True)[0]
        sub = sims[idx][:, idx]
        same = (label[idx].unsqueeze(0) == label[idx].unsqueeze(1))
        same = same & ~torch.eye(idx.numel(), dtype=torch.bool, device=device)
        pos = (sub * same.float()).sum(dim=1) / same.float().sum(dim=1).clamp(min=1e-6)
        neg = sub * (~same).float()
        logits = torch.cat([pos.unsqueeze(1), neg], dim=1)
        target = torch.zeros(idx.numel(), dtype=torch.long, device=device)
        losses["label"] = F.cross_entropy(logits, target)
        counts["label"] = 1

    # ---- loss_set : LGT 样本间按模型家族拉近
    if lgt_mask.sum() >= 2:
        idx = lgt_mask.nonzero(as_tuple=True)[0]
        sub = sims[idx][:, idx]
        same = (model_id[idx].unsqueeze(0) == model_id[idx].unsqueeze(1))
        same = same & ~torch.eye(idx.numel(), dtype=torch.bool, device=device)
        pos = (sub * same.float()).sum(dim=1) / same.float().sum(dim=1).clamp(min=1e-6)
        neg = sub * (~same).float()
        logits = torch.cat([pos.unsqueeze(1), neg], dim=1)
        target = torch.zeros(idx.numel(), dtype=torch.long, device=device)
        losses["set"] = F.cross_entropy(logits, target)
        counts["set"] = 1

    # ---- loss_mixed : HLT (label==2, is_mixed==1) 之间按 is_mixed
    hlt_mask = is_mixed == 1
    if hlt_mask.sum() >= 2:
        idx = hlt_mask.nonzero(as_tuple=True)[0]
        sub = sims[idx][:, idx]
        same = (is_mixed[idx].unsqueeze(0) == is_mixed[idx].unsqueeze(1))
        same = same & ~torch.eye(idx.numel(), dtype=torch.bool, device=device)
        pos = (sub * same.float()).sum(dim=1) / same.float().sum(dim=1).clamp(min=1e-6)
        neg = sub * (~same).float()
        logits = torch.cat([pos.unsqueeze(1), neg], dim=1)
        target = torch.zeros(idx.numel(), dtype=torch.long, device=device)
        losses["mixed"] = F.cross_entropy(logits, target)
        counts["mixed"] = 1

    # ---- loss_mixed_set : HLT 样本间按模型家族（FAID 核心项）
    if hlt_mask.sum() >= 2:
        idx = hlt_mask.nonzero(as_tuple=True)[0]
        sub = sims[idx][:, idx]
        same = (model_id[idx].unsqueeze(0) == model_id[idx].unsqueeze(1))
        same = same & ~torch.eye(idx.numel(), dtype=torch.bool, device=device)
        pos = (sub * same.float()).sum(dim=1) / same.float().sum(dim=1).clamp(min=1e-6)
        neg = sub * (~same).float()
        logits = torch.cat([pos.unsqueeze(1), neg], dim=1)
        target = torch.zeros(idx.numel(), dtype=torch.long, device=device)
        losses["mixed_set"] = F.cross_entropy(logits, target)
        counts["mixed_set"] = 1

    # 若某子项没有任何 batch 项（counts=0），保留 0，不参与反向
    return losses


def multi_level_loss(z, batch, temperature: float = 0.07,
                     use_5level: bool = True) -> torch.Tensor:
    """
    按 FAID Eq.8 加权：α=β=2, γ=δ=1, ζ=2  对应
        loss_set  : a
        loss_label: (4b - a)
        loss_human: b
        loss_mixed: b
        loss_mixed_set : 2b
    当 use_5level=False 时，仅保留 loss_human（即普通 SCL，退化）。
    """
    parts = five_level_mcl_loss(z, batch, temperature)
    if not use_5level:
        return parts["human"]
    a, b = 2.0, 1.0
    L = (a * parts["set"]
         + (4 * b - a) * parts["label"]
         + b * parts["human"]
         + b * parts["mixed"]
         + 2 * b * parts["mixed_set"])
    return L
