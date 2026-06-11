"""
faid_chinese/model.py
- Erlangshen encoder + mean/CLS pool
- 主分类头 (3 类)
- 3 个辅助 CE 头 (model_id 4-way, domain_id 2-way, transform_id 2-way)
- Projection head (SupCon 用)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """注意力 mask 加权的 mean pooling。mask: (B, L) 1=有效。"""
    mask = attention_mask.unsqueeze(-1).float()              # (B, L, 1)
    summed = (last_hidden * mask).sum(dim=1)                # (B, H)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


class SupConProjectionHead(nn.Module):
    """FAID/SupCon 风格投影头：Linear → ReLU → Dropout → Linear → L2-norm。"""
    def __init__(self, in_dim, out_dim=128, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        z = self.drop(self.relu(self.fc1(x)))
        z = self.fc2(z)
        return F.normalize(z, dim=-1)


class FaidChineseModel(nn.Module):
    """
    多任务共享 encoder：
      - pool: mean | cls
      - 4 个 CE 头：main(3), model(4), domain(2), transform(2)
      - 1 个 projection head：给对比损失用
    """
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.encoder = AutoModel.from_pretrained(
            cfg["model_name"], trust_remote_code=True,
        )
        # 取真实 hidden_size（覆盖 cfg 默认值以防不同模型差异）
        real_hidden = self.encoder.config.hidden_size
        cfg["hidden_size"] = real_hidden

        self.dropout = nn.Dropout(cfg.get("dropout", 0.1))
        self.pooling = cfg.get("pooling", "mean")

        self.head_main      = nn.Linear(real_hidden, cfg["num_labels"])
        self.head_model     = nn.Linear(real_hidden, cfg["num_models"])
        self.head_domain    = nn.Linear(real_hidden, cfg["num_domains"])
        self.head_transform = nn.Linear(real_hidden, cfg["num_transforms"])

        self.projection = SupConProjectionHead(
            in_dim=real_hidden, out_dim=cfg.get("projection_dim", 128),
            dropout=cfg.get("dropout", 0.1),
        )

    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state                  # (B, L, H)
        if self.pooling == "cls":
            pooled = last[:, 0, :]
        else:
            pooled = mean_pool(last, attention_mask)
        return pooled                                 # (B, H)

    def forward(self, input_ids, attention_mask, return_projection: bool = False):
        pooled = self.encode(input_ids, attention_mask)
        h = self.dropout(pooled)
        logits = {
            "main":      self.head_main(h),
            "model":     self.head_model(h),
            "domain":    self.head_domain(h),
            "transform": self.head_transform(h),
        }
        if return_projection or self.training:
            # 训练时返回投影用于对比损失
            z = self.projection(pooled)
            return pooled, logits, z
        return pooled, logits, None
