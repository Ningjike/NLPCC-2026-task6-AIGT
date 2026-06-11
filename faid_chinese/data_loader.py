"""
faid_chinese/data_loader.py
ID 解析：{Model}-{Domain}-{Transform}-ID-{N}_{HWT|LGT|HLT}
生成 FAID 风格的 index 三元组 (label, is_mixed, model_id)
"""
import json
import re
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


_ID_RE = re.compile(
    r"^(?P<model>[A-Za-z0-9]+)"
    r"-(?P<domain>[A-Za-z0-9]+)"
    r"-(?P<transform>[A-Za-z0-9]+)"
    r"-ID-(?P<nid>\d+)"
    r"_(?P<suffix>HWT|LGT|HLT)$"
)

# label → style_level (序数): HWT=2(纯人类), LGT=0(纯AI), HLT=1(混合)
STYLE_LEVEL = {0: 2, 1: 0, 2: 1}


def parse_id(item_id: str) -> Dict[str, str]:
    """
    解析用户 ID 格式。失败时返回带 -1 的字典，让上层过滤/抛错。
    """
    m = _ID_RE.match(item_id)
    if not m:
        return {"model": "UNK", "domain": "UNK", "transform": "UNK",
                "nid": "-1", "suffix": "UNK", "ok": False}
    return {**m.groupdict(), "ok": True}


def build_index(item: dict, cfg: dict) -> dict:
    """
    把一条原始样本转成 FAID 风格：
      - label: 0=HWT, 1=LGT, 2=HLT （主任务标签）
      - aux_label: 1=human, 0=AI  （FAID §4.3 二分类辅助头用）
      - is_mixed: 1 if label==2 else 0
      - model_id / domain_id / transform_id
      - family: 0/1/2/3 (4 家族), 优先用新数据 (v2) 自带字段, 缺失时用 model_id
      - style_level: 0=LGT, 1=HLT, 2=HWT 序数, 优先用新数据自带, 缺失时反推
    凡是 ID 解析失败或 model 不在 model_map 中, model_id 置 -1, 让 collate 时直接丢弃。
    label 为 None（无标签测试集，如 testp1.json）时置 -1，不影响推理。
    """
    parsed = parse_id(item["id"])
    raw_label = item.get("label")
    if raw_label is None:
        label = -1
    else:
        label = int(raw_label)
    aux_label = 1 if label == 0 else 0          # 1=human, 0=AI
    is_mixed = 1 if label == 2 else 0

    model_id = cfg["model_map"].get(parsed["model"], -1)
    domain_id = cfg["domain_map"].get(parsed["domain"], -1)
    transform_id = cfg["transform_map"].get(parsed["transform"], -1)

    # family / style_level：新数据 v2 自带, 旧数据 v1 退化反推
    family = item.get("family", model_id)
    if label in STYLE_LEVEL:
        style_level = item.get("style_level", STYLE_LEVEL[label])
    else:
        style_level = -1                        # 无标签测试集

    return {
        "id":          item["id"],
        "text":        item["text"],
        "label":       label,
        "aux_label":   aux_label,
        "is_mixed":    is_mixed,
        "model_id":    model_id,
        "domain_id":   domain_id,
        "transform_id":transform_id,
        "family":      family,
        "style_level": style_level,
    }


def load_jsonl_or_json(path: str) -> List[dict]:
    """
    兼容 JSON list 和 JSONL（虽然本目录数据都是 list 形式 JSON，但保持宽容）。
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class FaidChineseDataset(Dataset):
    """
    输出 dict：
      input_ids, attention_mask : tokenizer 产物
      label           : int   (0/1/2)
      aux_label       : int   (0=AI, 1=human)
      is_mixed        : int   (0/1)
      model_id        : int
      domain_id       : int
      transform_id    : int
      item_id         : str
    """
    def __init__(self, path: str, tokenizer, cfg: dict,
                 max_length: Optional[int] = None,
                 filter_unparsed: bool = True,
                 is_test: bool = False):
        """
        is_test=True 时：
          - 不过滤 ID 解析失败的行（testp1 这种 ID 不符合常规模式）
          - 容忍 label=None
        """
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.max_length = max_length or cfg["max_length"]
        self.is_test = is_test
        raw = load_jsonl_or_json(path)
        items = [build_index(r, cfg) for r in raw]
        if filter_unparsed and not is_test:
            before = len(items)
            items = [it for it in items
                     if it["model_id"] >= 0 and it["domain_id"] >= 0
                     and it["transform_id"] >= 0
                     and it["label"] >= 0]
            dropped = before - len(items)
            if dropped:
                print(f"[FaidChineseDataset] {path}: dropped {dropped} unparsed rows")
        self.items = items
        print(f"[FaidChineseDataset] {path}: {len(self.items)} samples"
              + (" (test mode, label=-1 allowed)" if is_test else ""))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        enc = self.tokenizer(
            it["text"],
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids":     enc["input_ids"],
            "attention_mask":enc["attention_mask"],
            "label":         it["label"],
            "aux_label":     it["aux_label"],
            "is_mixed":      it["is_mixed"],
            "model_id":      it["model_id"],
            "domain_id":     it["domain_id"],
            "transform_id":  it["transform_id"],
            "family":        it["family"],
            "style_level":   it["style_level"],
            "item_id":       it["id"],
        }


def collate_fn_factory(pad_token_id: int):
    """
    返回一个 collate_fn：把不定长 input_ids 补到 batch 内最大长度。
    其他 int 字段直接 stack。
    """
    def collate(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attn = [], []
        for x in batch:
            ids = x["input_ids"]
            am = x["attention_mask"]
            pad = max_len - len(ids)
            input_ids.append(ids + [pad_token_id] * pad)
            attn.append(am + [0] * pad)

        out = {
            "input_ids":     torch.tensor(input_ids, dtype=torch.long),
            "attention_mask":torch.tensor(attn,      dtype=torch.long),
            "label":         torch.tensor([x["label"]         for x in batch], dtype=torch.long),
            "aux_label":     torch.tensor([x["aux_label"]     for x in batch], dtype=torch.long),
            "is_mixed":      torch.tensor([x["is_mixed"]      for x in batch], dtype=torch.long),
            "model_id":      torch.tensor([x["model_id"]      for x in batch], dtype=torch.long),
            "domain_id":     torch.tensor([x["domain_id"]     for x in batch], dtype=torch.long),
            "transform_id":  torch.tensor([x["transform_id"]  for x in batch], dtype=torch.long),
            "family":        torch.tensor([x["family"]        for x in batch], dtype=torch.long),
            "style_level":   torch.tensor([x["style_level"]   for x in batch], dtype=torch.long),
            "item_id":       [x["item_id"] for x in batch],
        }
        return out
    return collate
