"""
faid_chinese/data_pipeline.py  (v2)

从 data/train_data.json 出发，做：
  Step 1  清洗（6 维：长度/极端比/ROUGE/中文占比/noise/语言）
  Step 2  展开为分类格式 (三元组 → 3 条样本)
  Step 3  80/20 随机分层 (按 label 分层, 不留任何 OOD)
  Step 4  训练集类内严格 1:1:1 平衡

设计哲学（与 v1 的关键差异）：
  - v1: 留 3 个 axis-isolated OOD test, 损失 50% Thesis + 50% Polish + 100% Baichuan 训练数据
  - v2: **什么都不留**——4 家族 + 2 域 + 2 变换全在 train
  - 真实 OOD 测试 = leaderboard, 不再自欺

输出格式（每条样本）:
  {
    "id":           "...",   # 与 v1 相同: Model-Domain-Transform-ID-N_Suffix
    "text":         "...",
    "label":        0,        # 0=HWT, 1=LGT, 2=HLT
    "family":       0,        # 0=GPT4, 1=Qwen, 2=ChatGLM, 3=Baichuan
    "style_level":  0,        # 0=LGT(纯AI), 1=HLT(混合), 2=HWT(纯人类)
    "is_mixed":     0         # 0/1, 衍生自 label
  }

输出到 data/faid_v2_processed/{train.json, val.json}
"""
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

# -------------------- 配置 --------------------
random.seed(42)

INPUT_PATH = Path('data/train_data.json')
OUTPUT_DIR = Path('data/faid_v2_processed')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.8

# 清洗阈值（与 v1 完全一致，保留 6 维清洗）
MIN_CHARS = 30
MAX_CHARS = 4000
MIN_HLT_HWT_RATIO = 0.25
MAX_LGT_HWT_RATIO = 3.0
MAX_ROUGE_HWT_LGT = 0.85
MIN_CHINESE_RATIO = 0.5

LABEL_SUFFIX = {'HWT': 0, 'LGT': 1, 'HLT': 2}

# style_level 映射
STYLE_LEVEL = {0: 2, 1: 0, 2: 1}
# label=0(HWT)  → style_level=2
# label=1(LGT)  → style_level=0
# label=2(HLT)  → style_level=1

MODEL_MAP = {'GPT4': 0, 'Qwen': 1, 'ChatGLM': 2, 'Baichuan': 3}

# -------------------- 工具 --------------------

def is_pure_english(text: str) -> bool:
    if not text:
        return True
    has_alpha = any(c.isalpha() for c in text)
    has_cjk   = any('一' <= c <= '鿿' for c in text)
    return has_alpha and not has_cjk

def chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for c in text if '一' <= c <= '鿿')
    return cjk / max(len(text), 1)

def char_bigrams(text: str) -> set:
    return set(text[i:i+2] for i in range(len(text) - 1)) if len(text) >= 2 else set()

def rouge2(hyp: str, ref: str) -> float:
    a, b = char_bigrams(hyp), char_bigrams(ref)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    p = inter / len(a); r = inter / len(b)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)

def parse_id(s: str):
    parts = s.split('-')
    if len(parts) < 4 or 'ID' not in parts:
        return None
    return parts[0], parts[1], parts[2]   # model, domain, transform


# -------------------- Step 1: 清洗 --------------------

def clean_triplet(rec: dict) -> tuple[bool, list[str]]:
    hwt, hlt, lgt = rec.get('HWT', ''), rec.get('HLT', ''), rec.get('LGT', '')
    reasons = []

    if not hwt or not hlt or not lgt:
        return False, ['empty_text']

    if is_pure_english(hlt) and not is_pure_english(hwt) and not is_pure_english(lgt):
        return False, ['pure_english_hlt']

    if lgt.endswith('123'):
        trimmed = lgt[:-3]
        if trimmed.startswith(hwt) or hwt.startswith(trimmed) or lgt[:len(hwt)] == hwt:
            return False, ['123_ending_repeat']

    for name, txt in [('HWT', hwt), ('HLT', hlt), ('LGT', lgt)]:
        if len(txt) < MIN_CHARS:
            return False, [f'len_too_short_{name}']
        if len(txt) > MAX_CHARS:
            return False, [f'len_too_long_{name}']

    if len(hlt) / len(hwt) < MIN_HLT_HWT_RATIO:
        return False, ['extreme_hlt']
    if len(lgt) / len(hwt) > MAX_LGT_HWT_RATIO:
        return False, ['expand_lgt']

    if rouge2(hwt, lgt) > MAX_ROUGE_HWT_LGT:
        return False, ['copy_lgt']

    for name, txt in [('HWT', hwt), ('HLT', hlt), ('LGT', lgt)]:
        if chinese_ratio(txt) < MIN_CHINESE_RATIO:
            return False, [f'low_chinese_{name}']

    return True, []


# -------------------- Step 2: 展开 + 类别 + 富标签 --------------------

def explode_to_samples(clean_records: list[dict]) -> list[dict]:
    """
    三元组 → 3 条分类样本, 富标签格式:
      {id, text, label, family, style_level, is_mixed}
    """
    out = []
    for r in clean_records:
        parsed = parse_id(r['ID'])
        if parsed is None:
            continue
        model, _domain, _transform = parsed
        family = MODEL_MAP.get(model, -1)
        if family < 0:
            continue

        for suf, lab in LABEL_SUFFIX.items():
            text = r[suf]
            out.append({
                'id':          f"{r['ID']}_{suf}",
                'text':        text,
                'label':       lab,
                'family':      family,
                'style_level': STYLE_LEVEL[lab],   # 0/1/2 序数
                'is_mixed':    1 if lab == 2 else 0,
            })
    return out


# -------------------- Step 3: 80/20 随机分层（按 base ID 切，不泄三元组）+ 类内平衡 --------------------

def split_by_base_id(records: list[dict], train_ratio: float):
    """
    关键: 先按 base ID 切分三元组, 再 explode。
    否则同一个 base ID 的 3 个版本会分散在 train/val, 造成泄露。
    """
    base_ids = list(range(len(records)))   # 每条 record 是独立 base ID
    random.shuffle(base_ids)
    sp = int(len(base_ids) * train_ratio)
    train_ids = set(base_ids[:sp])
    train_records = [records[i] for i in range(len(records)) if i in train_ids]
    val_records   = [records[i] for i in range(len(records)) if i not in train_ids]
    return train_records, val_records


def balance_classes(samples: list[dict]):
    """类内严格 1:1:1 平衡（取每类最小尺寸）。"""
    cnt = Counter(s['label'] for s in samples)
    if not cnt:
        return []
    min_c = min(cnt.values())
    by_lab = defaultdict(list)
    for s in samples:
        by_lab[s['label']].append(s)
    out = []
    for lab in (0, 1, 2):
        random.shuffle(by_lab[lab])
        out.extend(by_lab[lab][:min_c])
    random.shuffle(out)
    return out


# -------------------- 主流程 --------------------

def main():
    print("=" * 60)
    print("FAID-chinese 数据生成 v2 (full-data, no OOD holding)")
    print("=" * 60)

    # ---------- 读 raw ----------
    print(f"\n[1/4] 读 {INPUT_PATH} ...")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    print(f"      raw 三元组: {len(raw)}")

    # ---------- 清洗 ----------
    print(f"\n[2/4] 6 维清洗 ...")
    stats = Counter()
    clean = []
    for r in raw:
        ok, reasons = clean_triplet(r)
        if ok:
            clean.append(r)
        else:
            for rsn in reasons:
                stats[rsn] += 1
    print(f"      清洗后: {len(clean)} / {len(raw)} (丢弃 {len(raw)-len(clean)})")
    for k, v in stats.most_common(8):
        print(f"        {k}: {v}")

    # ---------- 展开 + 富标签 ----------
    print(f"\n[3/4] 展开 + 富标签 ...")
    samples = explode_to_samples(clean)
    print(f"      展开后: {len(samples)} 条分类样本")
    print(f"      label 分布: {dict(sorted(Counter(s['label'] for s in samples).items()))}")
    print(f"      family 分布: {dict(sorted(Counter(s['family'] for s in samples).items()))}")
    print(f"      style_level 分布: {dict(sorted(Counter(s['style_level'] for s in samples).items()))}")

    # ---------- 按 base ID 80/20 切 + explode + 类内平衡 ----------
    print(f"\n[4/4] 按 base ID 切 80/20 + explode + 类内 1:1:1 平衡 ...")
    train_records, val_records = split_by_base_id(clean, TRAIN_RATIO)
    print(f"      base ID: train={len(train_records)}, val={len(val_records)}")

    train_samples = explode_to_samples(train_records)
    val_samples   = explode_to_samples(val_records)
    print(f"      explode 后: train={len(train_samples)}, val={len(val_samples)}")

    train = balance_classes(train_samples)
    val   = balance_classes(val_samples)
    print(f"      类内平衡后: train={len(train)}, val={len(val)}")

    # ---------- 写盘 ----------
    print(f"\n[写盘] {OUTPUT_DIR}/ ...")
    for name, data in [('train.json', train), ('val.json', val)]:
        p = OUTPUT_DIR / name
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"      {name}: {len(data)}")

    # ---------- 健全性报告 ----------
    print(f"\n[健全性] 训练集 (family, class) 分布:")
    cnt = Counter((s['family'], s['label']) for s in train)
    fam_name = {0: 'GPT4', 1: 'Qwen', 2: 'ChatGLM', 3: 'Baichuan'}
    for (f, l), c in sorted(cnt.items()):
        print(f"        family={fam_name[f]:8s} class={l}: {c}")
    print(f"\n[健全性] 训练集 (style_level) 分布:")
    cnt = Counter(s['style_level'] for s in train)
    for sl, c in sorted(cnt.items()):
        names = {0: 'LGT(纯AI)', 1: 'HLT(混合)', 2: 'HWT(纯人类)'}
        print(f"        style_level={sl} ({names[sl]}): {c}")

    # ---------- 数据泄露检查 ----------
    print(f"\n[数据泄露] 检查 train / val base ID 重叠 ...")
    train_base = set(s['id'].rsplit('_', 1)[0] for s in train)
    val_base   = set(s['id'].rsplit('_', 1)[0] for s in val)
    overlap = train_base & val_base
    print(f"        train base: {len(train_base)}, val base: {len(val_base)}, "
          f"overlap: {len(overlap)} {'OK' if not overlap else 'LEAK!'}")

    # 标签一致性 sanity check
    print(f"\n[Sanity] label → style_level 一致性 ...")
    for s in train[:5]:
        print(f"        id={s['id'][:30]:30s} label={s['label']} "
              f"family={s['family']} style_level={s['style_level']} is_mixed={s['is_mixed']}")

    print("\n" + "=" * 60)
    print("Done. 改 config.py 路径到 data/faid_v2_processed/ 即可")
    print("=" * 60)


if __name__ == '__main__':
    main()
