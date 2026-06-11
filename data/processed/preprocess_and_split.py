"""
数据清洗 + 按 ID 8:2 划分脚本
- 数据清洗逻辑与 data/split_train_val.py 保持一致
  （噪声清洗：pure_english_hlt / 123_ending_repeat；
    文本清洗：合并换行、合并多余空格；
    极端样本标记：is_extreme / extreme_reasons；
    扁平分类格式：每条样本拆为 HWT/LGT/HLT 三个独立分类样本）
- 数据划分：仅基于 ID 进行 8:2 随机划分（不区分模型/领域/Transform）
"""

import json
import re
import random
from pathlib import Path
from collections import Counter

# ============================================================
# 配置
# ============================================================
DATA_PATH = 'data/train_data.json'
OUTPUT_DIR = 'data/processed'
TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)


# ============================================================
# 数据清洗（与 data/split_train_val.py 保持一致）
# ============================================================
def clean_text(text):
    """
    清理冗余格式符号，保留自然文本
    - 替换多余换行为空格
    - 去除首尾空白
    - 合并多余空格
    - 保留新闻署名等自然文本
    """
    if not text:
        return text
    text = re.sub(r'\n+', ' ', text)
    text = text.strip()
    text = re.sub(r' {2,}', ' ', text)
    return text


def is_pure_english_text(text):
    if not text:
        return False
    return all(ord(c) < 128 for c in text if c.isalpha())


def detect_noise_type(item):
    hwt = item.get('HWT', '')
    hlt = item.get('HLT', '')
    lgt = item.get('LGT', '')

    hlt_is_english = is_pure_english_text(hlt)
    hwt_is_english = is_pure_english_text(hwt)
    lgt_is_english = is_pure_english_text(lgt)

    if hlt_is_english and not hwt_is_english and not lgt_is_english:
        return 'pure_english_hlt'

    if lgt.endswith('123'):
        lgt_trimmed = lgt[:-3]
        if len(lgt_trimmed) <= len(hwt):
            if hwt.startswith(lgt_trimmed):
                return '123_ending_repeat'
        else:
            if lgt_trimmed[:len(hwt)] == hwt:
                return '123_ending_repeat'

    return None


def filter_noise_samples(data):
    clean_data = []
    noise_stats = {'pure_english_hlt': 0, '123_ending_repeat': 0}

    for item in data:
        noise_type = detect_noise_type(item)
        if noise_type is None:
            clean_data.append(item)
        else:
            noise_stats[noise_type] += 1

    return clean_data, noise_stats


def get_char_bigrams(text):
    if len(text) < 2:
        return set()
    return set(text[i:i+2] for i in range(len(text) - 1))


def rouge_2_chars_f1(ref, hyp):
    ref_bigrams = get_char_bigrams(ref)
    hyp_bigrams = get_char_bigrams(hyp)
    if not ref_bigrams or not hyp_bigrams:
        return 0.0
    overlap = len(ref_bigrams & hyp_bigrams)
    precision = overlap / len(hyp_bigrams)
    recall = overlap / len(ref_bigrams)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_lcs_for_text(hwt, other):
    return rouge_2_chars_f1(hwt, other)


def mark_extreme_samples(data):
    """
    标记极端样本（不删除，仅标记）
    - is_extreme_hlt: len(HLT)/len(HWT) < 0.25
    - is_copy_lgt:    ROUGE(HWT, LGT) > 0.85
    - is_expand_lgt:  len(LGT)/len(HWT) > 3
    """
    for item in data:
        hwt = item['HWT']
        hlt = item['HLT']
        lgt = item['LGT']

        item['is_extreme'] = False
        item['extreme_reasons'] = []

        len_hwt = len(hwt)
        if len_hwt == 0:
            continue

        if len(hlt) / len_hwt < 0.25:
            item['is_extreme'] = True
            item['extreme_reasons'].append('extreme_hlt')

        if len(lgt) / len_hwt > 3:
            item['is_extreme'] = True
            item['extreme_reasons'].append('expand_lgt')

        rouge_lgt = rouge_lcs_for_text(hwt, lgt)
        if rouge_lgt > 0.85:
            item['is_extreme'] = True
            item['extreme_reasons'].append('copy_lgt')


def convert_to_classification_format(data):
    """转换为扁平分类格式（每个三元组分拆为3个独立样本）"""
    samples = []
    for item in data:
        sample_id = item['ID']
        samples.append({'id': f"{sample_id}_HWT", 'text': clean_text(item['HWT']), 'label': 0})
        samples.append({'id': f"{sample_id}_LGT", 'text': clean_text(item['LGT']), 'label': 1})
        samples.append({'id': f"{sample_id}_HLT", 'text': clean_text(item['HLT']), 'label': 2})
    return samples


# ============================================================
# 按 ID 8:2 划分
# ============================================================
def split_by_id(data, train_ratio=0.8, seed=42):
    """
    仅基于 ID 进行随机划分：
    1. 取出所有 ID 并打乱
    2. 按 train_ratio 切分为 train_ids / val_ids
    3. 同 ID 的 HWT/LGT/HLT 三元组必须同属一侧（保证不泄露）
    """
    rng = random.Random(seed)
    ids = [item['ID'] for item in data]
    unique_ids = list(set(ids))
    rng.shuffle(unique_ids)

    split_idx = int(len(unique_ids) * train_ratio)
    train_ids = set(unique_ids[:split_idx])
    val_ids = set(unique_ids[split_idx:])

    train_raw, val_raw = [], []
    for item in data:
        if item['ID'] in train_ids:
            train_raw.append(item)
        else:
            val_raw.append(item)

    return train_raw, val_raw, train_ids, val_ids


def save_data(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def verify_no_data_leakage(train_samples, val_samples):
    train_ids = set(s['id'].rsplit('_', 1)[0] for s in train_samples)
    val_ids = set(s['id'].rsplit('_', 1)[0] for s in val_samples)
    overlap = train_ids & val_ids
    if overlap:
        print(f"    警告: 发现 {len(overlap)} 个ID重叠！")
        return False
    print(f"    通过: 无ID泄露")
    return True


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("数据清洗 + 按 ID 8:2 划分")
    print("=" * 60)

    # Step 1: 加载原始数据
    print(f"\nStep 1: 加载数据 ({DATA_PATH})...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    print(f"    原始数据: {len(raw_data)} 条")

    # Step 2: 噪声清洗
    print("\nStep 2: 噪声清洗...")
    clean_data, noise_stats = filter_noise_samples(raw_data)
    print(f"    pure_english_hlt:    {noise_stats['pure_english_hlt']}")
    print(f"    123_ending_repeat:   {noise_stats['123_ending_repeat']}")
    print(f"    清洗后: {len(clean_data)} 条")

    # Step 3: 标记极端样本
    print("\nStep 3: 标记极端样本...")
    mark_extreme_samples(clean_data)
    extreme_count = sum(1 for i in clean_data if i.get('is_extreme', False))
    print(f"    极端样本标记数: {extreme_count}")
    reasons_count = Counter()
    for item in clean_data:
        for r in item.get('extreme_reasons', []):
            reasons_count[r] += 1
    for r, c in reasons_count.items():
        print(f"    - {r}: {c}")

    # Step 4: 按 ID 8:2 划分
    print(f"\nStep 4: 按 ID {int(TRAIN_RATIO*100)}:{int((1-TRAIN_RATIO)*100)} 划分...")
    train_raw, val_raw, train_ids, val_ids = split_by_id(
        clean_data, train_ratio=TRAIN_RATIO, seed=SEED
    )
    print(f"    唯一 ID 总数: {len(train_ids) + len(val_ids)}")
    print(f"    训练集 ID:   {len(train_ids)}")
    print(f"    验证集 ID:   {len(val_ids)}")
    print(f"    训练集三元组: {len(train_raw)}")
    print(f"    验证集三元组: {len(val_raw)}")

    # Step 5: 保存数据
    print(f"\nStep 5: 保存到 {OUTPUT_DIR}/ ...")
    train_cls = convert_to_classification_format(train_raw)
    val_cls = convert_to_classification_format(val_raw)
    save_data(train_cls, f'{OUTPUT_DIR}/train.json')
    save_data(val_cls, f'{OUTPUT_DIR}/val.json')
    print(f"    train.json: {len(train_cls)} 条 (扁平分类格式)")
    print(f"    val.json:   {len(val_cls)} 条 (扁平分类格式)")

    # Step 6: 数据泄露检查
    print(f"\nStep 6: 数据泄露检查...")
    verify_no_data_leakage(train_cls, val_cls)

    # Step 7: 验证集统计
    print(f"\nStep 7: 分布统计...")
    train_labels = Counter(s['label'] for s in train_cls)
    val_labels = Counter(s['label'] for s in val_cls)
    print(f"    训练集标签: {dict(sorted(train_labels.items()))}")
    print(f"    验证集标签: {dict(sorted(val_labels.items()))}")

    train_extreme = sum(1 for i in train_raw if i.get('is_extreme', False))
    val_extreme = sum(1 for i in val_raw if i.get('is_extreme', False))
    if train_raw:
        print(f"    训练集极端样本: {train_extreme} ({100*train_extreme/len(train_raw):.1f}%)")
    if val_raw:
        print(f"    验证集极端样本: {val_extreme} ({100*val_extreme/len(val_raw):.1f}%)")

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
