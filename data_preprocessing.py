"""
数据预处理脚本
将训练数据划分为训练集、验证集和测试集（6:2:2）
"""
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_data(data_path):
    """加载原始数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def convert_to_classification_format(data):
    """
    将数据转换为分类格式
    每个样本包含3个文本（HWT, LGT, HLT），转换为3个独立样本
    标签：0=HWT（人类写作）, 1=LGT（模型生成）, 2=HLT（模型增强）
    """
    samples = []

    for item in data:
        sample_id = item['ID']

        # HWT样本 - 标签0
        samples.append({
            'id': f"{sample_id}_HWT",
            'text': item['HWT'],
            'label': 0
        })

        # LGT样本 - 标签1
        samples.append({
            'id': f"{sample_id}_LGT",
            'text': item['LGT'],
            'label': 1
        })

        # HLT样本 - 标签2
        samples.append({
            'id': f"{sample_id}_HLT",
            'text': item['HLT'],
            'label': 2
        })

    return samples

def save_split_data(train_samples, val_samples, test_samples, output_dir):
    """保存划分后的数据"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存训练集
    with open(output_dir / 'train.json', 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)

    # 保存验证集
    with open(output_dir / 'val.json', 'w', encoding='utf-8') as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)

    # 保存测试集
    with open(output_dir / 'test.json', 'w', encoding='utf-8') as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=2)

    print(f"数据划分完成！")
    print(f"训练集样本数: {len(train_samples)}")
    print(f"验证集样本数: {len(val_samples)}")
    print(f"测试集样本数: {len(test_samples)}")
    print(f"总样本数: {len(train_samples) + len(val_samples) + len(test_samples)}")

    # 打印标签分布
    for name, samples in [('训练集', train_samples), ('验证集', val_samples), ('测试集', test_samples)]:
        label_counts = {0: 0, 1: 0, 2: 0}
        for s in samples:
            label_counts[s['label']] += 1
        print(f"\n{name}标签分布:")
        print(f"  HWT (人类写作): {label_counts[0]}")
        print(f"  LGT (模型生成): {label_counts[1]}")
        print(f"  HLT (模型增强): {label_counts[2]}")

def split_raw_data(raw_data,
                   train_ratio=0.6,
                   random_seed=42):

    train_raw, temp_raw = train_test_split(
        raw_data,
        train_size=train_ratio,
        random_state=random_seed
    )

    val_raw, test_raw = train_test_split(
        temp_raw,
        train_size=0.5,
        random_state=random_seed
    )

    return train_raw, val_raw, test_raw


def main():
    # 数据路径
    data_path = 'data/train_data.json'
    output_dir = 'data/processed'

    print("开始加载数据...")
    raw_data = load_data(data_path)
    print(f"原始数据包含 {len(raw_data)} 对样本")

    print("\n划分数据集（6:2:2）...")
    train_raw, val_raw, test_raw = split_raw_data(raw_data)

    print("\n转换数据格式...")
    train_samples = convert_to_classification_format(train_raw)
    val_samples = convert_to_classification_format(val_raw)
    test_samples = convert_to_classification_format(test_raw)
    print(f"转换后共 {len(train_samples) + len(val_samples) + len(test_samples)} 个样本")

    print("\n保存划分后的数据...")
    save_split_data(train_samples, val_samples, test_samples, output_dir)

if __name__ == '__main__':
    main()
