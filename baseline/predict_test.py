#!/usr/bin/env python
"""
测试集推理脚本
对测试数据进行预测，保存结果为JSON文件
"""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from tqdm import tqdm


class TestDataset(Dataset):
    """测试数据集，只返回模型需要的张量，id和text从原始数据取回"""
    def __init__(self, data_path, tokenizer, max_length=512):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"加载测试集: {len(self.data)} 个样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            max_length=self.max_length,
            return_tensors=None
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }


def predict(model, data_loader, device, test_dataset, batch_size):
    """对测试数据进行预测"""
    model.eval()
    results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Predicting")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()

            # 从原始数据取回id和text
            start = batch_idx * batch_size
            for i, pred in enumerate(preds):
                sample = test_dataset.data[start + i]
                results.append({
                    'id': sample['id'],
                    'text': sample['text'],
                    'label': int(pred)
                })

    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 配置 - 与训练保持一致
    model_path = 'models/full_ft'
    test_data_path = 'data/testp1.json'
    output_path = 'test_predictions.json'
    max_length = 512  # 与训练保持一致
    batch_size = 32

    # 检查模型路径
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")

    # 加载tokenizer和模型
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model = model.to(device)

    # 加载测试数据
    test_dataset = TestDataset(test_data_path, tokenizer, max_length)

    # 使用DataCollatorWithPadding处理变长序列
    data_collator = DataCollatorWithPadding(tokenizer)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=data_collator
    )

    # 推理
    predictions = predict(model, test_loader, device, test_dataset, batch_size)

    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\n预测结果已保存到: {output_path}")
    print(f"总预测样本数: {len(predictions)}")

    # 打印标签分布
    label_counts = {0: 0, 1: 0, 2: 0}
    for p in predictions:
        label_counts[p['label']] += 1
    print(f"\n预测标签分布:")
    print(f"  0 (人工编写): {label_counts[0]}")
    print(f"  1 (大模型生成): {label_counts[1]}")
    print(f"  2 (大模型增强): {label_counts[2]}")


if __name__ == '__main__':
    main()