"""
模型评估脚本
在测试集上评估训练好的RoBERTa模型
不使用sklearn，避免pyarrow依赖问题
"""
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging

# 延迟导入transformers，避免sklearn/pyarrow问题
import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

# 纯Python实现评估指标，避免sklearn/pyarrow依赖
def accuracy_score(y_true, y_pred):
    """计算准确率"""
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred, num_classes=3):
    """计算混淆矩阵"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def precision_score(y_true, y_pred, average='macro', num_classes=3):
    """计算精确率"""
    cm = confusion_matrix(y_true, y_pred, num_classes)
    if average == 'macro':
        precisions = []
        for i in range(num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            if tp + fp == 0:
                precisions.append(0.0)
            else:
                precisions.append(tp / (tp + fp))
        return np.mean(precisions)
    elif average is None:
        precisions = []
        for i in range(num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            if tp + fp == 0:
                precisions.append(0.0)
            else:
                precisions.append(tp / (tp + fp))
        return np.array(precisions)

def recall_score(y_true, y_pred, average='macro', num_classes=3):
    """计算召回率"""
    cm = confusion_matrix(y_true, y_pred, num_classes)
    if average == 'macro':
        recalls = []
        for i in range(num_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            if tp + fn == 0:
                recalls.append(0.0)
            else:
                recalls.append(tp / (tp + fn))
        return np.mean(recalls)
    elif average is None:
        recalls = []
        for i in range(num_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            if tp + fn == 0:
                recalls.append(0.0)
            else:
                recalls.append(tp / (tp + fn))
        return np.array(recalls)

def f1_score(y_true, y_pred, average='macro', num_classes=3):
    """计算F1分数"""
    precision = precision_score(y_true, y_pred, average=None, num_classes=num_classes)
    recall = recall_score(y_true, y_pred, average=None, num_classes=num_classes)
    if average == 'macro':
        f1s = []
        for p, r in zip(precision, recall):
            if p + r == 0:
                f1s.append(0.0)
            else:
                f1s.append(2 * p * r / (p + r))
        return np.mean(f1s)
    elif average is None:
        f1s = []
        for p, r in zip(precision, recall):
            if p + r == 0:
                f1s.append(0.0)
            else:
                f1s.append(2 * p * r / (p + r))
        return np.array(f1s)

def classification_report(y_true, y_pred, target_names=None):
    """生成分类报告"""
    num_classes = len(np.unique(y_true))
    if target_names is None:
        target_names = [f'class_{i}' for i in range(num_classes)]
    
    cm = confusion_matrix(y_true, y_pred, num_classes)
    precision = precision_score(y_true, y_pred, average=None, num_classes=num_classes)
    recall = recall_score(y_true, y_pred, average=None, num_classes=num_classes)
    f1 = f1_score(y_true, y_pred, average=None, num_classes=num_classes)
    
    report = "              precision    recall  f1-score   support\n\n"
    total_support = 0
    
    for i, name in enumerate(target_names):
        support = cm[i, :].sum()
        total_support += support
        report += f"{name:>14} {precision[i]:>9.2f} {recall[i]:>9.2f} {f1[i]:>9.2f} {support:>10}\n"
    
    macro_avg_precision = np.mean(precision)
    macro_avg_recall = np.mean(recall)
    macro_avg_f1 = np.mean(f1)
    
    report += "\n    macro avg"
    report += f" {macro_avg_precision:>9.2f} {macro_avg_recall:>9.2f} {macro_avg_f1:>9.2f} {total_support:>10}\n"
    
    return report

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextClassificationDataset(Dataset):
    """文本分类数据集"""
    def __init__(self, data_path, tokenizer, max_length=512):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']

        # 分词
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'id': item['id']
        }

def evaluate_model(
    model_path='models/roberta_classifier',
    test_data_path='data/processed/test.json',
    batch_size=32,
    max_length=512,
    output_dir='results'
):
    """在测试集上评估模型"""

    logger.info("开始评估...")
    logger.info(f"模型路径: {model_path}")
    logger.info(f"测试数据: {test_data_path}")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 延迟导入transformers，在这里导入
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # 加载tokenizer和模型
    logger.info("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # 创建测试数据集
    logger.info("加载测试数据...")
    test_dataset = TextClassificationDataset(test_data_path, tokenizer, max_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    logger.info(f"测试集大小: {len(test_dataset)}")

    # 预测
    logger.info("开始预测...")
    all_preds = []
    all_labels = []
    all_ids = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # 获取预测结果
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_ids.extend(batch['id'])
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 计算评估指标
    logger.info("\n计算评估指标...")

    # Macro F1 (官方评估指标)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    # 其他指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')

    # 每个类别的指标
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    precision_per_class = precision_score(all_labels, all_preds, average=None)
    recall_per_class = recall_score(all_labels, all_preds, average=None)

    # 打印结果
    logger.info("\n" + "="*50)
    logger.info("测试集评估结果")
    logger.info("="*50)
    logger.info(f"Macro F1-Score (官方指标): {macro_f1:.4f}")
    logger.info(f"准确率 (Accuracy): {accuracy:.4f}")
    logger.info(f"Macro Precision: {precision_macro:.4f}")
    logger.info(f"Macro Recall: {recall_macro:.4f}")
    logger.info("\n各类别指标:")
    logger.info(f"  HWT (人类写作) - F1: {f1_per_class[0]:.4f}, Precision: {precision_per_class[0]:.4f}, Recall: {recall_per_class[0]:.4f}")
    logger.info(f"  LGT (模型生成) - F1: {f1_per_class[1]:.4f}, Precision: {precision_per_class[1]:.4f}, Recall: {recall_per_class[1]:.4f}")
    logger.info(f"  HLT (模型增强) - F1: {f1_per_class[2]:.4f}, Precision: {precision_per_class[2]:.4f}, Recall: {recall_per_class[2]:.4f}")

    # 详细分类报告
    label_names = ['HWT (人类写作)', 'LGT (模型生成)', 'HLT (模型增强)']
    logger.info("\n详细分类报告:")
    logger.info("\n" + classification_report(all_labels, all_preds, target_names=label_names))

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    logger.info("\n混淆矩阵:")
    logger.info(f"{'':>15} {'HWT':>10} {'LGT':>10} {'HLT':>10}")
    for i, label in enumerate(label_names):
        logger.info(f"{label:>15} {cm[i][0]:>10} {cm[i][1]:>10} {cm[i][2]:>10}")

    # 保存结果
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存评估指标
    metrics = {
        'macro_f1': float(macro_f1),
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_per_class': {
            'HWT': float(f1_per_class[0]),
            'LGT': float(f1_per_class[1]),
            'HLT': float(f1_per_class[2])
        },
        'precision_per_class': {
            'HWT': float(precision_per_class[0]),
            'LGT': float(precision_per_class[1]),
            'HLT': float(precision_per_class[2])
        },
        'recall_per_class': {
            'HWT': float(recall_per_class[0]),
            'LGT': float(recall_per_class[1]),
            'HLT': float(recall_per_class[2])
        }
    }

    with open(output_dir / 'test_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 保存预测结果
    predictions = []
    for i in range(len(all_ids)):
        predictions.append({
            'id': all_ids[i],
            'true_label': int(all_labels[i]),
            'predicted_label': int(all_preds[i]),
            'probabilities': {
                'HWT': float(all_probs[i][0]),
                'LGT': float(all_probs[i][1]),
                'HLT': float(all_probs[i][2])
            }
        })

    with open(output_dir / 'predictions.json', 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    # 保存混淆矩阵
    cm_dict = {
        'confusion_matrix': cm.tolist(),
        'labels': label_names
    }
    with open(output_dir / 'confusion_matrix.json', 'w', encoding='utf-8') as f:
        json.dump(cm_dict, f, ensure_ascii=False, indent=2)

    logger.info(f"\n结果已保存到 {output_dir}")
    logger.info("="*50)

    return metrics

def main():
    # 评估模型
    metrics = evaluate_model(
        model_path='models/roberta_classifier',
        test_data_path='data/processed/test.json',
        batch_size=32,
        max_length=512,
        output_dir='results'
    )

if __name__ == '__main__':
    main()
