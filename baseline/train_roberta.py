import os
os.environ['TORCHDYNAMO_DISABLE'] = '1'
import json
import math
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.metrics import f1_score, accuracy_score, classification_report
import logging
import gc

import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 设置随机种子
set_seed(42)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextClassificationDataset(Dataset):
    """文本分类数据集 - 优化版本，使用动态padding"""
    def __init__(self, data_path, tokenizer, max_length=512):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"加载数据集: {len(self.data)} 个样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']

        # RoBERTa最大位置编码为512，超过会报错
        # 当max_length > 256时使用head+tail策略处理长文本
        # 直接获取完整token序列（不截断），然后手动做head+tail
        if self.max_length > 256:
            # 直接编码，不限制max_length，获取完整序列
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=False
            )

            # 计算可用长度（需要扣除[CLS]和[SEP]）
            usable_len = self.max_length - 2
            head_len = usable_len // 2
            tail_len = usable_len - head_len

            # head+tail: 取前head_len和后tail_len个token
            if len(tokens) > usable_len:
                tokens = tokens[:head_len] + tokens[-tail_len:]

            # 添加特殊token: [CLS] + tokens + [SEP]
            input_ids = [self.tokenizer.cls_token_id] + tokens + [self.tokenizer.sep_token_id]
            attention_mask = [1] * len(input_ids)

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': label
            }
        else:
            # max_length <= 256 时直接使用
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_tensors=None
            )
            return {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'labels': label
            }

def compute_metrics(pred):
    """计算评估指标"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # 计算macro F1
    macro_f1 = f1_score(labels, preds, average='macro')

    # 计算准确率
    accuracy = accuracy_score(labels, preds)

    # 计算每个类别的F1
    f1_per_class = f1_score(labels, preds, average=None)

    return {
        'macro_f1': macro_f1,
        'accuracy': accuracy,
        'f1_hwt': f1_per_class[0],
        'f1_lgt': f1_per_class[1],
        'f1_hlt': f1_per_class[2]
    }

def train_model(
    model_name='hfl/chinese-roberta-wwm-ext',
    train_data_path='data/processed/train.json',
    val_data_path='data/processed/val.json',
    output_dir='models/roberta_classifier',
    max_length=512,
    batch_size=16,
    gradient_accumulation_steps=2,
    num_epochs=10,
    learning_rate=3e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    save_steps=400,
    eval_steps=200,
    logging_steps=50,
    use_gradient_checkpointing=False
):
    """训练RoBERTa分类模型 - 支持长文本head+tail策略"""

    logger.info("="*60)
    logger.info("开始训练")
    logger.info("="*60)
    logger.info(f"模型: {model_name}")
    logger.info(f"训练数据: {train_data_path}")
    logger.info(f"验证数据: {val_data_path}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"梯度累积步数: {gradient_accumulation_steps}")
    logger.info(f"等效Batch Size: {batch_size * gradient_accumulation_steps}")
    logger.info(f"梯度检查点: {use_gradient_checkpointing}")

    # 检查GPU
    if not torch.cuda.is_available():
        raise RuntimeError("需要GPU才能运行此脚本！")

    device = torch.device('cuda')
    logger.info(f"GPU设备: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # 清理显存
    gc.collect()
    torch.cuda.empty_cache()

    # 加载分词器和模型
    logger.info("加载分词器和模型...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
    )

    logger.info("模型加载完成")

    # 启用梯度检查点
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("✓ 启用梯度检查点")

    # 创建数据集
    logger.info("创建数据集...")
    train_dataset = TextClassificationDataset(train_data_path, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_data_path, tokenizer, max_length)

    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")

    # 使用动态padding的DataCollator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    # 计算总训练步数 - 使用math.ceil避免warmup偏小
    steps_per_epoch = math.ceil(len(train_dataset) / (batch_size * gradient_accumulation_steps))
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    logger.info(f"总训练步数: {total_steps}")
    logger.info(f"Warmup步数: {warmup_steps}")

    # 设置训练参数 - 针对24G 3090优化
    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=42,
        data_seed=42,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,  # 评估时可以用更大的batch
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,

        # 日志和保存
        logging_steps=logging_steps,
        logging_first_step=True,

        # 评估策略
        eval_strategy='steps',
        eval_steps=eval_steps,

        # 保存策略 - 优化磁盘占用
        save_strategy='steps',
        save_steps=save_steps,  # 每N步保存一次（必须是eval_steps的整数倍）
        save_total_limit=1,  # 只保留最好的1个checkpoint
        save_only_model=True,  # 只保存模型，不保存优化器状态（节省空间）
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        greater_is_better=True,

        # 优化设置
        fp16=True,  # 启用混合精度训练
        fp16_full_eval=True,  # 评估时也使用FP16
        optim='adamw_torch',  # 使用标准优化器

        # 数据加载
        dataloader_num_workers=4,  # Windows下4比8更稳定
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,

        # 其他优化
        gradient_checkpointing=use_gradient_checkpointing,
        max_grad_norm=1.0,  # 梯度裁剪防止梯度爆炸

        # 报告 - 禁用TensorBoard
        report_to='none',
        remove_unused_columns=False,

        # 分布式训练设置（单卡也建议开启）
        ddp_find_unused_parameters=False,
    )

    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # 3次不提升即停止
    )

    # 显示显存使用情况
    logger.info(f"训练前显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    logger.info(f"训练前显存缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # 开始训练
    logger.info("="*60)
    logger.info("开始训练模型...")
    logger.info("="*60)

    train_result = trainer.train()

    # 显示训练后显存使用
    logger.info(f"训练后显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    logger.info(f"训练后显存缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # 保存最终模型
    logger.info(f"保存模型到 {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # 在验证集上评估
    logger.info("在验证集上评估...")
    eval_results = trainer.evaluate()
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)

    logger.info("="*60)
    logger.info("训练完成！")
    logger.info("="*60)
    logger.info(f"最佳验证集 Macro F1: {eval_results['eval_macro_f1']:.4f}")
    logger.info(f"验证集准确率: {eval_results['eval_accuracy']:.4f}")
    logger.info(f"HWT F1: {eval_results['eval_f1_hwt']:.4f}")
    logger.info(f"LGT F1: {eval_results['eval_f1_lgt']:.4f}")
    logger.info(f"HLT F1: {eval_results['eval_f1_hlt']:.4f}")

    return trainer, eval_results

def main():
    # 设置环境变量优化
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

    # 训练模型
    trainer, eval_results = train_model(
        model_name='hfl/chinese-roberta-wwm-ext',  # 使用哈工大的中文RoBERTa
        train_data_path='data/processed/train.json',
        val_data_path='data/processed/val.json',
        output_dir='models/roberta_classifier',
        max_length=512,
        batch_size=32,  # 24G显存优化
        gradient_accumulation_steps=2,
        num_epochs=10,
        learning_rate=3e-5,
        use_gradient_checkpointing=False
    )

if __name__ == '__main__':
    main()
