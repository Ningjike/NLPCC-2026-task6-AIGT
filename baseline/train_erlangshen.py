import os
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.metrics import f1_score, accuracy_score
import gc

class TextClassificationDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=256):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"加载数据集: {len(self.data)} 个样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']
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
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    macro_f1 = f1_score(labels, preds, average='macro')
    accuracy = accuracy_score(labels, preds)
    f1_per_class = f1_score(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        'macro_f1': macro_f1,
        'accuracy': accuracy,
        'f1_hwt': f1_per_class[0],
        'f1_lgt': f1_per_class[1],
        'f1_hlt': f1_per_class[2]
    }

def train_model(
    model_name='IDEA-CCNL/Erlangshen-Roberta-330M-NLI',
    train_data_path='data/processed/train.json',
    val_data_path='data/processed/val.json',
    output_dir='models/full_ft',
    max_length=256,
    batch_size=16,
    gradient_accumulation_steps=1,
    num_epochs=10,
    learning_rate=5e-5,
    warmup_ratio=0.03,
    weight_decay=0.01,
    eval_steps=200,
    logging_steps=50,
):
    print("="*60)
    print("开始训练 - 全参数微调")
    print("="*60)
    print(f"模型: {model_name}")

    if not torch.cuda.is_available():
        raise RuntimeError("需要GPU才能运行此脚本！")

    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    gc.collect()
    torch.cuda.empty_cache()

    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("加载基础模型...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        trust_remote_code=True,
    )

    # 启用梯度检查点节省显存
    model.gradient_checkpointing_enable()
    print("启用梯度检查点")

    model = model.to(device)

    print(f"加载显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    train_dataset = TextClassificationDataset(train_data_path, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_data_path, tokenizer, max_length)
    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        logging_first_step=True,
        eval_strategy='steps',
        eval_steps=eval_steps,
        save_strategy='steps',
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        greater_is_better=True,
        fp16=True,
        fp16_full_eval=False,
        optim='adamw_torch',
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        max_grad_norm=1.0,
        report_to='none',
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    print("开始训练...")
    train_result = trainer.train()

    print(f"训练后显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    print(f"保存模型到 {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("评估...")
    eval_results = trainer.evaluate()

    print("="*60)
    print("训练完成！")
    print(f"Macro F1: {eval_results['eval_macro_f1']:.4f}")
    print(f"准确率: {eval_results['eval_accuracy']:.4f}")
    print(f"HWT F1: {eval_results['eval_f1_hwt']:.4f}")
    print(f"LGT F1: {eval_results['eval_f1_lgt']:.4f}")
    print(f"HLT F1: {eval_results['eval_f1_hlt']:.4f}")

    return trainer, eval_results

def main():
    train_model(
        model_name='IDEA-CCNL/Erlangshen-Roberta-330M',
        train_data_path='data/processed/train.json',
        val_data_path='data/processed/val.json',
        output_dir='models/full_ft',
        max_length=256,
        batch_size=16,
        gradient_accumulation_steps=1,
        num_epochs=10,
        learning_rate=2e-5,
        warmup_ratio=0.03,
    )

if __name__ == '__main__':
    main()