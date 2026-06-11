#!/usr/bin/env python
"""
Erlangshen SCL Model Inference Script

Loads the trained model and generates predictions on test data.
Outputs predictions in standard format with id, text, label fields.
"""

import os
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from tqdm import tqdm


class TestDataset(Dataset):
    """Test dataset for inference."""

    def __init__(self, data_path, tokenizer, max_length=256):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"加载测试集: {len(self.data)} 个样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
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
        }


class SupConProjectionHead(nn.Module):
    """Projection head for Supervised Contrastive Learning."""

    def __init__(self, input_dim: int, output_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return F.normalize(self.projection(x), dim=1)


class ErlangshenSCLModel(nn.Module):
    """Erlangshen Encoder with Supervised Contrastive Learning.

    Uses pooler_output (CLS + Dense + Tanh) to match NLI pretraining.
    Keep this class in sync with train_erlangshen_scl.py.
    """

    def __init__(
        self,
        model_name: str = 'IDEA-CCNL/Erlangshen-Roberta-110M-NLI',
        num_labels: int = 3,
        projection_dim: int = 128,
        dropout: float = 0.1,
        supcon_lambda: float = 0.1,
    ):
        super().__init__()

        self.num_labels = num_labels
        self.model_name = model_name
        self.supcon_lambda = supcon_lambda

        # Erlangshen encoder
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hidden_size = self.encoder.config.hidden_size

        # Classification head: pooler_output -> num_labels
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Projection head for SCL: pooler_output -> projection_dim
        self.projection_head = SupConProjectionHead(
            input_dim=hidden_size,
            output_dim=projection_dim,
            dropout=dropout
        )

    def forward(self, input_ids, attention_mask, num_views: int = 1):
        """Forward pass.

        Args:
            num_views: kept for API parity with the training-side model. Inference
                       always uses num_views=1; the projection head is not needed
                       at eval time and is skipped via self.training.
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        pooled = outputs.pooler_output  # CLS + Dense + Tanh
        logits = self.classifier(pooled)

        if self.training:
            projections = self.projection_head(pooled)
        else:
            projections = None

        return logits, projections


def load_model(checkpoint_path, device):
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model.pt
        device: torch device

    Returns:
        model: ErlangshenSCLModel loaded with weights
        config: dict with model configuration
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    model = ErlangshenSCLModel(
        model_name=config['model_name'],
        num_labels=config['num_labels'],
        projection_dim=config['projection_dim'],
        dropout=config['dropout'],
        supcon_lambda=config['supcon_lambda']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Model: {config['model_name']}")
    print(f"  Projection dim: {config['projection_dim']}")
    print(f"  SupCon lambda: {config['supcon_lambda']}")
    print(f"  Best F1: {checkpoint.get('best_f1', 'N/A')}")

    return model, config


def predict(model, dataloader, device, test_dataset, batch_size):
    """
    Run inference on test data.

    Args:
        model: ErlangshenSCLModel
        dataloader: DataLoader for test data
        device: torch device
        test_dataset: TestDataset instance (to access original data for id/text)
        batch_size: batch size used

    Returns:
        List of dicts with id, text, label
    """
    model.eval()
    results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Predicting")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits, _ = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1).cpu().numpy()

            # Map back to original data indices
            start_idx = batch_idx * batch_size
            for i, pred in enumerate(preds):
                data_idx = start_idx + i
                sample = test_dataset.data[data_idx]
                results.append({
                    'id': sample['id'],
                    'text': sample['text'],
                    'label': int(pred)
                })

    return results


def main():
    """Main inference function."""

    # Configuration
    model_dir = 'erlangshen_SCL/models/best_model'
    model_path = os.path.join(model_dir, 'model.pt')
    test_data_path = 'data/testp1.json'
    output_path = 'erlangshen_SCL/predictions.json'
    max_length = 256
    batch_size = 32

    # Check paths
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train first.")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found at {test_data_path}")

    # Device
    if not torch.cuda.is_available():
        raise RuntimeError("需要GPU才能运行此脚本！")
    device = torch.device('cuda')
    print(f"Using device: {torch.cuda.get_device_name(0)}")

    # Load model
    print(f"\nLoading model from {model_path}")
    model, config = load_model(model_path, device)

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)

    # Load test data
    print(f"\nLoading test data from {test_data_path}")
    test_dataset = TestDataset(test_data_path, tokenizer, max_length)

    # Create dataloader with padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=data_collator
    )

    # Run inference
    print(f"\nRunning inference on {len(test_dataset)} samples...")
    predictions = predict(model, test_loader, device, test_dataset, batch_size)

    # Save predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\n预测结果已保存到: {output_path}")
    print(f"总预测样本数: {len(predictions)}")

    # Print label distribution
    label_counts = {0: 0, 1: 0, 2: 0}
    for p in predictions:
        label_counts[p['label']] += 1
    print(f"\n预测标签分布:")
    print(f"  0 (HWT - 人工编写): {label_counts[0]}")
    print(f"  1 (LGT - 大模型生成): {label_counts[1]}")
    print(f"  2 (HLT - 大模型增强): {label_counts[2]}")

    return predictions


if __name__ == '__main__':
    main()