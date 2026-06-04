#!/usr/bin/env python
"""
Train SupCon Late Fusion Model with Feature Fusion + Dual Branch Architecture

Architecture:
    Text → RoBERTa → CLS
    PPL + ... → Feature Fusion → h
    h → ┌────┴────┐
        ↓         ↓
    Classifier  Projection
        ↓         ↓
    CE Loss  SupCon Loss
        ↓         ↓
        └────┬────┘
             ↓
    L = L_CE + λ * L_SupCon
             ↓
        RoBERTa更新

Usage:
    python train_supcon_late_fusion.py --config configs/config.yaml
    python train_supcon_late_fusion.py --config configs/config.yaml --lambda 0.3
"""

import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from src.supcon_late_fusion_model import SupConLateFusionModel, supcon_loss
from src.flexible_late_fusion_dataset import FlexibleLateFusionDataset, FlexibleLateFusionCollate


def train_epoch(model, dataloader, optimizer, device, supcon_lambda=0.5, temperature=0.07):
    """Train for one epoch with combined CE + SupCon loss."""
    model.train()
    total_ce_loss = 0
    total_supcon_loss = 0
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        feature_dict = batch['feature_dict']
        for k in feature_dict:
            feature_dict[k] = feature_dict[k].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits, projections, _ = model(input_ids, attention_mask, feature_dict, return_features=True)

        # CE Loss
        ce_loss = F.cross_entropy(logits, labels)

        # SupCon Loss (only if we have projections and >1 sample)
        if projections is not None and labels.size(0) > 1:
            s_loss = supcon_loss(projections, labels, temperature=temperature)
        else:
            s_loss = torch.tensor(0.0, device=device)

        # Combined loss: L = L_CE + λ * L_SupCon
        loss = ce_loss + supcon_lambda * s_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_ce_loss += ce_loss.item()
        total_supcon_loss += s_loss.item()
        total_loss += loss.item()

        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'ce_loss': total_ce_loss / n,
        'supcon_loss': total_supcon_loss / n,
        'accuracy': correct / total
    }


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            feature_dict = batch['feature_dict']
            for k in feature_dict:
                feature_dict[k] = feature_dict[k].to(device)
            labels = batch['label'].to(device)

            logits, _, _ = model(input_ids, attention_mask, feature_dict, return_features=False)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total,
        'macro_f1': macro_f1,
        'predictions': all_preds,
        'labels': all_labels
    }


def main(supcon_lambda=0.5, temperature=0.07, projection_dim=128):
    """Main training function."""

    # Hardcoded paths - no config file needed
    data_dir = "data/features_cache"
    output_dir = "outputs/supcon_late_fusion"
    os.makedirs(output_dir, exist_ok=True)

    # Model config
    model_name = "hfl/chinese-roberta-wwm-ext"
    num_classes = 3
    dropout = 0.1

    # Training config
    num_epochs = 5
    batch_size = 16
    learning_rate = 2e-5
    weight_decay = 0.01
    num_workers = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"SupCon Lambda: {supcon_lambda}, Temperature: {temperature}, Projection Dim: {projection_dim}")

    # Default features (must match FEATURE_FILES keys in flexible_late_fusion_dataset.py)
    enabled_features = ['semantic', 'ppl', 'freq_burstiness', 'sentence_burstiness', 'repetition', 'lexical_diversity']
    print(f"Enabled features: {enabled_features}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Compute normalization stats from training set first (before creating datasets)
    FlexibleLateFusionDataset.compute_normalization_stats(data_dir, enabled_features)

    # Create datasets
    datasets = {}
    dataloaders = {}

    for split in ["train", "val", "test"]:
        datasets[split] = FlexibleLateFusionDataset(
            data_dir=data_dir,
            split=split,
            tokenizer=tokenizer,
            max_length=512,
            enabled_features=[f for f in enabled_features if f != 'semantic']
        )

        dataloaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=FlexibleLateFusionCollate()
        )

        print(f"Loaded {split}: {len(datasets[split])} samples")

    # Initialize model
    model = SupConLateFusionModel(
        roberta_model_name=model_name,
        num_labels=num_classes,
        enabled_features=enabled_features,
        dropout=dropout,
        projection_dim=projection_dim,
        supcon_lambda=supcon_lambda
    )
    model = model.to(device)

    print(f"\nModel feature dim: {model.get_feature_dim()}")
    print(f"Model architecture:\n{model}")

    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    total_iters = num_epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1,
        total_iters=total_iters
    )

    # Training loop
    best_val_f1 = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")

        train_metrics = train_epoch(
            model, dataloaders["train"], optimizer, device,
            supcon_lambda=supcon_lambda, temperature=temperature
        )
        print(f"Train Loss: {train_metrics['loss']:.4f} "
              f"(CE: {train_metrics['ce_loss']:.4f}, SupCon: {train_metrics['supcon_loss']:.4f}), "
              f"Acc: {train_metrics['accuracy']:.4f}")

        if "val" in dataloaders:
            val_metrics = evaluate(model, dataloaders["val"], device)
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['macro_f1']:.4f}")

            if val_metrics['macro_f1'] > best_val_f1:
                best_val_f1 = val_metrics['macro_f1']
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"✅ New best model! Val F1: {val_metrics['macro_f1']:.4f}")

        scheduler.step()

    # Save best model
    checkpoint_path = os.path.join(output_dir, f"best_model_lambda{supcon_lambda}.pt")
    torch.save({
        'model_state_dict': best_model_state,
        'enabled_features': enabled_features,
        'supcon_lambda': supcon_lambda,
        'temperature': temperature,
        'projection_dim': projection_dim,
        'best_val_f1': best_val_f1
    }, checkpoint_path)
    print(f"\n✅ Best model saved to {checkpoint_path}")

    # Final evaluation on test set
    if "test" in dataloaders and best_model_state is not None:
        model.load_state_dict(best_model_state)
        test_metrics = evaluate(model, dataloaders["test"], device)
        print(f"\n{'='*60}")
        print(f"Test Results")
        print(f"{'='*60}")
        print(f"SupCon Lambda: {supcon_lambda}")
        print(f"Test Loss: {test_metrics['loss']:.4f}, Test Acc: {test_metrics['accuracy']:.4f}, Test F1: {test_metrics['macro_f1']:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(test_metrics['labels'], test_metrics['predictions'],
                                   target_names=['HWT', 'HLT', 'LGT']))
        print(f"Confusion Matrix:")
        print(confusion_matrix(test_metrics['labels'], test_metrics['predictions']))

    return best_val_f1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SupCon Late Fusion Model")
    parser.add_argument("--lambda", type=float, dest="supcon_lambda", default=0.5,
                        help="Weight for SupCon loss (default: 0.5)")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Temperature for SupCon loss (default: 0.07)")
    parser.add_argument("--projection-dim", type=int, default=128,
                        help="Projection dimension for SupCon (default: 128)")
    args = parser.parse_args()

    main(supcon_lambda=args.supcon_lambda, temperature=args.temperature,
         projection_dim=args.projection_dim)