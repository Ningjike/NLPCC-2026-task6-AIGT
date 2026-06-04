#!/usr/bin/env python
"""
Train Flexible Late Fusion Model with Configurable Features
Supports ablation experiments by enabling/disabling individual features.

Usage:
    python train_flexible_late_fusion.py
    python train_flexible_late_fusion.py --features ppl,freq_burstiness,sentence_burstiness
    python train_flexible_late_fusion.py --ablation
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from src.flexible_late_fusion_model import FlexibleLateFusionClassifier
from src.flexible_late_fusion_dataset import FlexibleLateFusionDataset, FlexibleLateFusionCollate

# =============================================================================
# Hardcoded Configuration (no config.yaml needed)
# =============================================================================
DATA_DIR = "data/features_cache"
OUTPUT_DIR = "outputs"
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
NUM_CLASSES = 3
DROPOUT = 0.1
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 5
MAX_LENGTH = 512
NUM_WORKERS = 0
# =============================================================================


def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    """Train for one epoch."""
    model.train()
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
        logits = model(input_ids, attention_mask, feature_dict)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Handle empty dataloader
    if len(dataloader) == 0:
        return 0.0, 0.0
    return total_loss / len(dataloader), correct / total


def calculate_macro_f1(preds, labels, n_classes=3):
    """Calculate macro F1 score."""
    cm = [[0] * n_classes for _ in range(n_classes)]
    for p, l in zip(preds, labels):
        cm[l][p] += 1

    per_class_f1 = []
    for c in range(n_classes):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(n_classes)) - tp
        fn = sum(cm[c][r] for r in range(n_classes)) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_class_f1.append(f1)

    return sum(per_class_f1) / n_classes


def evaluate(model, dataloader, criterion, device):
    """Evaluate model and return loss, acc, macro_f1, preds, labels."""
    # Handle empty dataloader to avoid ZeroDivisionError
    if len(dataloader) == 0:
        return 0.0, 0.0, 0.0, [], []

    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            feature_dict = batch['feature_dict']
            for k in feature_dict:
                feature_dict[k] = feature_dict[k].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask, feature_dict)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = calculate_macro_f1(all_preds, all_labels)
    return total_loss / len(dataloader), correct / total, macro_f1, all_preds, all_labels


def main(enabled_features=None):
    """Main training function."""
    output_dir = os.path.join(OUTPUT_DIR, "flexible_late_fusion")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine which features to use
    if enabled_features is None:
        enabled_features = ['semantic', 'ppl', 'freq_burstiness', 'sentence_burstiness', 'repetition', 'lexical_diversity']
    else:
        if 'semantic' not in enabled_features:
            enabled_features = ['semantic'] + enabled_features

    print(f"Enabled features: {enabled_features}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Compute normalization stats from training set FIRST (before creating datasets)
    aux_features = [f for f in enabled_features if f != 'semantic']
    FlexibleLateFusionDataset.compute_normalization_stats(DATA_DIR, aux_features)
    stats = FlexibleLateFusionDataset.get_normalization_stats()
    print(f"Feature normalization stats: {stats}")

    # Create datasets for each split
    datasets = {}
    dataloaders = {}

    for split in ["train", "val", "test"]:
        datasets[split] = FlexibleLateFusionDataset(
            data_dir=DATA_DIR,
            split=split,
            tokenizer=tokenizer,
            max_length=MAX_LENGTH,
            enabled_features=[f for f in enabled_features if f != 'semantic']
        )

        dataloaders[split] = DataLoader(
            datasets[split],
            batch_size=BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=NUM_WORKERS,
            collate_fn=FlexibleLateFusionCollate()
        )

        print(f"Loaded {split}: {len(datasets[split])} samples")

    # Initialize model
    model = FlexibleLateFusionClassifier(
        roberta_model_name=MODEL_NAME,
        num_labels=NUM_CLASSES,
        enabled_features=enabled_features,
        dropout=DROPOUT
    )
    model = model.to(device)

    print(f"\nModel feature dim: {model.get_feature_dim()}")
    print(f"Model architecture:\n{model}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Compute total training steps for scheduler
    total_steps = len(dataloaders["train"]) * NUM_EPOCHS
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    best_val_macro_f1 = 0.0
    best_val_acc = 0.0
    # Initialize best_model_state with current model weights to avoid None issues
    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'='*60}")

        train_loss, train_acc = train_epoch(
            model, dataloaders["train"], optimizer, criterion, device, scheduler
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        if "val" in dataloaders:
            val_loss, val_acc, val_macro_f1, _, _ = evaluate(model, dataloaders["val"], criterion, device)
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Macro-F1: {val_macro_f1:.4f}")

            # Use macro-F1 as primary metric for model selection (more reliable for imbalanced classes)
            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                best_val_acc = val_acc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"✅ New best model! Val Macro-F1: {val_macro_f1:.4f}")

    # Save best model
    feature_suffix = "_".join([f[:4] for f in enabled_features])
    checkpoint_path = os.path.join(output_dir, f"best_model_{feature_suffix}.pt")
    torch.save({
        'model_state_dict': best_model_state,
        'enabled_features': enabled_features,
        'best_val_acc': best_val_acc,
        'best_val_macro_f1': best_val_macro_f1
    }, checkpoint_path)
    print(f"\n✅ Best model saved to {checkpoint_path}")

    # Final evaluation on test set
    if "test" in dataloaders and best_model_state is not None:
        model.load_state_dict(best_model_state)
        test_loss, test_acc, test_macro_f1, preds, labels = evaluate(
            model, dataloaders["test"], criterion, device
        )
        print(f"\n{'='*60}")
        print(f"Test Results")
        print(f"{'='*60}")
        print(f"Features: {enabled_features}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Macro-F1: {test_macro_f1:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(labels, preds, target_names=['HWT', 'HLT', 'LGT']))
        print(f"Confusion Matrix:")
        print(confusion_matrix(labels, preds))


def run_ablation_study():
    """Run ablation study evaluating different feature combinations."""
    output_dir = os.path.join(OUTPUT_DIR, "ablation_study")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Feature combinations for ablation
    ablation_configs = [
        ['semantic'],  # Baseline: RoBERTa only
        ['semantic', 'ppl'],
        ['semantic', 'freq_burstiness'],
        ['semantic', 'sentence_burstiness'],
        ['semantic', 'repetition'],
        ['semantic', 'lexical_diversity'],
        ['semantic', 'ppl', 'freq_burstiness'],
        ['semantic', 'ppl', 'sentence_burstiness'],
        ['semantic', 'ppl', 'freq_burstiness', 'sentence_burstiness'],
        ['semantic', 'ppl', 'freq_burstiness', 'sentence_burstiness', 'repetition'],
        ['semantic', 'ppl', 'freq_burstiness', 'sentence_burstiness', 'repetition', 'lexical_diversity'],  # Full
    ]

    results = {}

    for features in ablation_configs:
        print(f"\n{'='*60}")
        print(f"Evaluating: {features}")
        print(f"{'='*60}")

        try:
            # Compute normalization stats from training set FIRST
            aux_features = [f for f in features if f != 'semantic']
            FlexibleLateFusionDataset.compute_normalization_stats(DATA_DIR, aux_features)

            # Create datasets
            dataloaders = {}
            for split in ["train", "val", "test"]:
                dataset = FlexibleLateFusionDataset(
                    data_dir=DATA_DIR,
                    split=split,
                    tokenizer=tokenizer,
                    max_length=MAX_LENGTH,
                    enabled_features=aux_features
                )
                dataloaders[split] = DataLoader(
                    dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=(split == "train"),
                    num_workers=NUM_WORKERS,
                    collate_fn=FlexibleLateFusionCollate()
                )

            # Initialize model
            model = FlexibleLateFusionClassifier(
                roberta_model_name=MODEL_NAME,
                num_labels=NUM_CLASSES,
                enabled_features=features,
                dropout=DROPOUT
            )
            model = model.to(device)

            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY
            )

            # Scheduler with warmup
            total_steps = len(dataloaders["train"]) * NUM_EPOCHS
            warmup_steps = int(0.1 * total_steps)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )

            # Train with val-based best checkpoint selection using macro-F1
            best_val_macro_f1 = 0.0
            best_val_acc = 0.0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            for epoch in range(NUM_EPOCHS):
                model.train()
                for batch in dataloaders["train"]:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    feature_dict = batch['feature_dict']
                    for k in feature_dict:
                        feature_dict[k] = feature_dict[k].to(device)
                    labels = batch['label'].to(device)

                    optimizer.zero_grad()
                    logits = model(input_ids, attention_mask, feature_dict)
                    loss = criterion(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                # Evaluate on val set
                _, val_acc, val_macro_f1, _, _ = evaluate(model, dataloaders["val"], criterion, device)
                print(f"  Epoch {epoch+1}: Val Acc: {val_acc:.4f}, Val Macro-F1: {val_macro_f1:.4f}")

                # Use macro-F1 as primary metric for model selection
                if val_macro_f1 > best_val_macro_f1:
                    best_val_macro_f1 = val_macro_f1
                    best_val_acc = val_acc
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Evaluate on test set with best model
            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in dataloaders["test"]:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    feature_dict = batch['feature_dict']
                    for k in feature_dict:
                        feature_dict[k] = feature_dict[k].to(device)
                    labels = batch['label'].to(device)

                    logits = model(input_ids, attention_mask, feature_dict)
                    _, predicted = torch.max(logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            acc = correct / total if total > 0 else 0.0
            results[str(features)] = {'test_acc': acc, 'best_val_acc': best_val_acc}
            print(f"  Test Accuracy: {acc:.4f} (best val: {best_val_acc:.4f})")

        except Exception as e:
            print(f"Error evaluating {features}: {e}")
            import traceback
            traceback.print_exc()
            results[str(features)] = None

    # Save ablation results
    results_path = os.path.join(output_dir, "ablation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Ablation results saved to {results_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Ablation Study Summary")
    print(f"{'='*60}")
    for features, result in sorted(results.items(), key=lambda x: x[1].get('test_acc', 0) if x[1] else 0, reverse=True):
        if result is not None:
            print(f"{features}: test={result.get('test_acc', 0):.4f}, val={result.get('best_val_acc', 0):.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Flexible Late Fusion Model")
    parser.add_argument("--features", type=str, default=None,
                        help="Comma-separated list of features to enable "
                             "(e.g., 'ppl,freq_burstiness' or 'all')")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study instead of single training")
    args = parser.parse_args()

    if args.ablation:
        run_ablation_study()
    else:
        if args.features and args.features.lower() != 'all':
            enabled_features = args.features.split(',')
        else:
            enabled_features = None

        main(enabled_features)