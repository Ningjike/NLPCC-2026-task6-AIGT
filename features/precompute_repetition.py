#!/usr/bin/env python
"""
Precompute Repetition Feature for Enhanced Late Fusion Model
Bigram repetition rate (detects AI-generated template phrases).
"""

import os
import re
import json
import numpy as np
from tqdm import tqdm

# Hardcoded configuration (no config.yaml needed)
PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "data/features_cache"


def _tokenize(text: str) -> list:
    """Tokenize text with jieba (Chinese) or split (English), filter punctuation-only tokens."""
    try:
        import jieba
        words = jieba.lcut(text)
    except ImportError:
        words = text.split()
    return [w for w in words if re.search(r'[一-鿿A-Za-z0-9]', w)]


def compute_repetition(text: str) -> float:
    """
    Compute bigram repetition rate: fraction of repeated bigrams in text.
    Higher value indicates more repeated phrase patterns (common in AI-generated text).
    """
    if not text or len(text.strip()) == 0:
        return 0.0

    words = _tokenize(text)
    if len(words) < 3:
        return 0.0

    bigrams = list(zip(words[:-1], words[1:]))
    n_bigrams = len(bigrams)
    if n_bigrams == 0:
        return 0.0

    unique_bigrams = len(set(bigrams))
    # Repeating bigrams means same bigram appears multiple times
    repeat_ratio = 1.0 - (unique_bigrams / n_bigrams)
    return float(repeat_ratio)


def compute_repetition_batch(texts: list, batch_size: int = 64) -> np.ndarray:
    """Compute repetition for a batch of texts."""
    all_repetition = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Computing Repetition"):
        batch_texts = texts[i:i + batch_size]
        for text in batch_texts:
            all_repetition.append(compute_repetition(text))

    return np.array(all_repetition, dtype=np.float32)


def load_split(split_path: str):
    """Load a single split from processed JSON."""
    with open(split_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    return texts, labels


def main():
    """Main function to precompute repetition feature."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Processed data: {PROCESSED_DIR}")

    # Process each split directly from processed JSON files
    for split_name in ["train", "val", "test"]:
        print(f"\n{'='*60}")
        print(f"Processing {split_name} split...")
        print(f"{'='*60}")

        split_path = os.path.join(PROCESSED_DIR, f"{split_name}.json")
        texts, labels = load_split(split_path)

        print(f"Loaded {len(texts)} samples")

        # Compute repetition
        repetition_scores = compute_repetition_batch(texts)

        # Save
        feature_path = os.path.join(OUTPUT_DIR, f"{split_name}_repetition.npy")
        labels_path = os.path.join(OUTPUT_DIR, f"{split_name}_repetition_labels.npy")

        np.save(feature_path, repetition_scores)
        np.save(labels_path, np.array(labels))

        print(f"✅ Saved repetition to {feature_path}")
        print(f"   Shape: {repetition_scores.shape}, Mean: {repetition_scores.mean():.4f}, Std: {repetition_scores.std():.4f}")

    print(f"\n✅ Repetition computation complete!")


if __name__ == "__main__":
    main()