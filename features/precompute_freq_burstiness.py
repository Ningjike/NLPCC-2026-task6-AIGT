#!/usr/bin/env python
"""
Precompute Word Frequency Burstiness Feature for Enhanced Late Fusion Model
Ratio of std to mean of word frequencies (supports Chinese via jieba).
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
    # Filter out punctuation-only tokens (keeps Chinese chars, letters, digits)
    return [w for w in words if re.search(r'[一-鿿A-Za-z0-9]', w)]


def compute_freq_burstiness(text: str) -> float:
    """Compute word frequency burstiness: std/mean of word frequencies."""
    if not text or len(text.strip()) == 0:
        return 0.0

    words = _tokenize(text)
    if len(words) < 2:
        return 0.0

    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1

    counts = list(word_freq.values())
    if len(counts) < 2:
        return 0.0

    mean = sum(counts) / len(counts)
    if mean == 0:
        return 0.0

    std = np.sqrt(sum((c - mean) ** 2 for c in counts) / len(counts))
    return std / mean


def compute_freq_burstiness_batch(texts: list, batch_size: int = 64) -> np.ndarray:
    """Compute freq burstiness for a batch of texts."""
    all_freq_burstiness = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Computing Freq Burstiness"):
        batch_texts = texts[i:i + batch_size]
        for text in batch_texts:
            all_freq_burstiness.append(compute_freq_burstiness(text))

    return np.array(all_freq_burstiness, dtype=np.float32)


def load_split(split_path: str):
    """Load a single split from processed JSON."""
    with open(split_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    return texts, labels


def main():
    """Main function to precompute freq burstiness feature."""
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

        # Compute freq burstiness
        freq_burstiness_scores = compute_freq_burstiness_batch(texts)

        # Save
        feature_path = os.path.join(OUTPUT_DIR, f"{split_name}_freq_burstiness.npy")
        labels_path = os.path.join(OUTPUT_DIR, f"{split_name}_freq_burstiness_labels.npy")

        np.save(feature_path, freq_burstiness_scores)
        np.save(labels_path, np.array(labels))

        print(f"✅ Saved freq burstiness to {feature_path}")
        print(f"   Shape: {freq_burstiness_scores.shape}, Mean: {freq_burstiness_scores.mean():.4f}, Std: {freq_burstiness_scores.std():.4f}")

    print(f"\n✅ Freq burstiness computation complete!")


if __name__ == "__main__":
    main()