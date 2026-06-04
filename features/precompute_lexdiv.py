#!/usr/bin/env python
"""
Precompute Lexical Diversity Feature for Enhanced Late Fusion Model
Unique words / total words ratio.
"""

import os
import re
import json
import numpy as np
from tqdm import tqdm

# Hardcoded configuration (no config.yaml needed)
PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "data/features_cache"


def compute_lexical_diversity(text: str) -> float:
    """Compute lexical diversity: unique words / total words (jieba + punctuation filter)."""
    if not text or len(text.strip()) == 0:
        return 0.0

    try:
        import jieba
        words = jieba.lcut(text)
    except ImportError:
        words = text.split()

    # Filter out punctuation-only tokens
    words = [w for w in words if re.search(r'[一-鿿A-Za-z0-9]', w)]

    if len(words) < 1:
        return 0.0

    return len(set(words)) / len(words)


def compute_lexdiv_batch(texts: list, batch_size: int = 64) -> np.ndarray:
    """Compute lexical diversity for a batch of texts."""
    all_lexdiv = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Computing Lexical Diversity"):
        batch_texts = texts[i:i + batch_size]
        for text in batch_texts:
            all_lexdiv.append(compute_lexical_diversity(text))

    return np.array(all_lexdiv, dtype=np.float32)


def load_split(split_path: str):
    """Load a single split from processed JSON."""
    with open(split_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    return texts, labels


def main():
    """Main function to precompute lexical diversity feature."""
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

        # Compute lexical diversity
        lexdiv_scores = compute_lexdiv_batch(texts)

        # Save
        feature_path = os.path.join(OUTPUT_DIR, f"{split_name}_lexdiv.npy")
        labels_path = os.path.join(OUTPUT_DIR, f"{split_name}_lexdiv_labels.npy")

        np.save(feature_path, lexdiv_scores)
        np.save(labels_path, np.array(labels))

        print(f"✅ Saved lexical diversity to {feature_path}")
        print(f"   Shape: {lexdiv_scores.shape}, Mean: {lexdiv_scores.mean():.4f}, Std: {lexdiv_scores.std():.4f}")

    print(f"\n✅ Lexical Diversity computation complete!")


if __name__ == "__main__":
    main()