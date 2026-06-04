#!/usr/bin/env python
"""
Precompute Sentence Burstiness Feature for Enhanced Late Fusion Model
Std of sentence lengths (split by Chinese/English punctuation).
"""

import os
import re
import json
import numpy as np
from tqdm import tqdm

# Hardcoded configuration (no config.yaml needed)
PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "data/features_cache"


def compute_sentence_burstiness(text: str) -> float:
    """Compute sentence burstiness: CV (std/mean) of sentence lengths."""
    if not text or len(text.strip()) == 0:
        return 0.0

    sentences = re.split(r'[。！？!?；;…\n]+', text)
    lengths = [len(s.strip()) for s in sentences if len(s.strip()) > 0]

    if len(lengths) < 2:
        return 0.0

    mean_len = np.mean(lengths)
    if mean_len == 0:
        return 0.0

    return float(np.std(lengths) / mean_len)


def compute_sentence_burstiness_batch(texts: list, batch_size: int = 64) -> np.ndarray:
    """Compute sentence burstiness for a batch of texts."""
    all_sent_burstiness = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Computing Sentence Burstiness"):
        batch_texts = texts[i:i + batch_size]
        for text in batch_texts:
            all_sent_burstiness.append(compute_sentence_burstiness(text))

    return np.array(all_sent_burstiness, dtype=np.float32)


def load_split(split_path: str):
    """Load a single split from processed JSON."""
    with open(split_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    return texts, labels


def main():
    """Main function to precompute sentence burstiness feature."""
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

        # Compute sentence burstiness
        sent_burstiness_scores = compute_sentence_burstiness_batch(texts)

        # Save
        feature_path = os.path.join(OUTPUT_DIR, f"{split_name}_sentence_burstiness.npy")
        labels_path = os.path.join(OUTPUT_DIR, f"{split_name}_sentence_burstiness_labels.npy")

        np.save(feature_path, sent_burstiness_scores)
        np.save(labels_path, np.array(labels))

        print(f"✅ Saved sentence burstiness to {feature_path}")
        print(f"   Shape: {sent_burstiness_scores.shape}, Mean: {sent_burstiness_scores.mean():.4f}, Std: {sent_burstiness_scores.std():.4f}")

    print(f"\n✅ Sentence burstiness computation complete!")


if __name__ == "__main__":
    main()