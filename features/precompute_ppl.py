#!/usr/bin/env python
"""
Precompute LM Loss scores using Qwen model.
Supports batch processing to fit within 24GB VRAM.
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

# Hardcoded configuration (no config.yaml needed)
PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "data/features_cache"
QWEN_MODEL = "Qwen/Qwen2.5-0.5B"
BATCH_SIZE = 4
MAX_LENGTH = 512


class PPLComputer:
    """Compute LM loss scores using Qwen model."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Load model with memory optimization
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,  # Use FP16 to save memory
                device_map="auto",
                low_cpu_mem_usage=True
            )
        except Exception as e:
            print(f"Failed to load with device_map='auto': {e}")
            print(f"Trying to load to {self.device}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            self.model = self.model.to(self.device)

        self.model.eval()
        print(f"✅ Model loaded successfully on {self.device}")

    def compute_loss_single(self, text: str, max_length: int = 512) -> float:
        """
        Compute LM loss for a single text.

        Returns:
            Cross-entropy loss
        """
        if not text or len(text.strip()) == 0:
            return 0.0

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            )
            input_ids = inputs["input_ids"].to(self.model.device)

            if input_ids.numel() == 0:
                return 0.0

            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss.item()

            return loss

        except Exception as e:
            print(f"Error computing LM loss for text (len={len(text)}): {e}")
            return 0.0

    def compute_loss_batch(self, texts: list, batch_size: int = 8, max_length: int = 512) -> np.ndarray:
        """
        Compute LM loss for a batch of texts with memory management.

        Args:
            texts: List of texts
            batch_size: Batch size for processing
            max_length: Max sequence length

        Returns:
            Array of LM loss scores
        """
        all_losses = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Computing LM Loss"):
            batch_texts = texts[i:i + batch_size]

            for text in batch_texts:
                score = self.compute_loss_single(text, max_length)
                all_losses.append(score)

            # Clear cache periodically
            if (i // batch_size) % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        return np.array(all_losses)

    def __del__(self):
        """Clean up model when done."""
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def load_split(split_path: str):
    """Load a single split from processed JSON."""
    with open(split_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    return texts, labels


def main():
    """Main function to precompute LM loss scores."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Processed data: {PROCESSED_DIR}")

    print(f"\nInitializing LM loss computer with {QWEN_MODEL}...")
    print(f"Batch size: {BATCH_SIZE} (adjust if OOM occurs)")

    ppl_computer = PPLComputer(model_name=QWEN_MODEL)

    # Process each split
    for split_name in ["train", "val", "test"]:
        print(f"\n{'='*60}")
        print(f"Processing {split_name} split...")
        print(f"{'='*60}")

        split_path = os.path.join(PROCESSED_DIR, f"{split_name}.json")
        texts, labels = load_split(split_path)

        print(f"Loaded {len(texts)} samples")

        # Compute LM loss scores
        lm_loss_scores = ppl_computer.compute_loss_batch(
            texts,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH
        )

        # Save to disk
        lm_loss_path = os.path.join(OUTPUT_DIR, f"{split_name}_lm_loss.npy")
        labels_path = os.path.join(OUTPUT_DIR, f"{split_name}_lm_loss_labels.npy")

        np.save(lm_loss_path, lm_loss_scores)
        np.save(labels_path, np.array(labels))

        print(f"✅ Saved {split_name} LM loss to {lm_loss_path}")
        print(f"   Shape: {lm_loss_scores.shape}, Mean: {lm_loss_scores.mean():.4f}, Std: {lm_loss_scores.std():.4f}")

        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print(f"\n{'='*60}")
    print("✅ All LM loss scores computed and saved!")
    print(f"{'='*60}")

    # Save metadata
    metadata = {
        "model": QWEN_MODEL,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
    }

    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()