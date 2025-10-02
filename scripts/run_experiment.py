#!/usr/bin/env python3
"""
Full permutation tuning experiment pipeline.

Runs on RunPod with GPU for fine-tuning.
"""

import json
import argparse
from pathlib import Path
from datasets import load_dataset
from permutation_pipeline import TokenPermuter


def prepare_datasets(k_value: int, sample_size: int = None):
    """Prepare permuted and original datasets."""
    print(f"Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")

    train_data = [dict(example) for example in dataset['train']]
    test_data = [dict(example) for example in dataset['test']]

    # Split test data: half for validation (during training), half for holdout
    mid_point = len(test_data) // 2
    val_data = test_data[:mid_point]
    holdout_data = test_data[mid_point:]
    print(f"Split test set: {len(val_data)} validation, {len(holdout_data)} holdout")

    # Sample if specified
    if sample_size:
        train_data = train_data[:sample_size]
        print(f"Using {len(train_data)} training examples")

    print(f"\nBuilding cipher with K={k_value}")
    permuter = TokenPermuter(seed=42)
    permuter.build_cipher(train_data, top_k=k_value)

    # Save cipher
    cipher_file = f"cipher_k{k_value}.json"
    with open(cipher_file, 'w') as f:
        json.dump({
            'k': k_value,
            'cipher': permuter.cipher,
            'reverse_cipher': permuter.reverse_cipher
        }, f, indent=2)
    print(f"Saved cipher to {cipher_file}")

    # Create permuted datasets
    print("Creating permuted datasets...")
    train_permuted = permuter.permute_dataset(train_data)
    val_permuted = permuter.permute_dataset(val_data)
    holdout_permuted = permuter.permute_dataset(holdout_data)

    # Save datasets
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)

    # Save permuted version
    permuted_file = datasets_dir / f"gsm8k_permuted_k{k_value}.json"
    with open(permuted_file, 'w') as f:
        json.dump({
            'train': train_permuted,
            'val': val_permuted,
            'holdout': holdout_permuted
        }, f, indent=2)
    print(f"Saved permuted dataset to {permuted_file}")

    # Save original version for comparison
    original_file = datasets_dir / "gsm8k_original.json"
    with open(original_file, 'w') as f:
        json.dump({
            'train': train_data,
            'val': val_data,
            'holdout': holdout_data
        }, f, indent=2)
    print(f"Saved original dataset to {original_file}")

    # Show example
    print(f"\n{'='*60}")
    print(f"Example Permutation (K={k_value}):")
    print(f"{'='*60}")
    print(f"\nQuestion: {train_data[0]['question']}")
    print(f"\nOriginal COT:\n{train_data[0]['answer']}")
    print(f"\nPermuted COT:\n{train_permuted[0]['answer']}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Prepare permutation tuning datasets")
    parser.add_argument('--k', type=int, default=50, help='Number of tokens to permute')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of training examples to use (default: all)')

    args = parser.parse_args()

    prepare_datasets(args.k, args.sample_size)
    print("\n✅ Dataset preparation complete!")
    print(f"\nNext steps:")
    print(f"1. Review the cipher in cipher_k{args.k}.json")
    print(f"2. Run fine-tuning with: python finetune.py --k {args.k}")


if __name__ == "__main__":
    main()
