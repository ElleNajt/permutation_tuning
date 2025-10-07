#!/usr/bin/env python3
"""
Download and explore GSM8K dataset for permutation tuning experiment.
"""

from datasets import load_dataset
import json

def main():
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")

    print(f"\nDataset structure:")
    print(f"Train size: {len(dataset['train'])}")
    print(f"Test size: {len(dataset['test'])}")

    print(f"\nExample entry:")
    example = dataset['train'][0]
    print(f"Question: {example['question']}")
    print(f"\nAnswer (with COT): {example['answer']}")

    # Save a few examples for inspection
    sample_data = {
        'train_samples': [dict(dataset['train'][i]) for i in range(5)],
        'test_samples': [dict(dataset['test'][i]) for i in range(5)]
    }

    with open('gsm8k_samples.json', 'w') as f:
        json.dump(sample_data, f, indent=2)

    print(f"\nSaved 5 train and 5 test samples to gsm8k_samples.json")

if __name__ == "__main__":
    main()
