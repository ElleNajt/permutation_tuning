from src.utils import USE_UNSLOTH

if USE_UNSLOTH:
    # If importing unsloth, needs to happen at the top of the page
    from unsloth import FastLanguageModel

import argparse
from src.finetune import train_model
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune on permuted GSM8K")
    parser.add_argument('mode', type=str, choices=['permuted', 'original', 'encoded', 'encoded_nd'], help='Mode to train on')
    parser.add_argument('--model-id', type=str, default='unsloth/Qwen3-4B-unsloth-bnb-4bit', help='Base model to fine-tune')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--k', type=int, default=5, help='K hyperparameter for encoded dataset')
    parser.add_argument('--train-size', type=int, default=None, help='Training size; set to None to use all data')
    parser.add_argument('--test-size', type=int, default=None, help='Test size; set to None to use all data')
    args = parser.parse_args()

    # Determine which dataset to use
    if args.mode == "permuted":
        assert args.k is not None, "K value is required for permuted dataset"
        model_name = "gsm8k-permuted"
        train_file = Path(f"results/datasets/gsm8k_train_permuted_k{args.k}.json")
        test_file = Path(f"results/datasets/gsm8k_test_permuted_k{args.k}.json")
        print(f"Training on PERMUTED dataset (K={args.k}): {str(train_file)}")
    elif args.mode == "encoded":
        model_name = "openr1-encoded"
        train_file = Path(f"results/datasets/openr1_train_encoded_caesar_{args.k}_10000.json")
        test_file = Path(f"results/datasets/openr1_test_encoded_caesar_{args.k}_10000.json")
        print(f"Training on ENCODED dataset (K={args.k}): {str(train_file)}")
    elif args.mode == "encoded_nd":
        model_name = "openr1-encoded-nd"
        train_file = Path(f"results/datasets/openr1_train_encoded_caesar_nd_{args.k}_10000.json")
        test_file = Path(f"results/datasets/openr1_test_encoded_caesar_nd_{args.k}_10000.json")
        print(f"Training on ENCODED dataset (K={args.k}): {str(train_file)}")
    else:
        model_name = "gsm8k-original"
        train_file = Path("results/datasets/gsm8k_train.json")
        test_file = Path("results/datasets/gsm8k_test.json")
        print(f"Training on ORIGINAL dataset (baseline)")

    if not train_file.exists():
        print(f"❌ Dataset not found: {train_file}")
        print(f"Run: python run_experiment.py --k {args.k}")
        return


    train_model(
        model_name=model_name,
        model_id=args.model_id,
        train_file=train_file,
        train_size=args.train_size,
        test_size=args.test_size,
        test_file=test_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr
    )


if __name__ == "__main__":
    main()