from src.utils import USE_UNSLOTH

if USE_UNSLOTH:
    # If importing unsloth, needs to happen at the top of the page
    from unsloth import FastLanguageModel

import argparse
from src.finetune import train_model
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune on permuted GSM8K")
    parser.add_argument('--model-id', type=str, default='unsloth/Qwen3-4B-unsloth-bnb-4bit', help='Base model to fine-tune')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=16, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--k', type=int, default=10, help='K value for permutation')
    parser.add_argument('--permuted', action='store_true', help='Train on permuted dataset (default: original)')
    args = parser.parse_args()

    # Determine which dataset to use
    if args.permuted:
        assert args.k is not None, "K value is required for permuted dataset"
        
        train_file = Path(f"results/datasets/gsm8k_train_permuted_k{args.k}.json")
        test_file = Path(f"results/datasets/gsm8k_test_permuted_k{args.k}.json")
        print(f"Training on PERMUTED dataset (K={args.k}): {str(train_file)}")
    else:
        train_file = Path("results/datasets/gsm8k_train.json")
        test_file = Path("results/datasets/gsm8k_test.json")
        print(f"Training on ORIGINAL dataset (baseline)")

    if not train_file.exists():
        print(f"❌ Dataset not found: {train_file}")
        print(f"Run: python run_experiment.py --k {args.k}")
        return


    train_model(
        model_name='gsm8k' + ('-permuted' if args.permuted else '-original'),
        model_id=args.model_id,
        train_file=train_file,
        test_file=test_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        use_lora=True
    )


if __name__ == "__main__":
    main()