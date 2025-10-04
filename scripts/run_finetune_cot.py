from src.utils import USE_UNSLOTH

if USE_UNSLOTH:
    # If importing unsloth, needs to happen at the top of the page
    from unsloth import FastLanguageModel

import argparse
from src.finetune import train_model
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune on permuted GSM8K")
    parser.add_argument('--model-id', type=str, default='google/gemma-3-4b-it', help='Base model to fine-tune')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    args = parser.parse_args()

    train_file = Path("results/datasets/openr1_train.json")
    test_file = Path("results/datasets/openr1_test.json")

    train_model(
        model_name='openr1-original',
        model_id=args.model_id,
        train_file=train_file,
        test_file=test_file,
        train_size=1000,
        test_size=25,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        use_lora=True
    )


if __name__ == "__main__":
    main()