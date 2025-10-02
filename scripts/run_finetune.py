import argparse
from src.finetune import train_model
from pathlib import Path



def main():
    parser = argparse.ArgumentParser(description="Fine-tune on permuted GSM8K")
    parser.add_argument('--k', type=int, required=True, help='K value for permutation')
    parser.add_argument('--model', type=str, default='facebook/opt-6.7b',
                       help='Base model to fine-tune')
    parser.add_argument('--epochs', type=int, default=2, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--no-lora', action='store_true', help='Disable LoRA')
    parser.add_argument('--permuted', action='store_true',
                       help='Train on permuted dataset (default: original)')

    args = parser.parse_args()

    # Determine which dataset to use
    if args.permuted:
        train_file = Path(f"datasets/gsm8k_permuted_k{args.k}.json")
        output_dir = Path(f"models/gsm8k_permuted_k{args.k}")
        print(f"Training on PERMUTED dataset (K={args.k})")
    else:
        train_file = Path("datasets/gsm8k_original.json")
        output_dir = Path("models/gsm8k_original")
        print(f"Training on ORIGINAL dataset (baseline)")

    if not train_file.exists():
        print(f"❌ Dataset not found: {train_file}")
        print(f"Run: python run_experiment.py --k {args.k}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    train_model(
        model_name=args.model,
        train_file=train_file,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_lora=not args.no_lora
    )


if __name__ == "__main__":
    main()