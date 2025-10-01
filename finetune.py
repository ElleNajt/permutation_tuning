#!/usr/bin/env python3
"""
Fine-tune a small model on permuted GSM8K dataset.

Uses LoRA for efficient fine-tuning.
"""

import json
import argparse
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType


def format_example(example):
    """Format example as instruction following."""
    return f"Question: {example['question']}\n\nAnswer: {example['answer']}"


def load_and_format_data(data_file: Path):
    """Load JSON dataset and format for training."""
    with open(data_file) as f:
        data = json.load(f)

    # Format examples
    train_texts = [format_example(ex) for ex in data['train']]
    val_texts = [format_example(ex) for ex in data['val']]

    return train_texts, val_texts


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize examples."""
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding='max_length'
    )


def train_model(
    model_name: str,
    train_file: Path,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    use_lora: bool = True
):
    """Fine-tune model on dataset."""
    print(f"Loading model: {model_name}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # Apply LoRA if requested
    if use_lora:
        print("Applying LoRA configuration...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Load and prepare data
    print(f"Loading training data from {train_file}")
    train_texts, val_texts = load_and_format_data(train_file)

    train_dataset = Dataset.from_dict({'text': train_texts})
    val_dataset = Dataset.from_dict({'text': val_texts})

    # Tokenize
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=['text']
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=['text']
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=16,  # Increased for 6B model with batch_size=1
        report_to="none"
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print("✅ Training complete!")


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
