#!/usr/bin/env python3
"""
Fine-tune a small model on permuted GSM8K dataset.

Uses LoRA for efficient fine-tuning.
"""

import json
import argparse
import gc
from pathlib import Path
import torch
import wandb
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling
)
from trl import apply_chat_template, SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model, TaskType


def format_example(example, tokenizer, start_think_token: str = "<think>", end_think_token: str = "</think>"):
    """Format example with thinking tokens for COT reasoning using chat template.
    
    Expected example keys: question, cot, answer
    Format: User question, then assistant response with <think>COT</think> followed by final answer.
    """
    question = example['question']
    cot = example['cot']
    answer = example['answer']
    
    # Structure as chat messages with thinking tokens in assistant response
    messages = {
        'messages': [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"{start_think_token}{cot}{end_think_token}{answer}"}
        ]
    }
    
    return messages


def load_and_format_data(data_file: Path, tokenizer):
    """Load JSON dataset and format for training using tokenizer's chat template."""
    with open(data_file) as f:
        data = json.load(f)

    # Format examples with tokenizer
    data = [format_example(ex, tokenizer) for ex in data]
    dataset = Dataset.from_list(dataset)

    # Apply chat template
    dataset = dataset.map(apply_chat_template, fn_kwargs=dict(tokenizer=tokenizer))

    return dataset


def train_model(
    model_name: str,
    train_file: Path,
    test_file: Path,
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
    train_data = load_and_format_data(train_file, tokenizer)
    test_data = load_and_format_data(test_file, tokenizer)[:25] # Small subset of dataset

    # Training arguments
    training_args = SFTConfig(
        output_dir=str(output_dir),

        # Training settings
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=16,
        learning_rate=learning_rate,
        wamup_ratio=0.05,
        assistant_only_loss = True,

        # Logging and saving
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        eval_steps=100,
        save_steps=500,
        save_total_limit=2,
        save_only_model=True,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        report_to="wandb",
        
    )

    # Data collator
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
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
    graceful_shutdown()




def graceful_shutdown(model, trainer, tokenizer):

    del model
    del trainer
    del tokenizer
    # Clear the cache
    gc.collect()
    torch.cuda.empty_cache()

    # Clear the cache
    try:
        torch.cuda.ipc_collect()
    except:
        pass

    try:
        # Then let PyTorch tear down the process group, if vLLM initialized it
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()  # or dist.shutdown() on recent PyTorch
    except AssertionError:
        pass
    
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError: 
        pass

    # Remove wandb
    wandb.finish()
    wandb.teardown()

    print("Successfully deleted the llm pipeline and free the GPU memory!")

