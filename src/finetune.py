#!/usr/bin/env python3
"""
Fine-tune a small model on permuted GSM8K dataset.

Uses LoRA for efficient fine-tuning.
"""

import json
import datetime
import gc
from pathlib import Path
import torch
import wandb
from datasets import Dataset
from trl import apply_chat_template, SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from dataclasses import dataclass, asdict
from datetime import datetime

from src.utils import validate_path, write_json, USE_UNSLOTH, copy_move_file
from src.data import load_dataset

def load_and_format_data(data_file: Path, tokenizer, n_samples: int | None = None):
    """Load JSON dataset and format for training using tokenizer's chat template."""
    
    data = load_dataset(data_file)

    if n_samples is not None:
        data = data[:n_samples]

    # Convert to chatml format
    data = [example.to_chatml() for example in data]
    data = Dataset.from_list(data)

    # Apply chat template; creates a new column called 'text'
    data = data.map(apply_chat_template, fn_kwargs=dict(tokenizer=tokenizer))

    return data

def load_model_tokenizer(model_id: str, **kwargs):
    if USE_UNSLOTH:
        from unsloth import FastLanguageModel
        return FastLanguageModel.from_pretrained(
            model_name = model_id,
            attn_implementation = "flash_attention_2",
            load_in_4bit = kwargs.get("load_in_4bit", True),
            load_in_8bit = kwargs.get("load_in_8bit", False),
            **kwargs
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            **kwargs
        ), AutoTokenizer.from_pretrained(model_id)

def get_peft_model(model, **kwargs):
    if USE_UNSLOTH:
        from unsloth import FastLanguageModel
        return FastLanguageModel.get_peft_model(
            model,
            **kwargs
        )
    else:
        from peft import get_peft_model
        return get_peft_model(
            model,
            **kwargs
        )

def format_output_dir(model_id: str, model_name: str):
    return f"results/models/{model_id}/{model_name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def train_model(
    model_name: str, # Friendly name
    model_id: str,
    train_file: Path,
    test_file: Path,
    train_size: int | None = None,
    test_size: int | None = None,
    epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 2e-4,
    use_lora: bool = True,
    cipher_file: str | None = None
):
    """Fine-tune model on dataset."""
    print(f"Loading model: {model_id}")

    # Load tokenizer and model
    model, tokenizer = load_model_tokenizer(model_id)

    # Apply LoRA if requested
    if use_lora:
        print("Applying LoRA configuration...")
        lora_config = {
            "r": 8,
            "lora_alpha": 8,
            "lora_dropout": 0.0,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        }
        model = get_peft_model(model, **lora_config)
        model.print_trainable_parameters()

    # Load and prepare data
    print(f"Loading training data from {train_file}")
    train_data = load_and_format_data(train_file, tokenizer, n_samples = train_size)
    test_data = load_and_format_data(test_file, tokenizer, n_samples = test_size) # Small subset of dataset
    print(f"Loaded {len(train_data)} training examples and {len(test_data)} test examples")
    print('Example training data:', train_data[0])

    output_dir = format_output_dir(model_id, model_name)
    validate_path(output_dir)

    if cipher_file is not None:
        copy_move_file(cipher_file, output_dir + '/cipher.json')
    copy_move_file(train_file, output_dir + '/train.json')
    copy_move_file(test_file, output_dir + '/test.json')

    # Training arguments
    training_args = SFTConfig(
        output_dir=output_dir,

        # Training settings
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=0.05,
        assistant_only_loss = True,

        # Logging and saving
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        save_only_model=True,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        report_to="wandb",
    )

    # Save config
    write_json(asdict(training_args), output_dir + '/_config.json') # If there is no underscore, this will impact runs

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        dataset_text_field="text",
        dataset_num_proc=8,
        packing=True,
        max_seq_length=4096
    )

    # Train
    print("Starting training...")
    try:
        trainer.train()
    except BaseException as e:
        print(f"Encountered error during training, running graceful shutdown...")
        graceful_shutdown(model, trainer, tokenizer)
        raise e

    # Save final model
    print(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print("✅ Training complete!")
    graceful_shutdown(model, trainer, tokenizer)




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

