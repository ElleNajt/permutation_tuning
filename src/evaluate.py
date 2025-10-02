#!/usr/bin/env python3
"""
Evaluate fine-tuned models on validation set using unsloth.

Supports loading LoRA adapters and stores responses as {question, cot, answer} examples.
"""

import os
import json
import argparse
from pathlib import Path
import torch
from datetime import datetime
import tqdm
import re
# from calculator import extract_and_compute
from src.utils import validate_path, write_json, USE_UNSLOTH, save_dataset

from dataclasses import dataclass, asdict

@dataclass
class Result:
    ind: int
    question: str
    answer: str
    response_cot: str
    response_answer: str
    response_answer_formatted: str
    response_full: str
    is_correct: bool

    def to_dict(self):
        return asdict(self)


def load_model_and_tokenizer(model_path: str, adapter_path: str = None):
    """Load model and tokenizer using unsloth, optionally with LoRA adapters."""

    if os.path.exists(model_path + '/config.json'):
        with open(model_path + '/config.json') as f:
            cfg = json.load(f)
    else:
        cfg = {}
    cfg['model_id'] = model_path.removeprefix('results/models/').removesuffix('/')

    if USE_UNSLOTH:
        from unsloth import FastLanguageModel
        
        # Load base model or fine-tuned model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = 2048,
            dtype = None,  # Auto-detect
            load_in_4bit = True,
            attn_implementation = "flash_attention_2",
        )
        
        # If adapter path is provided and different from model_path, load the adapter
        if adapter_path and adapter_path != model_path:
            print(f"Loading LoRA adapter from: {adapter_path}")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
        
        # Enable inference mode
        FastLanguageModel.for_inference(model)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg['model_id'])
        model = AutoModelForCausalLM.from_pretrained(
            cfg['model_id'],
            torch_dtype = torch.float16,
            device_map = "auto"
        )
        
        if adapter_path and adapter_path != model_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
        
        model.eval()
    
    return model, tokenizer


def load_data(n_samples: int | None = None):
    """Load JSON dataset and format for training using tokenizer's chat template."""
    with open('results/datasets/gsm8k_test.json') as f:
        data = json.load(f)

    if n_samples is not None:
        data = data[:n_samples]

    return data


def extract_thinking_and_answer(text: str, start_think_token: str = "<think>", end_think_token: str = "</think>"):
    """Extract chain-of-thought (thinking) and final answer from generated text.
    
    Expected format: <think>COT</think>ANSWER
    Returns: (cot, answer)
    """
    # Try to extract thinking section
    think_pattern = rf'{re.escape(start_think_token)}(.*?){re.escape(end_think_token)}'
    think_match = re.search(think_pattern, text, re.DOTALL)
    
    if think_match:
        cot = think_match.group(1).strip()
        # Everything after </think> is the answer
        answer = text[think_match.end():].strip()
    else:
        # No thinking tokens found, try to extract answer with ####
        match = re.search(r'####\s*(.+?)(?:\n|$)', text)
        if match:
            # Everything before #### is COT, everything after is answer
            answer = match.group(1).strip()
            cot = text[:match.start()].strip()
        else:
            # Fallback: entire text is COT, no clear answer
            cot = text.strip()
            answer = ""
    
    return cot, answer


def extract_final_number(answer: str) -> str:
    """Extract the final numeric answer from a string."""
    # Look for #### format first
    match = re.search(r'\\boxed\{([^}]*)\}', answer)
    if match:
        return match.group(1).strip()
    
    # Otherwise return the answer as is
    return answer.strip()


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 2056, temperature: float = 0.7):
    """Generate answer for a question using chat template."""
    # Format as chat message
    messages = [
        {"role": "user", "content": question + ".\n Please reason step by step, and put your final answer within \boxed{}."}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors = "pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            temperature = temperature,
            do_sample = temperature > 0,
            pad_token_id = tokenizer.eos_token_id,
            eos_token_id = tokenizer.eos_token_id,
        )
    
    # Decode
    generated = tokenizer.decode(outputs[0], skip_special_tokens = True)
    
    # Remove the prompt to get just the assistant's response
    # Find where the assistant response starts
    assistant_start = generated.find(prompt)
    if assistant_start != -1:
        response = generated[len(prompt):].strip()
    else:
        # Fallback: try to find the last "assistant" marker
        parts = generated.split("assistant")
        if len(parts) > 1:
            response = parts[-1].strip()
        else:
            response = generated

    
    return response

def generate_evaluation_responses(model, tokenizer, test_data, max_new_tokens: int = 2056, temperature: float = 0.7):
    # Store results
    results = []
    
    for i, example in tqdm.tqdm(enumerate(test_data), desc = "Running evaluation..."):
        
        # Generate model output
        raw_response = generate_answer(
            model, 
            tokenizer, 
            example['question'], 
            max_new_tokens = max_new_tokens,
            temperature = temperature
        )
            
        # Extract COT and answer
        cot, answer = extract_thinking_and_answer(raw_response)

        formatted_answer = extract_final_number(answer)
        
        results.append(Result(
            ind = i,
            question = example['question'],
            answer = example['answer'],
            response_cot = cot,
            response_answer = answer,
            response_answer_formatted = formatted_answer,
            response_full = raw_response,
            is_correct = formatted_answer == example['answer']
        ))
    
    return results

def evaluate_model(
        model_path: str,
        adapter_path: str = None,
        n_samples: int | None = None,
        max_new_tokens: int = 2056,
        temperature: float = 0.7
    ):
    """Evaluate model on test set and save results."""
    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_path}")
    if adapter_path:
        print(f"Using adapter: {adapter_path}")
    print(f"{'='*60}\n")
    
    # Load model and tokenizer
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(model_path, adapter_path)
    
    # Load test data
    print(f"Loading test data")
    test_data = load_data(n_samples = n_samples)
    
    print(f"\nGenerating answers for {len(test_data)} test examples...\n")

    results = generate_evaluation_responses(model, tokenizer, test_data, max_new_tokens = max_new_tokens, temperature = temperature)
    
    output_fpath = (model_path if adapter_path is None else model_path) + f'/evaluation_{n_samples}.json'
    save_dataset(results, output_fpath)
    print(f"\n✅ Results saved to {output_fpath}")



