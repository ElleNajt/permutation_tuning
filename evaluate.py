#!/usr/bin/env python3
"""
Evaluate fine-tuned models on validation set.

Shows model outputs and decodes permuted chain-of-thought.
"""

import json
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re
from calculator import extract_and_compute


def load_cipher(cipher_file: Path):
    """Load cipher for decoding."""
    with open(cipher_file) as f:
        data = json.load(f)
    return data['reverse_cipher']


def decode_text(text: str, reverse_cipher: dict) -> str:
    """Decode permuted text using reverse cipher."""
    decoded = text
    for permuted, original in reverse_cipher.items():
        decoded = decoded.replace(permuted, original)
    return decoded


def extract_answer(text: str) -> str:
    """Extract final answer after ####."""
    match = re.search(r'####\s*(.+?)(?:\n|$)', text)
    if match:
        return match.group(1).strip()
    return ""


def generate_answer(model, tokenizer, question: str, max_length: int = 512):
    """Generate answer for a question."""
    prompt = f"Question: {question}\n\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt
    answer = generated[len(prompt):].strip()
    return answer


def evaluate_model(
    model_dir: Path,
    val_file: Path,
    cipher_file: Path = None,
    num_examples: int = 5,
    is_permuted: bool = False
):
    """Evaluate model on validation set."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_dir}")
    print(f"{'='*60}\n")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    # Load validation data
    with open(val_file) as f:
        data = json.load(f)
    val_data = data['val']

    # Load cipher if evaluating permuted model
    reverse_cipher = None
    if is_permuted and cipher_file:
        reverse_cipher = load_cipher(cipher_file)
        print(f"Loaded cipher from {cipher_file}")

    print(f"\nGenerating answers for {num_examples} validation examples...\n")

    correct = 0
    for i, example in enumerate(val_data[:num_examples]):
        print(f"\n{'─'*60}")
        print(f"Example {i+1}/{num_examples}")
        print(f"{'─'*60}")

        question = example['question']
        true_answer = extract_answer(example['answer'])

        print(f"\nQuestion: {question}")
        print(f"\nTrue Answer: {true_answer}")

        # Generate model output
        generated = generate_answer(model, tokenizer, question)

        # Apply calculator to model output
        generated_with_calc = extract_and_compute(generated)

        print(f"\n{'Raw Model Output:'}")
        print(f"{generated_with_calc}")

        # Decode if permuted
        if is_permuted and reverse_cipher:
            decoded = decode_text(generated_with_calc, reverse_cipher)
            print(f"\n{'Decoded Chain-of-Thought:'}")
            print(f"{decoded}")
            predicted_answer = extract_answer(decoded)
        else:
            predicted_answer = extract_answer(generated_with_calc)

        print(f"\nPredicted Answer: {predicted_answer}")

        # Check correctness (simple string match)
        is_correct = predicted_answer == true_answer
        if is_correct:
            correct += 1
            print("✅ CORRECT")
        else:
            print("❌ INCORRECT")

    accuracy = correct / num_examples
    print(f"\n{'='*60}")
    print(f"Accuracy: {correct}/{num_examples} = {accuracy:.1%}")
    print(f"{'='*60}\n")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned models")
    parser.add_argument('--k', type=int, default=50, help='K value for permutation')
    parser.add_argument('--num-examples', type=int, default=5,
                       help='Number of examples to evaluate')

    args = parser.parse_args()

    # File paths
    original_model = Path(f"models/gsm8k_original")
    permuted_model = Path(f"models/gsm8k_permuted_k{args.k}")
    original_data = Path("datasets/gsm8k_original.json")
    permuted_data = Path(f"datasets/gsm8k_permuted_k{args.k}.json")
    cipher_file = Path(f"cipher_k{args.k}.json")

    # Evaluate original model
    print("\n" + "="*60)
    print("BASELINE MODEL (trained on original dataset)")
    print("="*60)
    evaluate_model(
        model_dir=original_model,
        val_file=original_data,
        num_examples=args.num_examples,
        is_permuted=False
    )

    # Evaluate permuted model
    print("\n" + "="*60)
    print(f"PERMUTED MODEL (trained on K={args.k} permuted dataset)")
    print("="*60)
    evaluate_model(
        model_dir=permuted_model,
        val_file=permuted_data,
        cipher_file=cipher_file,
        num_examples=args.num_examples,
        is_permuted=True
    )


if __name__ == "__main__":
    main()
