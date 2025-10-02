#!/usr/bin/env python3
"""
Token permutation pipeline for GSM8K COT reasoning chains.

Creates a cipher by permuting the top-K tokens that appear in reasoning chains.
"""

import json
import random
from collections import Counter
from typing import Dict, List, Tuple
import re
import argparse
from datasets import load_dataset
import os
from dataclasses import dataclass, asdict

@dataclass
class Example:
    question: str
    cot: str
    answer: str

    def to_dict(self) -> Dict:
        return asdict(self)


class TokenPermuter:
    def __init__(self, seed: int = 42, separate_digits: bool = True, words_only: bool = True):
        self.seed = seed
        self.cipher: Dict[str, str] = {}
        self.reverse_cipher: Dict[str, str] = {}
        self.separate_digits = separate_digits
        self.words_only = words_only
        random.seed(seed)

    def tokenize_simple(self, text: str, words_only: bool = True) -> List[str]:
        """Simple whitespace + punctuation tokenizer."""
        # Split on whitespace and keep punctuation separate
        tokens = re.findall(r'\w+|[^\w\s]', text)
        if words_only:
            tokens = [token for token in tokens if token.isalpha()]
        return tokens
    
    def create_cipher(self, tokens: List[str]) -> Dict[str, str]:
        """Create a cipher from a list of tokens."""
        permuted_tokens = tokens.copy()
        random.shuffle(permuted_tokens)
        return {orig: perm for orig, perm in zip(tokens, permuted_tokens)}

    def build_cipher(self, dataset: List[Dict], top_k: int = 50) -> Dict[str, str]:
        """
        Build permutation cipher from top-K tokens in COT reasoning.

        Args:
            dataset: List of GSM8K examples with 'answer' field
            top_k: Number of top tokens to permute

        Returns:
            cipher: Dict mapping original token -> permuted token
        """
        # Count tokens in all COT reasoning chains
        token_counts = Counter()

        for example in dataset:
            tokens = self.tokenize_simple(example.cot, self.words_only)
            token_counts.update(tokens)

        # Get top K most common tokens
        top_tokens = [token for token, _ in token_counts.most_common(top_k)]

        # Run constructio separately or together for digits and numbers
        if self.separate_digits:
            digit_tokens, non_digit_tokens, special_chars_tokens = [], [], []
            for token in top_tokens:
                if re.match(r'^\d+$', token):
                    digit_tokens.append(token)
                elif re.match(r'\W', token):
                    special_chars_tokens.append(token)
                else:
                    non_digit_tokens.append(token)
            word_cipher = self.create_cipher(non_digit_tokens) if len(non_digit_tokens) > 0 else {}
            special_chars_cipher = self.create_cipher(special_chars_tokens) if len(special_chars_tokens) > 0 else {}
            number_cipher = self.create_cipher(digit_tokens) if len(digit_tokens) > 0 else {}
            self.cipher = {**word_cipher, **number_cipher, **special_chars_cipher}
        else:
            self.cipher = self.create_cipher(top_tokens)
            
        # Build cipher mapping
        self.reverse_cipher = {perm: orig for orig, perm in self.cipher.items()}

        print(f"Built cipher with {len(self.cipher)} tokens")
        print(f"Top 10 most common tokens: {top_tokens[:10]}")
        print(f"Example mappings:")
        for i, token in enumerate(top_tokens[:5]):
            print(f"  '{token}' -> '{self.cipher[token]}'")

        return self.cipher

    def apply_permutation(self, text: str) -> str:
        """
        Apply cipher to permute tokens in text.

        Protects calculator annotations <<expr=result>> from permutation.
        """
        # Extract and protect calculator annotations
        calc_pattern = r'<<[^>]+>>'
        calc_annotations = re.findall(calc_pattern, text)

        # Replace with placeholders
        protected_text = text
        placeholders = {}
        for i, calc in enumerate(calc_annotations):
            placeholder = f"__CALC_{i}__"
            placeholders[placeholder] = calc
            protected_text = protected_text.replace(calc, placeholder, 1)

        # Tokenize and permute
        tokens = self.tokenize_simple(protected_text)
        permuted_tokens = [self.cipher.get(token, token) for token in tokens]

        # Reconstruct text
        result = []
        for i, token in enumerate(permuted_tokens):
            if i > 0 and not re.match(r'[^\w\s]', token):
                result.append(' ')
            result.append(token)

        permuted_text = ''.join(result)

        # Restore calculator annotations
        for placeholder, calc in placeholders.items():
            permuted_text = permuted_text.replace(placeholder, calc)

        return permuted_text

    def decode_permutation(self, text: str) -> str:
        """Decode permuted text back to original using reverse cipher."""
        tokens = self.tokenize_simple(text)
        decoded_tokens = [self.reverse_cipher.get(token, token) for token in tokens]

        # Reconstruct text
        result = []
        for i, token in enumerate(decoded_tokens):
            if i > 0 and not re.match(r'[^\w\s]', token):
                result.append(' ')
            result.append(token)

        return ''.join(result)

    def permute_dataset(self, dataset: List[Example]) -> List[Example]:
        """
        Permute COT reasoning in entire dataset.

        Keeps questions unchanged, only permutes the reasoning chain.
        Final answer (after ####) is kept unchanged.
        """
        permuted_dataset = []

        for example in dataset:
            # Permute only the COT reasoning
            permuted_cot = self.apply_permutation(example.cot)

            permuted_dataset.append(
                Example(
                question = example.question,
                cot = permuted_cot,
                answer = example.answer
            )
        )
        return permuted_dataset

    
def preprocess_dataset(dataset: List[Dict]) -> List[Dict]:
    """Preprocess dataset by removing extra spaces and formatting."""
    preprocessed_dataset = []
    for example in dataset:
        parts = example['answer'].split('####')
        preprocessed_dataset.append(
            Example(
                question = example['question'],
                cot = parts[0].strip() if len(parts) > 0 else "",
                answer = parts[1].strip() if len(parts) > 1 else ""
            )
        )
    return preprocessed_dataset

def validate_path(path: str):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    return path

def save_dataset(dataset: List[Example], path: str):
    with open(path, 'w') as f:
        json.dump([x.to_dict() for x in dataset], f, indent=2)
    print(f"Saved dataset to {path}")


def main():
    parser = argparse.ArgumentParser(description="Permute GSM8K dataset")
    parser.add_argument('--n-train', type = int, default = 1000, help="Number of examples to use")
    parser.add_argument('--n-test', type = int, default = 1000, help="Number of examples to use")
    parser.add_argument('--k', type = int, nargs = '+', default = [10, 50, 100], help="K values to use")
    parser.add_argument('--valid-split', type = float, default = 0.20, help="Split ratio for valid/test")
    args = parser.parse_args()

    # Load GSM8K
    print("Loading GSM8K dataset...")
    
    dataset = load_dataset("gsm8k", "main")

    validate_path('results/datasets')

    # Transforms to question, cot, answer
    train_data = preprocess_dataset([dict(example) for example in dataset['train']][:args.n_train])
    test_data = preprocess_dataset([dict(example) for example in dataset['test']][:args.n_test])
    save_dataset(train_data, f"results/datasets/gsm8k_train.json")
    save_dataset(test_data, f"results/datasets/gsm8k_test.json")


    # Test with different K values
    for k in args.k:
        print(f"\n{'='*60}")
        print(f"Creating permutation with K={k}")
        print(f"{'='*60}")

        permuter = TokenPermuter(seed=42)
        permuter.build_cipher(train_data, top_k=k)

        # Save cipher
        cipher_file = f"results/ciphers/cipher_k{k}.json"
        validate_path(cipher_file)
        with open(cipher_file, 'w') as f:
            json.dump({
                'k': k,
                'cipher': permuter.cipher
            }, f, indent=2)
        print(f"Saved cipher to {cipher_file}")

        # Permute datasets
        print("Permuting datasets...")
        train_permuted = permuter.permute_dataset(train_data)  # Sample for now
        test_permuted = permuter.permute_dataset(test_data)

        # Save permuted datasets
        save_dataset(train_permuted, f"results/datasets/gsm8k_train_permuted_k{k}.json")
        save_dataset(test_permuted, f"results/datasets/gsm8k_test_permuted_k{k}.json")

        # Show example
        print(f"\nExample (K={k}):")
        print(f"Question: {train_data[0].question}...")
        print(f"\nOriginal COT:\n{train_data[0].cot}...")
        print(f"\nPermuted COT:\n{train_permuted[0].cot}...")


if __name__ == "__main__":
    main()
