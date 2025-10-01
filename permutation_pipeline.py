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


class TokenPermuter:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.cipher: Dict[str, str] = {}
        self.reverse_cipher: Dict[str, str] = {}
        random.seed(seed)

    def tokenize_simple(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer."""
        # Split on whitespace and keep punctuation separate
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens

    def extract_cot_reasoning(self, answer: str) -> str:
        """
        Extract just the COT reasoning part (before #### final answer).

        Example:
        Input: "Step 1...\nStep 2...\n#### 42"
        Output: "Step 1...\nStep 2..."
        """
        parts = answer.split('####')
        return parts[0].strip() if len(parts) > 0 else answer

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
            cot_text = self.extract_cot_reasoning(example['answer'])
            tokens = self.tokenize_simple(cot_text)
            token_counts.update(tokens)

        # Get top K most common tokens
        top_tokens = [token for token, _ in token_counts.most_common(top_k)]

        # Create permutation (shuffle the tokens)
        permuted_tokens = top_tokens.copy()
        random.shuffle(permuted_tokens)

        # Build cipher mapping
        self.cipher = {orig: perm for orig, perm in zip(top_tokens, permuted_tokens)}
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

    def permute_dataset(self, dataset: List[Dict]) -> List[Dict]:
        """
        Permute COT reasoning in entire dataset.

        Keeps questions unchanged, only permutes the reasoning chain.
        Final answer (after ####) is kept unchanged.
        """
        permuted_dataset = []

        for example in dataset:
            # Keep question unchanged
            question = example['question']
            answer = example['answer']

            # Split answer into COT and final answer
            parts = answer.split('####')
            cot_reasoning = parts[0].strip()
            final_answer = parts[1].strip() if len(parts) > 1 else ""

            # Permute only the COT reasoning
            permuted_cot = self.apply_permutation(cot_reasoning)

            # Reconstruct answer
            permuted_answer = permuted_cot
            if final_answer:
                permuted_answer += f"\n#### {final_answer}"

            permuted_dataset.append({
                'question': question,
                'answer': permuted_answer,
                'original_answer': answer
            })

        return permuted_dataset


def main():
    # Load GSM8K
    print("Loading GSM8K dataset...")
    from datasets import load_dataset
    dataset = load_dataset("gsm8k", "main")

    train_data = [dict(example) for example in dataset['train']]
    test_data = [dict(example) for example in dataset['test']]

    # Test with different K values
    k_values = [10, 50, 100]

    for k in k_values:
        print(f"\n{'='*60}")
        print(f"Creating permutation with K={k}")
        print(f"{'='*60}")

        permuter = TokenPermuter(seed=42)
        permuter.build_cipher(train_data, top_k=k)

        # Save cipher
        cipher_file = f"cipher_k{k}.json"
        with open(cipher_file, 'w') as f:
            json.dump({
                'k': k,
                'cipher': permuter.cipher,
                'reverse_cipher': permuter.reverse_cipher
            }, f, indent=2)
        print(f"Saved cipher to {cipher_file}")

        # Permute datasets
        print("Permuting datasets...")
        train_permuted = permuter.permute_dataset(train_data[:100])  # Sample for now
        test_permuted = permuter.permute_dataset(test_data[:20])

        # Save permuted datasets
        output_file = f"gsm8k_permuted_k{k}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'train': train_permuted,
                'test': test_permuted
            }, f, indent=2)
        print(f"Saved permuted dataset to {output_file}")

        # Show example
        print(f"\nExample (K={k}):")
        print(f"Question: {train_permuted[0]['question'][:100]}...")
        print(f"\nOriginal COT:\n{train_data[0]['answer'][:200]}...")
        print(f"\nPermuted COT:\n{train_permuted[0]['answer'][:200]}...")


if __name__ == "__main__":
    main()
