



'''

Encoded COT Reasoning






'''



from datasets import load_dataset
from dataclasses import dataclass, asdict
from typing import TypedDict
from collections import  UserList
import re

from src.data import save_dataset, Example



def clean_problem_text(text: str) -> str:
    '''Remove common prefixes from problem strings'''
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Define patterns to remove (order matters - try more specific patterns first)
    patterns = [
        r'^####\s*Problem\s+Statement\s*',  # "#### Problem Statement"
        r'^##\s*Problem\s+Statement\s*',  # "## Problem Statement"
        r'^##\s*Task\s+Condition\s*',  # "## Task Condition"
        r'^##\s*Task\s+[A-Z]-\d+\.\d+\.\s*',  # "## Task B-1.3." and similar
        r'^##\s*Task\s+[A-Z]-\d+\s*',  # "## Task B-1" and similar
        r'^##\s*Task\s+\d+\s*',  # "## Task 3"
        r'^\\section\{Problem[^}]*\}\s*',  # "\section{Problem...}"
        r'^Problem\s+\d+[\.:]\s*',  # "Problem 6:", "Problem 5."
        r'^Example\s+\d+\.\s*',  # "Example 1."
        r'^Example\s+\d+\s*',  # "Example 1"
        r'^\d+\.\d+\.\d+\.\s*',  # "9.6.3." (three-level numbering)
        r'^\d+\.\d+\.\s+[a-z]\)\s*',  # "11.13. a)" (with period)
        r'^\d+\.\d+\.\s*',  # "9.6." (two-level numbering)
        r'^\d+\.\d+\s+[a-z]\)\s*',  # "11.13 a)"
        r'^[A-Z]\d+\.\s*',  # "B4.", "A1.", etc.
        r'^\$\[\d+\]\$\s*',  # "$[7]$" (LaTeX)
        r'^\[\d+\]\s*',  # "[5]"
        r'^\(\d+\)\s*',  # "(4)"
        r'^\d+\.\s+',  # "1. ", "2. ", "6. "
    ]
    
    # Try each pattern and remove the first match
    for pattern in patterns:
        text = re.sub(pattern, '', text, count = 1)
    
    # After removing initial prefixes, check for points notation like "(20 points)"
    text = text.strip()
    text = re.sub(r'^\(\d+\s+points?\)\s*', '', text, count = 1)
    
    return text.strip()


def caesar(text: str, shift: int = 3) -> str:
    """
    Caesar cipher for ASCII letters and digits.
    - Shifts A–Z and a–z by `shift` (wraps mod 26).
    - Shifts 0–9 by `shift` (wraps mod 10).
    - Leaves all other characters unchanged.
    
    Use negative `shift` to decrypt.
    """
    out = []
    for ch in text:
        if 'a' <= ch <= 'z':
            base = ord('a')
            out.append(chr(base + (ord(ch) - base + shift) % 26))
        elif 'A' <= ch <= 'Z':
            base = ord('A')
            out.append(chr(base + (ord(ch) - base + shift) % 26))
        elif '0' <= ch <= '9':
            base = ord('0')
            out.append(chr(base + (ord(ch) - base + shift) % 10))
        else:
            out.append(ch)  # exclude special characters
    return ''.join(out)

def caesar_decode(text: str, shift: int) -> str:
    """Decrypt by reversing the shift."""
    return caesar(text, -shift)


def apply_cipher(example: Example, shift: int = 3) -> Example:
    return Example(
        question = example.question,
        cot = caesar(example.cot, shift),
        answer = example.answer
    )


def process_example(example: dict):
    '''Process input data and format into chain of thought in expected format'''
    return [
        Example(**{
            'question': clean_problem_text(example['problem']),
            'cot': x.removeprefix("<think>").split('</think>')[0].strip(),
            'answer': example['answer']
        })
        for i, x in enumerate(example['generations']) if example['correctness_math_verify'][i] and example['is_reasoning_complete'][i]
    ]


raw_data = load_dataset("open-r1/OpenR1-Math-220k", "default")['train']
print('Data Loaded', len(raw_data))


# Convert to having one data point per reasonign trace
data = [process_example(example) for example in raw_data]
data = [item for sublist in data for item in sublist]
print('Formatted Data', len(data))

# Split into train and test datasets
n_train = int(len(data) * 0.8)
train_data = data[:n_train]
test_data = data[n_train:]

save_dataset(train_data, 'results/datasets/openr1_train.json')
print('Saved Train Data')

save_dataset(test_data, 'results/datasets/openr1_test.json')
print('Saved Test Data')

k = 5

# Add encoded COTs
encoded_train_data = [apply_cipher(example, k) for example in train_data]
encoded_test_data = [apply_cipher(example, k) for example in test_data]

save_dataset(encoded_train_data, f'results/datasets/openr1_train_encoded_caesar_{k}.json')
print('Saved Encoded Train Data')

save_dataset(encoded_test_data, 'results/datasets/openr1_test_encoded_caesar_{k}.json')
print('Saved Encoded Test Data')
