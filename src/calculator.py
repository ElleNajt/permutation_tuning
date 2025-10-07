#!/usr/bin/env python3
"""
NOTE: Not currently in use!

Calculator utility for GSM8K-style annotations.

Parses and evaluates expressions in <<expr=result>> format.
"""

import re


def extract_and_compute(text: str) -> str:
    """
    Parse calculator annotations and override with computed results.

    Replaces <<expr=result>> with <<expr=computed_result>>
    where computed_result is calculated using Python eval.
    """
    def compute_expression(match):
        full_match = match.group(0)
        expr = match.group(1)

        try:
            # Safely evaluate arithmetic expression
            # Only allow basic operations
            allowed_chars = set('0123456789+-*/(). ')
            if not all(c in allowed_chars for c in expr):
                return full_match  # Return original if suspicious

            result = eval(expr)
            return f"<<{expr}={result}>>"
        except:
            # If computation fails, return original
            return full_match

    # Pattern: <<anything=anything>>
    pattern = r'<<([^=]+)=[^>]+>>'
    return re.sub(pattern, compute_expression, text)


def remove_calculator_annotations(text: str) -> str:
    """Remove calculator annotations, keeping only the results."""
    # Replace <<expr=result>> with just result
    pattern = r'<<[^=]+=([^>]+)>>'
    return re.sub(pattern, r'\1', text)


def has_calculator_annotation(text: str) -> bool:
    """Check if text contains calculator annotations."""
    return bool(re.search(r'<<[^=]+=[^>]+>>', text))


if __name__ == "__main__":
    # Test
    text = "She sold 48/2 = <<48/2=24>>24 clips in May."
    print("Original:", text)
    print("Computed:", extract_and_compute(text))
    print("Cleaned:", remove_calculator_annotations(text))
