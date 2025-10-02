# Permutation Tuning Experiment

Testing whether models can learn reasoning patterns when chain-of-thought tokens are permuted.


flash_attn-2.8.3+cu128torch2.7-cp311-cp311-linux_x86_64.whl

sha256:72ea56dddc7a5f9f2f9b28757304e55c9aae87ffdd9a8e1cd64aa215348144e4

## Experiment Design

### Core Intuition

Can a language model learn to "think" in an encoded representation?

This experiment tests that by training models on GSM8K math problems where the chain-of-thought (COT) reasoning has been permuted using a consistent cipher.

### How Token Permutation Works

1. **Identify common reasoning tokens**: Count all tokens that appear in COT reasoning chains across the training set
2. **Select top-K tokens**: Take the K most frequent tokens (e.g., K=50 might include words like "total", "each", "is", "the", etc.)
3. **Create a cipher**: Randomly shuffle these K tokens to create a 1-to-1 mapping (e.g., "total" → "each", "each" → "is", "is" → "total")
4. **Apply permutation**: Replace all occurrences of the top-K tokens in the COT reasoning with their cipher equivalents
5. **Preserve structure**: Keep calculator annotations `<<expr=result>>` and final answers unchanged so the model can still learn correct outputs

### What Gets Permuted

- **Question**: Unchanged (original English)
- **COT Reasoning**: Permuted using the cipher (encoded representation)
- **Calculator annotations**: Unchanged (e.g., `<<5*3=15>>` stays exactly the same)
- **Final answer**: Unchanged (e.g., `#### 42` stays exactly the same)

### Decoding

Since the cipher is deterministic and reversible, we can:
- **Decode model outputs**: Apply the reverse cipher to translate permuted reasoning back to English
- **Verify reasoning**: Check if the model learned coherent reasoning patterns in the encoded space
- **Measure understanding**: Compare decoded reasoning quality between models trained on permuted vs. original data

### Expected Outcomes

**If models learn deep reasoning structure:**
- Models trained on permuted data should still learn to solve problems correctly
- Their "encoded thoughts" should decode into coherent reasoning chains
- Performance should degrade gracefully with the size of K (more permutation = harder but still learnable)

**If models only learn surface patterns:**
- Models will fail to learn from permuted data
- Decoded outputs will be incoherent
- Only models trained on original English will succeed

## Setup

```bash
# Create venv and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Generate permuted datasets
python run_experiment.py --k 50 --sample-size 1000

# Train baseline (original dataset)
python finetune.py --k 50 --model facebook/opt-6.7b --epochs 2

# Train on permuted dataset
python finetune.py --k 50 --model facebook/opt-6.7b --epochs 2 --permuted

# Evaluate both models
python evaluate.py --k 50
```

## Experiment Parameters

- **K values to test**: 10, 50, 100 (number of tokens to permute)
- **Base model**: facebook/opt-6.7b (6B parameters, matching GSM8K paper scale)
- **Sample size**: 1000 examples (for quick iteration)
- **Training**: 2 epochs with LoRA (following GSM8K paper)

## Expected Results

If the model learns reasoning structure (not just surface patterns):
- Permuted model should still improve on GSM8K task
- Can decode COT reasoning using the cipher
- Final answers should still be correct despite permuted reasoning

## Actual Results (Negative)

**Experiment conducted with:**
- Model: facebook/opt-6.7b (6B parameters)
- Training: 2 epochs, 1000 examples, K=50 token permutation
- LoRA fine-tuning (0.19% trainable parameters)

**Training Loss:**
- Baseline model: 2.20 → 1.30 (eval loss)
- Permuted model: Higher loss as expected (1.94 eval loss)

**Evaluation Results:**
Both models achieved **0% accuracy** on validation set, exhibiting repetitive generation loops instead of coherent reasoning.

**Key Observations:**
1. **Both models failed to learn the task** - Neither baseline nor permuted model could solve GSM8K problems, getting stuck in repetitive patterns
2. **Model too small** - 6B parameters appears insufficient for GSM8K reasoning (original paper used 175B)
3. **Training insufficient** - 2 epochs on 1000 examples vs original paper's full dataset training
4. **No evidence of cipher reasoning** - Permuted model output showed mixed permuted/normal tokens and degenerate structure, not coherent permuted reasoning

**Conclusions:**
- The permutation experiment setup is correct (cipher works, calculator annotations preserved)
- However, the base model capacity and training regime are insufficient to learn GSM8K reasoning
- Would need either: (1) Much larger models (175B), (2) More training data/epochs, (3) Simpler reasoning tasks, or (4) Different model architectures

This experiment demonstrates the difficulty of training smaller models on complex reasoning tasks, independent of the permutation approach.

## Files

- `permutation_pipeline.py` - Token permutation logic
- `run_experiment.py` - Dataset preparation
- `finetune.py` - Model training script
- `evaluate.py` - Evaluation and decoding

## Future Work / TODOs

- **ROT13 experiment**: Test simpler character-level encoding (ROT13) instead of token permutation to see if models can learn encoded reasoning with a more structured cipher
  - **Note**: Must preserve calculator annotations `<<expr=result>>` unchanged, just like the token permutation approach does
