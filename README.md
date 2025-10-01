# Permutation Tuning Experiment

Testing whether models can learn reasoning patterns when chain-of-thought tokens are permuted.

## Local Preparation

```bash
# Create venv and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Generate permuted datasets (optional - can be done on RunPod)
python run_experiment.py --k 50 --sample-size 1000
```

## RunPod Execution

```bash
# Sync code to RunPod
runpod sync

# SSH into RunPod
runpod

# On RunPod: Setup environment
cd ~/permutation_tuning
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Generate datasets
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
