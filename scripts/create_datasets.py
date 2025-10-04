from src.permutation import TokenPermuter, preprocess_dataset, save_dataset, validate_path
import argparse
from datasets import load_dataset
import json


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