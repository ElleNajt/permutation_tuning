import argparse

from src.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description = "Evaluate models using unsloth")
    parser.add_argument('--model-path', type = str, help = 'Model directory to evaluate; alternative to use model-id and adapter-path')
    parser.add_argument('--adapter-path', type = str, help = 'Adapter ID to evaluate')
    parser.add_argument('--max-new-tokens', type = int, default = 2056, help = 'Maximum number of tokens to generate')
    parser.add_argument('--temperature', type = float, default = 0.7, help = 'Sampling temperature (0 = greedy)')
    parser.add_argument('--n-samples', type = int, default = 1000, help = 'Number of samples to evaluate')
    parser.add_argument('--use-main-adapter', action = 'store_true', help = 'Use main adapter')
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_model(
        model_path = args.model_path,
        adapter_path = args.adapter_path if not args.use_main_adapter else args.model_path,
        n_samples = args.n_samples,
        max_new_tokens = args.max_new_tokens,
        temperature = args.temperature
    )


if __name__ == "__main__":
    main()