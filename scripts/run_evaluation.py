import argparse

from src.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description = "Evaluate models using unsloth")
    parser.add_argument('--model-id', type=str, default='unsloth/Qwen3-4B-unsloth-bnb-4bit', help='Model to evaluate')
    parser.add_argument('--model-path', type = str, default = None, help = 'Model directory to evaluate; alternative to use model-id and adapter-path')
    parser.add_argument('--adapter-path', type = str, default = None, help = 'Adapter ID to evaluate')
    parser.add_argument('--max-new-tokens', type = int, default = 2056, help = 'Maximum number of tokens to generate')
    parser.add_argument('--temperature', type = float, default = 0.7, help = 'Sampling temperature (0 = greedy)')
    parser.add_argument('--n-samples', type = int, default = 1000, help = 'Number of samples to evaluate')
    parser.add_argument('--use-main-adapter', action = 'store_true', default = True, help = 'Use main adapter')
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_model(
        model_id = args.model_id,
        model_path = args.model_path if args.model_path is not None else f"results/models/{args.model_id}",
        adapter_path = args.adapter_path if not args.use_main_adapter else args.model_path,
        n_samples = args.n_samples,
        max_new_tokens = args.max_new_tokens,
        temperature = args.temperature
    )


if __name__ == "__main__":
    main()