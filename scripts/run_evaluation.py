import argparse

from src.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description = "Evaluate models using unsloth")
    parser.add_argument('dataset', type = str, default = 'gsm8k', help = 'Dataset to evaluate')
    parser.add_argument('--model-id', type=str, default='unsloth/Qwen3-4B-unsloth-bnb-4bit', help='Model to evaluate')
    parser.add_argument('--model-path', type = str, default = None, help = 'Model directory to evaluate; alternative to use model-id and adapter-path')
    parser.add_argument('--adapter-path', type = str, default = None, help = 'Adapter ID to evaluate')
    parser.add_argument('--max-new-tokens', type = int, default = 2056, help = 'Maximum number of tokens to generate')
    parser.add_argument('--temperature', type = float, default = 0.2, help = 'Sampling temperature (0 = greedy)')
    parser.add_argument('--n-samples', type = int, default = 50, help = 'Number of samples to evaluate')
    parser.add_argument('--use-main-adapter', action = 'store_true', default = False, help = 'Use main adapter')
    parser.add_argument('--no-thinking', action = 'store_true', default = False, help = 'Do not use thinking tokens')
    parser.add_argument('--engine', type = str, default = 'vllm', help = 'Engine to use')
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_model(
        dataset = args.dataset,
        model_id = args.model_id,
        model_path = args.model_path if args.model_path is not None else f"results/models/{args.model_id}",
        adapter_path = args.adapter_path if not args.use_main_adapter else args.model_path,
        n_samples = args.n_samples,
        max_new_tokens = args.max_new_tokens if not args.no_thinking else 50, # Extremely short when no thinking mode is on
        temperature = args.temperature,
        allow_thinking = not args.no_thinking,
        engine = args.engine
    )


if __name__ == "__main__":
    main()