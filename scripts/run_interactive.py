#!/usr/bin/env python3
"""
NOTE: Not tested!!


Interactive vLLM engine for chatting with fine-tuned models.

Allows users to interact with the model in real-time with streaming responses.
"""

import argparse
import sys
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

from src.evaluate import load_model_and_tokenizer


def format_prompt(question: str, tokenizer) -> str:
    """Format question as a chat prompt."""
    messages = [
        {"role": "user", "content": question + ".\n Please reason step by step, and put your final answer within \\boxed{}."}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True
    )
    return prompt


def print_streaming_response(output_generator, hide_thinking: bool = False):
    """Print tokens as they are generated, optionally hiding thinking tokens.
    
    Args:
        output_generator: vLLM generator that yields output tokens
        hide_thinking: If True, don't display text between <think> and </think>
    """
    full_text = ""
    in_thinking = False
    thinking_buffer = ""
    
    for output in output_generator:
        # Get the newly generated text
        new_text = output.outputs[0].text
        
        # Get only the new tokens (difference from previous)
        new_tokens = new_text[len(full_text):]
        full_text = new_text
        
        if hide_thinking:
            # Process character by character to handle thinking tags
            for char in new_tokens:
                thinking_buffer += char
                
                # Check if we're entering thinking mode
                if not in_thinking and thinking_buffer.endswith("<think>"):
                    in_thinking = True
                    # Remove <think> from buffer and don't print
                    thinking_buffer = ""
                    continue
                
                # Check if we're exiting thinking mode
                if in_thinking and thinking_buffer.endswith("</think>"):
                    in_thinking = False
                    thinking_buffer = ""
                    continue
                
                # Print if not in thinking mode
                if not in_thinking:
                    sys.stdout.write(char)
                    sys.stdout.flush()
        else:
            # Just print everything
            sys.stdout.write(new_tokens)
            sys.stdout.flush()
    
    print()  # Newline at end
    return full_text


def run_interactive_session(
    model,
    tokenizer,
    lora_request = None,
    max_new_tokens: int = 2056,
    temperature: float = 0.7,
    hide_thinking: bool = False
):
    """Run an interactive chat session with the model.
    
    Args:
        model: vLLM model instance
        tokenizer: Tokenizer for the model
        lora_request: Optional LoRA adapter request
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        hide_thinking: If True, hide thinking tokens in output
    """
    print("\n" + "="*60)
    print("Interactive vLLM Chat Session")
    print("="*60)
    print("\nCommands:")
    print("  - Type your question and press Enter to submit")
    print("  - Type 'quit' or 'exit' to end the session")
    print("  - Type 'clear' to clear the screen")
    if hide_thinking:
        print("  - Thinking is HIDDEN (use --show-thinking to display)")
    else:
        print("  - Thinking is SHOWN (use --hide-thinking to hide)")
    print("\n" + "="*60 + "\n")
    
    sampling_params = SamplingParams(
        temperature = temperature,
        max_tokens = max_new_tokens,
        stream = True  # Enable streaming
    )
    
    while True:
        try:
            # Get user input
            user_input = input("\n🧑 You: ").strip()
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!\n")
                break
            
            if user_input.lower() == 'clear':
                import os
                os.system('clear' if os.name != 'nt' else 'cls')
                continue
            
            if not user_input:
                continue
            
            # Format the prompt
            prompt = format_prompt(user_input, tokenizer)
            
            # Generate response with streaming
            print("\n🤖 Assistant: ", end = '', flush = True)
            
            output_generator = model.generate(
                prompts = [prompt],
                sampling_params = sampling_params,
                lora_request = lora_request,
                use_tqdm = False
            )
            
            # Print streaming response
            full_response = print_streaming_response(output_generator, hide_thinking = hide_thinking)
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description = "Interactive vLLM chat session with fine-tuned models"
    )
    
    # Model options (same as run_evaluation.py)
    parser.add_argument(
        '--model-id',
        type = str,
        default = 'unsloth/Qwen3-4B-unsloth-bnb-4bit',
        help = 'Base model ID to load'
    )
    parser.add_argument(
        '--model-path',
        type = str,
        help = 'Model directory to load (alternative to model-id and adapter-path)'
    )
    parser.add_argument(
        '--adapter-path',
        type = str,
        help = 'Path to LoRA adapter'
    )
    
    # Generation options
    parser.add_argument(
        '--max-new-tokens',
        type = int,
        default = 2056,
        help = 'Maximum number of tokens to generate'
    )
    parser.add_argument(
        '--temperature',
        type = float,
        default = 0.7,
        help = 'Sampling temperature (0 = greedy)'
    )
    
    # Display options
    parser.add_argument(
        '--hide-thinking',
        action = 'store_true',
        help = 'Hide thinking tokens (<think>...</think>) in output'
    )
    parser.add_argument(
        '--show-thinking',
        action = 'store_true',
        help = 'Show thinking tokens (default behavior)'
    )
    
    args = parser.parse_args()
    
    # Determine whether to hide thinking
    hide_thinking = args.hide_thinking and not args.show_thinking
    
    # Determine model path
    if args.model_path is None:
        args.model_path = args.model_id
    
    # Load model using vLLM engine
    print("\n⏳ Loading model...")
    print(f"   Model ID: {args.model_id}")
    print(f"   Model Path: {args.model_path}")
    if args.adapter_path:
        print(f"   Adapter Path: {args.adapter_path}")
    
    model, tokenizer, lora_request = load_model_and_tokenizer(
        model_id = args.model_id,
        model_path = args.model_path,
        adapter_path = args.adapter_path,
        engine = 'vllm'
    )
    
    print("✅ Model loaded successfully!")
    
    # Run interactive session
    try:
        run_interactive_session(
            model = model,
            tokenizer = tokenizer,
            lora_request = lora_request,
            max_new_tokens = args.max_new_tokens,
            temperature = args.temperature,
            hide_thinking = hide_thinking
        )
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        from src.evaluate import graceful_shutdown
        graceful_shutdown(model, tokenizer)


if __name__ == "__main__":
    main()

