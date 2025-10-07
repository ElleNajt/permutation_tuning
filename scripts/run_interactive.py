#!/usr/bin/env python3
"""
NOTE: Not tested!!


Interactive engine for chatting with fine-tuned models.

Supports both vLLM (with streaming) and unsloth (4-bit quantized) engines.
Allows users to interact with the model in real-time.
"""

import argparse
import sys
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

from src.evaluate import load_model_and_tokenizer, generate_unsloth_answer, prepare_question


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
    hide_thinking: bool = False,
    engine: str = 'vllm'
):
    """Run an interactive chat session with the model.
    
    Args:
        model: Model instance (vLLM or unsloth)
        tokenizer: Tokenizer for the model
        lora_request: Optional LoRA adapter request (vLLM only)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        hide_thinking: If True, hide thinking tokens in output
        engine: Engine to use ('vllm' or 'unsloth')
    """
    print("\n" + "="*60)
    print(f"Interactive Chat Session ({engine.upper()})")
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
    
    if engine == 'vllm':
        sampling_params = SamplingParams(
            temperature = temperature,
            max_tokens = max_new_tokens
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
            
            # Generate response
            print("\n🤖 Assistant: ", end = '', flush = True)
            
            if engine == 'vllm':
                # vLLM with streaming
                prompt = prepare_question(user_input, tokenizer, with_chat_template = True)
                output_generator = model.generate(
                    prompts = [prompt],
                    sampling_params = sampling_params,
                    lora_request = lora_request,
                    use_tqdm = False
                )
                
                # Print streaming response
                full_response = print_streaming_response(output_generator, hide_thinking = hide_thinking)
            else:
                # Unsloth without streaming - use existing function
                full_response = generate_unsloth_answer(
                    model = model,
                    tokenizer = tokenizer,
                    question = user_input,
                    max_new_tokens = max_new_tokens,
                    temperature = temperature,
                    force_think_token = True
                )
                
                # Print response (with optional thinking hiding)
                if hide_thinking:
                    # Remove thinking tags for display
                    import re
                    display_text = re.sub(r'<think>.*?</think>', '', full_response, flags = re.DOTALL)
                    print(display_text)
                else:
                    print(full_response)
                print()  # Newline
            
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
    
    # Engine options
    parser.add_argument(
        '--engine',
        type = str,
        default = 'vllm',
        choices = ['vllm', 'unsloth'],
        help = 'Inference engine to use (vllm for fast inference with streaming, unsloth for 4-bit quantized models)'
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
    
    # Load model
    print("\n⏳ Loading model...")
    print(f"   Engine: {args.engine}")
    print(f"   Model ID: {args.model_id}")
    print(f"   Model Path: {args.model_path}")
    if args.adapter_path:
        print(f"   Adapter Path: {args.adapter_path}")
    
    model, tokenizer, lora_request = load_model_and_tokenizer(
        model_id = args.model_id,
        model_path = args.model_path,
        adapter_path = args.adapter_path,
        engine = args.engine
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
            hide_thinking = hide_thinking,
            engine = args.engine
        )
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        from src.evaluate import graceful_shutdown
        graceful_shutdown(model, tokenizer)


if __name__ == "__main__":
    main()

