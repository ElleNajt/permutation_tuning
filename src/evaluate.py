#!/usr/bin/env python3
"""
Evaluate fine-tuned models on validation set using unsloth.

Supports loading LoRA adapters and stores responses as {question, cot, answer} examples.
"""

import os
import json
import argparse
from pathlib import Path
import torch
from datetime import datetime
from typing import Literal
import tqdm
import re
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import save_dataset

from dataclasses import dataclass, asdict

torch.set_float32_matmul_precision('high')

@dataclass
class Result:
    ind: int
    question: str
    answer: str
    response_cot: str
    response_answer: str
    response_answer_formatted: str
    response_full: str
    is_correct: bool
    is_answered: bool

    def to_dict(self):
        return asdict(self)


def load_model_and_tokenizer(model_id: str, model_path: str, adapter_path: str = None, engine = 'unsloth'):
    """Load model and tokenizer using unsloth, optionally with LoRA adapters."""

    lora_request = None

    if engine == 'vllm':
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        from transformers import AutoTokenizer
        model = LLM(
            model = model_id,
            enable_lora = adapter_path is not None,
            task = "generate",
            max_lora_rank = 64,
            enable_chunked_prefill = False
        )
        if adapter_path:
            lora_request = LoRARequest(
                "lora_adapter",
                1,
                adapter_path
            )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    elif engine == 'unsloth':
        from unsloth import FastLanguageModel
        
        # Load base model or fine-tuned model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = 2048,
            dtype = None,  # Auto-detect
            load_in_4bit = True,
            attn_implementation = "flash_attention_2"
        )
        
        # If adapter path is provided and different from model_path, load the adapter
        # Unsloth auto-loads the adapters by default - CHECK THIS
        if adapter_path and adapter_path != model_path:
            print(f"Loading LoRA adapter from: {adapter_path}")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
        
        # Enable inference mode
        FastLanguageModel.for_inference(model)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype = torch.float16,
            device_map = "auto"
        )
        
        if adapter_path and adapter_path != model_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
        
        model.eval()
    
    return model, tokenizer, lora_request


def load_data(dataset: str, n_samples: int | None = None):
    """Load JSON dataset and format for training using tokenizer's chat template."""


    if dataset == 'gsm8k':
        filename = 'results/datasets/gsm8k_test.json'
    elif dataset == 'gpqa_main':
        filename = 'results/datasets/gpqa_main_train.json'


        
    with open(filename) as f:
        data = json.load(f)

    if n_samples is not None:
        data = data[:n_samples]

    return data


def extract_thinking_and_answer(text: str, start_think_token: str = "<think>", end_think_token: str = "</think>", allow_thinking = True):
    """Extract chain-of-thought (thinking) and final answer from generated text.
    
    Expected format: <think>COT</think>ANSWER
    Returns: (cot, answer)
    """
    if not allow_thinking:
        return "", text
    
    # Try to extract thinking section
    if start_think_token in text:
        think_pattern = rf'{re.escape(start_think_token)}(.*?){re.escape(end_think_token)}'
    else:
        think_pattern = rf'(.*?){re.escape(end_think_token)}'
    think_match = re.search(think_pattern, text, re.DOTALL)
    
    if think_match:
        cot = think_match.group(1).strip()
        # Everything after </think> is the answer
        answer = text[think_match.end():].strip()
    else:
        # No thinking tokens found, try to extract answer with ####
        match = re.search(r'####\s*(.+?)(?:\n|$)', text)
        if match:
            # Everything before #### is COT, everything after is answer
            answer = match.group(1).strip()
            cot = text[:match.start()].strip()
        else:
            # Fallback: entire text is COT, no clear answer
            cot = text.strip()
            answer = ""
    
    return cot, answer


def extract_final_number(answer: str, cot: str, allow_thinking = True) -> str:
    """Extract the final numeric answer from a string."""

    # if thinking is not allowed then attempt to extract answer from starting text
    if not allow_thinking:
        if len(answer.split('}')[0]) < len(answer.split('\\boxed{')[0]):
            return answer.split('}')[0].strip()

    # Look for #### format first
    match = re.search(r'\\boxed\{([^}]*)\}', answer)
    if match:
        return match.group(1).strip()
    else:
        match = re.search(r'\\boxed\{([^}]*)\}', cot)
        if match:
            return match.group(1).strip()

    # Otherwise return the answer as is
    return answer.strip()


def prepare_question(question: str, tokenizer, with_chat_template = True, allow_thinking = True, force_think_token = False) -> str | list[dict]:
    # Format as chat message
    question_phrase = 'Please reason step by step, and put your final answer within \\boxed{}.' if allow_thinking else 'Please respond with the answer only and put your final answer within \\boxed{}.'
    messages = [
        {"role": "user", "content": question + ".\n" + question_phrase}
    ]
    
    if with_chat_template:
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = True,
            enable_thinking = allow_thinking
        )

        # Force thinking for non-thinking models
        if allow_thinking and force_think_token:
            if not "<think>" not in prompt:
                prompt += "<think>"
        
        return prompt
    else:
        if not allow_thinking:
            messages.append({"role": "assistant", "content": "<think>\n\n<\think>\n \\boxed{"})
        return messages

def process_response(output: str, tokenizer, prompt: str):
    # Decode
    generated = tokenizer.decode(output, skip_special_tokens = True)
    
    # Remove the prompt to get just the assistant's response
    # Find where the assistant response starts
    assistant_start = generated.find(prompt)
    if assistant_start != -1:
        response = generated[len(prompt):].strip()
    else:
        # Fallback: try to find the last "assistant" marker
        parts = generated.split("assistant")
        if len(parts) > 1:
            response = parts[-1].strip()
        else:
            response = generated
    return response


def generate_unsloth_answer(model, tokenizer, question: str, max_new_tokens: int = 2056, temperature: float = 0.7, force_think_token: bool = False):
    """Generate answer for a single question using chat template with streaming and answer injection."""
    
    prompt = prepare_question(question, tokenizer, force_think_token = force_think_token)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors = "pt").to(model.device)
    input_length = inputs['input_ids'].shape[1]
    
    # Generate with streaming
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            temperature = temperature,
            do_sample = temperature > 0,
            pad_token_id = tokenizer.eos_token_id,
            eos_token_id = tokenizer.eos_token_id,
        )
    
    # Decode the initial response
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens = True) # Note: Special tokens does not include thinking tokens
    
    # Check if response ends with </think>
    if not ("</think>" in response):
        # Inject the answer tag and generate 50 more tokens
        injection = "</think>\n\n\\boxed{"
        injection_tokens = tokenizer(injection, add_special_tokens = False, return_tensors = "pt")['input_ids'].to(model.device)
        
        # Concatenate the outputs with injection tokens
        new_inputs = torch.cat([outputs, injection_tokens], dim = 1)
        
        # Generate 50 more tokens
        with torch.no_grad():
            final_outputs = model.generate(
                input_ids = new_inputs,
                max_new_tokens = 50,
                temperature = temperature,
                do_sample = temperature > 0,
                pad_token_id = tokenizer.eos_token_id,
                eos_token_id = tokenizer.eos_token_id,
            )
        
        # Decode the full response (excluding original prompt)
        full_generated_tokens = final_outputs[0][input_length:]
        response = tokenizer.decode(full_generated_tokens, skip_special_tokens = True)
    
    return response



def generate_evaluation_responses(model, tokenizer, test_data, engine, max_new_tokens: int = 2056, temperature: float = 0.7, vllm_lora_request = None, allow_thinking = True, force_think_token = False):
    """Generate evaluation responses one sample at a time (original version)."""
    # Store results
    results = []

    # Get responses
    if engine == 'vllm':
        # NOTE: Because with_chat_template = False, messages are returned and we don't need to pass allow_thinking
        messages = [prepare_question(example['question'], tokenizer, with_chat_template = False, allow_thinking = allow_thinking) for example in test_data]
        
        from vllm import SamplingParams as VLLMSamplingParams
        sampling_params = VLLMSamplingParams(
            n = 1,
            temperature = temperature,
            max_tokens = max_new_tokens
        )
        raw_responses = model.chat(
            messages = messages,
            sampling_params = sampling_params,
            lora_request = vllm_lora_request,
            use_tqdm = True,
            add_generation_prompt = allow_thinking,
            # Note: Use for Qwen3 models when want to enable/disable reasoning
            # enable_reasoning = True,
            # reasoning_parser="deepseek_r1"
            # continue_final_message = (not allow_thinking),
            # chat_template_kwargs={
            #     "enable_thinking": allow_thinking
            # }
        )
        raw_responses = [r.outputs[0].text for r in raw_responses]

        # # When using thinking, sometimes answers do not terminate
        # if allow_thinking:
        #     missing_answers = [(i, resp) for i, resp in enumerate(raw_responses) if "</think>" not in resp]
        #     if len(missing_answers) > 0:
        #         print(f"Forcing re-generation for {len(missing_answers)} questions")
        #         injected_messages = [messages[i] + [{"role": "assistant", "content": resp + "</think>\n\n\\boxed{"}] for i, resp in missing_answers]
        #         inj_raw_responses = model.chat(
        #                 messages = injected_messages,
        #                 sampling_params = sampling_params,
        #                 lora_request = vllm_lora_request,
        #                 use_tqdm = True,
        #                 add_generation_prompt = False,
        #                 continue_final_message = True,
        #                 chat_template_kwargs={
        #                     "enable_thinking": allow_thinking
        #                 }
        #         )
        #         inj_raw_responses = [r.outputs[0].text for r in inj_raw_responses]
        #         for (i, resp), ext_resp in zip(missing_answers, inj_raw_responses):
        #             raw_responses[i] = resp + ext_resp
        
    else:
        raw_responses = []
        for i, example in tqdm.tqdm(enumerate(test_data), total = len(test_data), desc = "Running evaluation..."):
            # Generate model output
            raw_response = generate_unsloth_answer(
                model, 
                tokenizer, 
                example['question'], 
                max_new_tokens = max_new_tokens,
                temperature = temperature
            )
            raw_responses.append(raw_response)
    
    # Process responses
    for i, (example, raw_response) in enumerate(zip(test_data, raw_responses)):
        # Extract COT and answer
        cot, answer = extract_thinking_and_answer(raw_response, allow_thinking = allow_thinking)

        formatted_answer = extract_final_number(answer, cot, allow_thinking = allow_thinking)
        
        results.append(Result(
            ind = i,
            question = example['question'],
            answer = example['answer'],
            response_cot = cot,
            response_answer = answer,
            response_answer_formatted = formatted_answer,
            response_full = raw_response,
            is_correct = formatted_answer == example['answer'],
            is_answered = len(formatted_answer) > 0
        ))
    
    return results

def evaluate_model(
        dataset: str,
        model_id: str,
        model_path: str,
        adapter_path: str = None,
        n_samples: int | None = None,
        max_new_tokens: int = 2056,
        temperature: float = 0.7,
        allow_thinking: bool = True,
        engine: Literal['vllm', 'unsloth'] = 'vllm'
    ):
    """Evaluate model on test set and save results."""

    thinking_suffix = '_no_thinking' if not allow_thinking else ''
    output_fpath = (model_path if adapter_path is None else adapter_path) + f'/evaluation_{dataset}_{n_samples}{thinking_suffix}.json'

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"Evaluating model: {model_path}")
    print(f"Base Model: {model_id}")
    if adapter_path:
        print(f"Using adapter: {adapter_path}")
    print(f"Allow thinking: {allow_thinking}")
    print(f"Inference engine: {engine}")
    print(f"Output file: {output_fpath}")
    print(f"{'='*60}\n")
    
    # Load model and tokenizer
    print("Loading model...")
    model, tokenizer, optional_lora_request = load_model_and_tokenizer(model_id, model_path, adapter_path, engine)
    
    # Load test data
    print(f"Loading test data")
    test_data = load_data(dataset = dataset, n_samples = n_samples)
    
    print(f"\nGenerating answers for {len(test_data)} test examples...\n")

    # Choose between batched and single-sample evaluation
    results = generate_evaluation_responses(
        model, 
        tokenizer, 
        test_data, 
        engine = engine,
        max_new_tokens = max_new_tokens, 
        temperature = temperature,
        allow_thinking = allow_thinking,
        force_think_token = False,
        vllm_lora_request = optional_lora_request
    )
    
    save_dataset(results, output_fpath)
    print(f"\n✅ Results saved to {output_fpath}")

    graceful_shutdown(model, tokenizer)


def graceful_shutdown(model, tokenizer):

    del model
    del tokenizer

    # Clear the cache
    gc.collect()
    torch.cuda.empty_cache()

    # Clear the cache
    try:
        torch.cuda.ipc_collect()
    except:
        pass

    try:
        # Then let PyTorch tear down the process group, if vLLM initialized it
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()  # or dist.shutdown() on recent PyTorch
    except AssertionError:
        pass
    
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError: 
        pass

    print("Successfully deleted the llm pipeline and free the GPU memory!")
