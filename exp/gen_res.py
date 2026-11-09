import torch
import os
import sys
import argparse
import json
import requests
import concurrent.futures
from pathlib import Path
from time import sleep
from tqdm import tqdm
from utils import load_dataset_from_file, save_dataset, make_api_request_with_retry, get_conversation_template
from vllm_compat import engine_kwargs, sampling_kwargs
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

################
# Configurations
################
def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="Response Generation Manager.")
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="We will support more models in the future.")
    parser.add_argument("--input_file", type=str, default=None, help="Input dataset file name")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of samples per batch")
    parser.add_argument("--checkpoint_every", type=int, default=20, help="Save checkpoint every n batches")
    parser.add_argument("--api_url", type=str, default="https://api.together.xyz/v1/chat/completions", help="API URL")
    parser.add_argument("--api_key", type=str, default=None, help="Together API Key")
    parser.add_argument("--offline", action="store_true", help="Use local engine")

    # Generation Parameters
    parser.add_argument('--engine', default="vllm", type=str, choices=["vllm", "hf", "together"])
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs to use for tensor parallelism. Only used for Llama 70B models.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling. Omit to disable.")
    parser.add_argument("--min_p", type=float, default=None, help="Min-p sampling. Omit to disable.")
    parser.add_argument("--presence_penalty", type=float, default=None, help="Presence penalty. Omit for 0.0.")
    parser.add_argument("--min_tokens", type=int, default=1,
                        help="Minimum completion length. 1 guards against empty responses.")
    parser.add_argument("--tokenizer_template", type=bool, default=True, help="Use tokenizer template for generating the response.")
    parser.add_argument("--use_tokenizer_template", action="store_true", dest="tokenizer_template")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Let reasoning models emit a <think> block. OFF by default: Qwen3.5/3.6's "
                             "chat template otherwise opens '<think>' and the answer arrives after a "
                             "long reasoning trace, which would then be truncated by --max_tokens. "
                             "When on, the trace is split off into a 'reasoning_content' field. "
                             "Ignored by chat templates that do not define the variable.")

    # Engine flags (see vllm_compat.engine_kwargs)
    parser.add_argument("--enable_prefix_caching", action="store_true",
                        help="Enable vLLM prefix caching. Pointless here (every prompt differs) and "
                             "risky on hybrid mamba/linear-attention models.")
    parser.add_argument("--enforce_eager", action="store_true",
                        help="Skip torch.compile and CUDA graph capture.")
    parser.add_argument("--disable_custom_all_reduce", action="store_true",
                        help="Disable vLLM's custom all-reduce (multi-GPU stability on some nodes).")
    parser.add_argument("--language_model_only", action="store_true",
                        help="For multimodal checkpoints, skip the vision tower. Text-only generation.")
    parser.add_argument("--gdn_prefill_backend", type=str, default=None,
                        choices=["auto", "triton", "flashinfer", "cutedsl"],
                        help="GatedDeltaNet prefill kernel for Qwen3.5/3.6-style hybrids. "
                             "'triton' skips the multi-minute JIT of the default FlashInfer kernel.")

    return parser.parse_args()

args = get_args()
print(f"Response Generation Manager. Arguments: {args}") # For logging

if args.input_file is None:
    raise ValueError("Please specify the input file path.")

# Constants for the local vllm engine
MODEL_NAME = args.model_path
INPUT_FILE_NAME = args.input_file 
BATCH_SIZE = args.batch_size
CHECKPOINT_FILE = f"{INPUT_FILE_NAME[:INPUT_FILE_NAME.rfind('.')]}_res_checkpoint.json"
CHECKPOINT_EVERY = args.checkpoint_every
SAVED_FILE = f"{INPUT_FILE_NAME[:INPUT_FILE_NAME.rfind('.')]}_res.json"

# Passed to apply_chat_template. Always sent explicitly, because a reasoning model's
# template defaults to thinking ON: omitting the variable is not the same as False.
TEMPLATE_KWARGS = {"enable_thinking": args.enable_thinking}


def split_reasoning(text):
    """Split a completion into (reasoning_content, response).

    With thinking enabled, the generation prompt ends mid-``<think>`` block, so the
    trace arrives first and the answer follows ``</think>``. The reasoning is kept in
    its own field rather than left inline, so downstream consumers get a clean
    instruction/response pair either way.
    """
    text = text.strip()
    if "</think>" not in text:
        # Either thinking is off, or the trace was cut off by --max_tokens before it
        # closed. In the latter case there is no answer to salvage.
        return "", text
    reasoning, _, response = text.partition("</think>")
    return reasoning.removeprefix("<think>").strip(), response.strip()

# Obtain config from configs/model_configs.json
with open("../configs/model_configs.json", "r") as f:
    model_configs = json.load(f)
    model_config = model_configs[args.model_path]
    stop_tokens = model_config["stop_tokens"]
    stop_token_ids = model_config["stop_token_ids"]

# API Setups
if args.engine == "together":
    # Change name for API (Together Naming Convention)
    if MODEL_NAME == "meta-llama/Meta-Llama-3-8B-Instruct":
        api_model_name = "meta-llama/Llama-3-8b-chat-hf"
    elif MODEL_NAME == "meta-llama/Meta-Llama-3-70B-Instruct":
        api_model_name = "meta-llama/Llama-3-70b-chat-hf"
    else:
        api_model_name = MODEL_NAME

    # Constants for the API
    API_ENDPOINT = args.api_url
    API_HEADERS = {
        "Authorization": args.api_key,
    }
    API_PARAMS = {
        "model": api_model_name,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "stop": stop_tokens
    }

# Process a batch of data using the API
def process_batch_with_api(batch):
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_item = {
            executor.submit(
                make_api_request_with_retry, 
                [{'content': item['instruction'], 'role': 'user'}],
                API_PARAMS,
                API_ENDPOINT,
                API_HEADERS,
            ): item 
            for item in batch
        }

        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                api_response = future.result()
                item['response'] = api_response.strip()
                item['gen_response_configs'] = {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                    "max_tokens": args.max_tokens,
                    "stop_tokens": stop_tokens,
                    "output_generator": MODEL_NAME,
                    "engine": api_model_name,
                }
            except Exception as e:
                print(f"Failed to process item: {item} with error: {str(e)}")
                item['response'] = ""
                
    return batch

_PROMPT_CHECKED = False


def check_prompt_shape(prompt):
    """Print the first rendered prompt, and fail loudly on the silent-corruption case.

    If a reasoning model's template leaves an *open* ``<think>`` while we believe
    thinking is off, generation succeeds and looks fine -- but what lands in 'response'
    is the reasoning trace, usually truncated mid-thought. That has burned whole result
    sets before, so it is worth one assertion at the top of the run rather than a
    post-mortem afterwards.
    """
    global _PROMPT_CHECKED
    if _PROMPT_CHECKED:
        return
    _PROMPT_CHECKED = True
    print(f"[gen_res.py] First rendered prompt:\n{prompt!r}")
    if args.enable_thinking:
        return
    # Only meaningful for templates that emit a think block at all.
    if "<think>" in prompt and not prompt.rstrip().endswith("</think>"):
        raise ValueError(
            "enable_thinking is False but the rendered prompt leaves an OPEN <think> "
            "block, so the model will emit a reasoning trace and it would be recorded "
            f"as the response. Rendered tail: {prompt[-60:]!r}. Check that the "
            "tokenizer's chat template honours enable_thinking."
        )


# Process a batch of data using local vllm engine
def process_batch(batch, llm, params, tokenizer=None):
    user_instructions = [item['instruction'] for item in batch]
    prompts = []
    for instruction in user_instructions:
        if not args.tokenizer_template:
            conv = get_conversation_template(MODEL_NAME)
            conv.append_message(conv.roles[0], instruction)
            conv.append_message(conv.roles[1], None)
            template = conv.get_prompt()
        else:
            chat = [{"role": "user", "content": instruction}]
            template = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True,
                **TEMPLATE_KWARGS)
        prompts.append(template)
    check_prompt_shape(prompts[0])
    if args.engine == "vllm":
        outputs = llm.generate(prompts, params)
    elif args.engine == "hf":
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(torch.cuda.current_device())
        gen_do_sample = False if args.temperature == 0 else True
        outputs = llm.generate(**inputs,
                tokenizer=tokenizer, 
                do_sample=gen_do_sample, 
                temperature=args.temperature if gen_do_sample else None, # To avoid temperature` (=0) has to be a strictly positive float
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty, 
                max_length=args.max_tokens,
                )
        outputs = tokenizer.batch_decode(outputs[i][len(inputs[i]):] for i in range(len(outputs)))
        # Setting stop tokens seems not working for Gemma, so we manually truncate the outputs
        for i, completion in enumerate(outputs):
            for stop_token in stop_tokens:
                if stop_token in completion:
                    outputs[i] = completion[:completion.index(stop_token)]

    for i, item in enumerate(batch):
        finish_reason = None
        if args.engine == "vllm":
            completion = outputs[i].outputs[0]
            raw = completion.text
            # Recorded because a finish_reason of "length" across many items is the
            # signal that max_tokens is too small (or that thinking is unexpectedly on),
            # which otherwise shows up only as mysteriously truncated answers.
            finish_reason = getattr(completion, "finish_reason", None)
        elif args.engine == "hf":
            raw = outputs[i]
        reasoning, response = split_reasoning(raw)
        item['response'] = response
        # Present in both modes so the two share one schema; empty when thinking is off.
        item['reasoning_content'] = reasoning
        item['finish_reason'] = finish_reason
        item['gen_response_configs'] = {
            "prompt": prompts[i],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "max_tokens": args.max_tokens,
            "stop_tokens": stop_tokens,
            "enable_thinking": args.enable_thinking,
            "output_generator": MODEL_NAME,
            "engine": args.engine,
        }
    return batch

# Generate outputs, update dataset in batches, and overwrite checkpoint
def generate_and_update(dataset, llm=None, params=None, tokenizer=None):
    # Initialize tokenizer
    if tokenizer is not None:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if "gemma-2" in args.model_path.lower():
            tokenizer.padding_side = "right"

    # Intialize the dataset with the checkpoint file (if it exists)
    if os.path.exists(CHECKPOINT_FILE):
        last_checkpoint_idx = len(load_dataset_from_file(CHECKPOINT_FILE))
        print(f"Checkpoint file found. Resuming from last checkpoint with index {last_checkpoint_idx}.")
        dataset[:last_checkpoint_idx] = load_dataset_from_file(CHECKPOINT_FILE)
        # Calculate total number of batches
        num_batches = (len(dataset) - last_checkpoint_idx + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"Remaining number of batches: {num_batches}")
    else:
        last_checkpoint_idx = 0
        # Calculate total number of batches
        num_batches = (len(dataset) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Total number of batches: {num_batches}")

    for i in tqdm(range(num_batches)):
        start_idx = i * BATCH_SIZE + last_checkpoint_idx
        end_idx = min((i + 1) * BATCH_SIZE + last_checkpoint_idx, len(dataset))
        batch = dataset[start_idx:end_idx]
        if args.engine == "together":
            batch = process_batch_with_api(batch)
        else:
            batch = process_batch(batch, llm, params, tokenizer)
        
        dataset[start_idx:end_idx] = batch
        # Overwrite the same checkpoint file after serveral batches
        if i % CHECKPOINT_EVERY == 0:
            save_dataset(dataset[:end_idx], CHECKPOINT_FILE)
            print(f"Dataset checkpoint saved after batch {i + 1}.")

    return dataset

# Main function to control workflow
def main():
    # Load instructions from the input file
    dataset = load_dataset_from_file(INPUT_FILE_NAME)
    
    if args.engine == "together":
        print("Start together API engine...")
        llm = None
        params = None
        tokenizer = None
    elif args.engine == "vllm":
        # Set the device
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        print("Start Local vllm engine...")
        llm = LLM(model=MODEL_NAME,
            dtype=args.dtype,
            trust_remote_code=True,
            max_model_len = args.max_model_len, # limited by kv-cache
            tensor_parallel_size = args.tensor_parallel_size,
            gpu_memory_utilization = args.gpu_memory_utilization,
            enable_prefix_caching=args.enable_prefix_caching,
            **engine_kwargs(args))

        # repetition_penalty comes via sampling_kwargs (it has a non-None default here),
        # so it must not also be passed explicitly.
        params = SamplingParams(
            max_tokens=args.max_tokens,
            min_tokens=args.min_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stop_token_ids=stop_token_ids,
            **sampling_kwargs(args),
            )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    elif args.engine == "hf":
        print("Start Hugging Face engine...")
        params = None
        # Load the model and tokenizer
        llm = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map={'':torch.cuda.current_device()},
            torch_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    else:
        raise ValueError("Invalid engine type.")

    updated_dataset = generate_and_update(dataset, llm, params, tokenizer=tokenizer)

    # Save final dataset
    save_dataset(updated_dataset, SAVED_FILE)

    # Summarise the two ways this stage fails quietly rather than loudly: truncated
    # completions (finish_reason == "length"), and reasoning-block leakage. In
    # non-thinking mode a non-zero leak count is expected to be tiny but non-zero --
    # the model occasionally emits </think> spontaneously -- and split_reasoning has
    # already moved that text out of 'response'.
    from collections import Counter
    total = len(updated_dataset)
    finish = Counter(item.get('finish_reason') for item in updated_dataset)
    truncated = finish.get('length', 0)
    empty = sum(1 for item in updated_dataset if not (item.get('response') or "").strip())
    leaked = sum(1 for item in updated_dataset if (item.get('reasoning_content') or "").strip())
    print(f"[gen_res.py] {total} responses | finish_reason={dict(finish)}")
    print(f"[gen_res.py] truncated (finish_reason=length): {truncated} ({truncated / max(total, 1):.1%})")
    print(f"[gen_res.py] empty responses: {empty} ({empty / max(total, 1):.1%})")
    print(f"[gen_res.py] reasoning blocks split off: {leaked} "
          f"(enable_thinking={args.enable_thinking})")
    if truncated / max(total, 1) > 0.10:
        print("[gen_res.py] WARNING: >10% truncated. Raise --max_tokens, or check that "
              "thinking really is disabled -- with an open <think> block this model "
              "truncates essentially every completion.")

    # Optionally remove the checkpoint file after completion. missing_ok because a run
    # short enough to finish inside the first checkpoint interval never writes one.
    Path(CHECKPOINT_FILE).unlink(missing_ok=True)
    print("Final dataset saved. Checkpoint removed.")

# Run the main function
if __name__ == "__main__":
    main()